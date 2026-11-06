"""Supervisor: que se aprobo, que sigue, guardas, contexto que ve y cierre."""

import pytest
from langchain_core.exceptions import OutputParserException
from langchain_core.messages import HumanMessage, SystemMessage

from state import CLAVES_POR_NODO, Contribucion, EstadoOrquestador, ResultadoValidacion
from supervisor import (
    PROMPT_SUPERVISOR,
    aplicar_guardas,
    aprobado,
    cerrar,
    crear_nodo_supervisor,
    hay_que_analizar,
    requisitos_cumplidos,
    resumen_del_trabajo,
    siguiente_logico,
)
from tests.conftest import crear_estado
from tests.fakes import SupervisorFalso, analisis_de_datos, decidir, hallazgos_de_datos


def _val(agente: str, aprobada: bool, paso: int, *problemas: str) -> ResultadoValidacion:
    return ResultadoValidacion(agente=agente, aprobada=aprobada, problemas=list(problemas), paso=paso)  # type: ignore[arg-type]


def _estado(aprobados: tuple[str, ...] = (), *, hallazgos: bool = True, **cambios: object) -> EstadoOrquestador:
    """Estado con los agentes dados ya aprobados, en orden (paso 1, 2, 3...)."""
    validaciones = [_val(agente, True, paso) for paso, agente in enumerate(aprobados, start=1)]
    base = hallazgos_de_datos()
    datos: dict[str, object] = {
        "hallazgos": base if hallazgos else None,
        "analisis": analisis_de_datos(base) if "analista" in aprobados else None,
        "validaciones": validaciones,
        "validacion": validaciones[-1] if validaciones else None,
        "pasos": len(aprobados),
    }
    datos.update(cambios)  # lo que pida el test pisa los valores por defecto
    return crear_estado(**datos)


# ---------- aprobado / siguiente_logico / requisitos ----------


def test_nada_esta_aprobado_al_principio() -> None:
    estado = crear_estado()
    assert not any(aprobado(estado, a) for a in ("investigador", "analista", "redactor"))  # type: ignore[arg-type]
    assert siguiente_logico(estado) == "investigador"


def test_un_aporte_rechazado_no_cuenta_como_aprobado() -> None:
    estado = crear_estado(validaciones=[_val("investigador", False, 1, "falta algo")])
    assert not aprobado(estado, "investigador") and siguiente_logico(estado) == "investigador"


def test_el_orden_logico_es_investigador_analista_redactor_cierre() -> None:
    assert siguiente_logico(_estado(("investigador",))) == "analista"
    assert siguiente_logico(_estado(("investigador", "analista"))) == "redactor"
    assert siguiente_logico(_estado(("investigador", "analista", "redactor"))) == "FINISH"


def test_si_la_consulta_esta_fuera_de_alcance_se_salta_al_analista() -> None:
    from state import Hallazgos

    estado = crear_estado(hallazgos=Hallazgos(fuera_de_alcance=True), validaciones=[_val("investigador", True, 1)])
    assert hay_que_analizar(estado) is False
    assert siguiente_logico(estado) == "redactor"


def test_si_no_hay_incidentes_tambien_se_salta_al_analista() -> None:
    from state import Hallazgos

    estado = crear_estado(hallazgos=Hallazgos(herramientas_usadas=["buscar_incidentes_calidad"]), validaciones=[_val("investigador", True, 1)])
    assert siguiente_logico(estado) == "redactor"


def test_un_aporte_aprobado_deja_de_valer_si_un_agente_anterior_vuelve_a_entregar() -> None:
    """El analista calculo sobre datos viejos: si el investigador rehace su trabajo, hay que reanalizar."""
    validaciones = [_val("investigador", True, 1), _val("analista", True, 2), _val("investigador", True, 3)]
    estado = crear_estado(hallazgos=hallazgos_de_datos(), validaciones=validaciones)
    assert aprobado(estado, "investigador") is True
    assert aprobado(estado, "analista") is False
    assert siguiente_logico(estado) == "analista"


def test_requisitos_de_cada_agente() -> None:
    assert requisitos_cumplidos("investigador", crear_estado()) is True
    assert requisitos_cumplidos("analista", crear_estado()) is False  # sin hallazgos aprobados
    assert requisitos_cumplidos("redactor", crear_estado()) is False
    despues_del_investigador = _estado(("investigador",))
    assert requisitos_cumplidos("analista", despues_del_investigador) is True
    assert requisitos_cumplidos("redactor", despues_del_investigador) is False  # falta el analisis
    assert requisitos_cumplidos("redactor", _estado(("investigador", "analista"))) is True
    assert requisitos_cumplidos("FINISH", _estado(("investigador", "analista", "redactor"))) is False  # FINISH no es un agente


# ---------- guardas ----------


@pytest.mark.parametrize(
    ("aprobados", "elige", "esperado", "corregida"),
    [
        ((), "investigador", "investigador", False),
        (("investigador",), "analista", "analista", False),
        (("investigador", "analista"), "redactor", "redactor", False),
        (("investigador", "analista", "redactor"), "FINISH", "FINISH", False),
        ((), "FINISH", "investigador", True),  # cierre prematuro
        ((), "analista", "investigador", True),  # salta un paso del que depende
        ((), "redactor", "investigador", True),
        (("investigador",), "redactor", "analista", True),  # redactar sin analizar
        (("investigador", "analista"), "FINISH", "redactor", True),  # cerrar sin informe aprobado
    ],
)
def test_las_guardas_solo_corrigen_lo_que_rompe_las_dependencias(
    aprobados: tuple[str, ...], elige: str, esperado: str, corregida: bool
) -> None:
    final, razon = aplicar_guardas(elige, _estado(aprobados), max_pasos=10, max_intentos=3)  # type: ignore[arg-type]
    assert final == esperado
    assert (razon is not None) is corregida


def test_el_supervisor_puede_elegir_rehacer_un_aporte_aprobado_mientras_queden_intentos() -> None:
    """La guarda no fuerza el orden: acota solo lo inseguro. Volver a consultar es una decision valida."""
    final, razon = aplicar_guardas("investigador", _estado(("investigador", "analista")), max_pasos=10, max_intentos=3)
    assert (final, razon) == ("investigador", None)


def test_al_llegar_al_tope_de_pasos_se_cierra_siempre() -> None:
    estado = _estado(("investigador",), pasos=10)
    final, razon = aplicar_guardas("analista", estado, max_pasos=10, max_intentos=3)
    assert final == "FINISH" and "límite de 10 pasos" in (razon or "")


def test_en_el_tope_de_pasos_cerrar_con_todo_aprobado_no_es_una_correccion() -> None:
    estado = _estado(("investigador", "analista", "redactor"), pasos=10)
    assert aplicar_guardas("FINISH", estado, max_pasos=10, max_intentos=3) == ("FINISH", None)


def test_un_agente_sin_intentos_se_reemplaza_por_el_siguiente_paso_logico() -> None:
    estado = _estado(("investigador",), intentos={"investigador": 3, "analista": 0})
    final, razon = aplicar_guardas("investigador", estado, max_pasos=10, max_intentos=3)
    assert final == "analista" and "ya usó sus 3 intentos" in (razon or "")


def test_si_el_paso_pendiente_se_quedo_sin_intentos_el_sistema_cierra() -> None:
    estado = _estado(("investigador",), intentos={"investigador": 1, "analista": 3})
    final, razon = aplicar_guardas("analista", estado, max_pasos=10, max_intentos=3)
    assert final == "FINISH" and "sin lograr un aporte aprobado" in (razon or "")


# ---------- lo que ve el supervisor ----------


def test_el_supervisor_ve_la_tarea_los_pasos_y_el_veredicto_de_cada_agente() -> None:
    estado = _estado(
        ("investigador",),
        contribuciones=[Contribucion(agente="investigador", paso=1, resumen="Consulté 2 incidentes.")],
    )
    texto = resumen_del_trabajo(estado, max_pasos=10, max_intentos=3)
    assert "TAREA DEL USUARIO" in texto and "1 de 10" in texto
    assert "investigador: intervino 1 vez/veces. Última validación: APROBADO." in texto
    assert "analista: todavía no intervino." in texto
    assert "2 incidente(s) con ficha completa" in texto


def test_el_supervisor_ve_los_problemas_de_un_aporte_rechazado() -> None:
    estado = crear_estado(
        hallazgos=hallazgos_de_datos(),
        contribuciones=[Contribucion(agente="analista", paso=2, resumen="Calculé.")],
        validaciones=[_val("analista", False, 2, "Falta el ranking de prioridad.")],
    )
    texto = resumen_del_trabajo(estado, max_pasos=10, max_intentos=3)
    assert "RECHAZADO" in texto and "Problema detectado: Falta el ranking de prioridad." in texto


def test_el_supervisor_no_recibe_datos_crudos_ni_el_historial_de_mensajes() -> None:
    """Contexto acotado: el resumen no incluye la ficha de los incidentes ni los mensajes del grafo."""
    estado = _estado(("investigador", "analista"), contribuciones=[Contribucion(agente="analista", paso=2, resumen="ok")])
    texto = resumen_del_trabajo(estado, max_pasos=10, max_intentos=3)
    for dato_crudo in ("registros_afectados", "Laura Perez", "Gerente Comercial", "sla_horas_habiles", "fecha_alta"):
        assert dato_crudo not in texto


def test_un_resumen_muy_largo_se_recorta() -> None:
    largo = Contribucion(agente="investigador", paso=1, resumen="palabra " * 500)
    texto = resumen_del_trabajo(_estado(("investigador",), contribuciones=[largo]), max_pasos=10, max_intentos=3)
    assert max(len(linea) for linea in texto.splitlines()) < 500


def test_el_prompt_define_al_supervisor_como_director_de_trafico_con_una_rubrica() -> None:
    assert "DIRECTOR DE TRÁFICO" in PROMPT_SUPERVISOR
    assert "no recolectás datos, no\ncalculás y no redactás" in PROMPT_SUPERVISOR
    for palabra in ("APROBADO", "RECHAZÓ", "FINISH", "fuera\n   de alcance"):
        assert palabra in PROMPT_SUPERVISOR


# ---------- cierre ----------


def test_cierre_completado_devuelve_el_informe_aprobado() -> None:
    estado = _estado(("investigador", "analista", "redactor"), informe="El informe final.")
    assert cerrar(estado, max_pasos=10) == {"estado_final": "completado", "respuesta_final": "El informe final."}


def test_cierre_por_limite_de_pasos_no_entrega_un_informe_sin_aprobar() -> None:
    estado = _estado(("investigador",), pasos=10, informe="Borrador sin aprobar")
    cierre = cerrar(estado, max_pasos=10)
    assert cierre["estado_final"] == "limite_de_pasos"
    assert "Borrador" not in cierre["respuesta_final"] and "límite de 10 pasos" in cierre["respuesta_final"]


def test_cierre_por_validacion_fallida_incluye_los_problemas_pendientes() -> None:
    estado = crear_estado(
        hallazgos=hallazgos_de_datos(), pasos=4,
        validaciones=[_val("analista", False, 4, "Falta el ranking de prioridad.")],
        validacion=_val("analista", False, 4, "Falta el ranking de prioridad."),
    )
    cierre = cerrar(estado, max_pasos=10)
    assert cierre["estado_final"] == "validacion_fallida"
    assert "Falta el ranking de prioridad." in cierre["respuesta_final"]


# ---------- el nodo ----------


async def test_el_nodo_escribe_solo_su_parcela_y_registra_la_decision() -> None:
    supervisor = SupervisorFalso(guion=[decidir("investigador", "Recolectá los críticos.", "arranco por los datos")])
    nodo = crear_nodo_supervisor(supervisor, max_pasos=10, max_intentos=3)
    salida = await nodo(crear_estado(), {})
    assert set(salida) <= CLAVES_POR_NODO["supervisor"]
    assert salida["siguiente"] == "investigador" and salida["pasos"] == 1 and salida["intentos"] == {"investigador": 1}
    registro = salida["decisiones"][0]
    assert registro.instruccion == "Recolectá los críticos." and registro.motivo == "arranco por los datos"
    assert registro.corregida_por_guarda is False
    contexto = supervisor.recibidos[0]
    assert isinstance(contexto[0], SystemMessage) and isinstance(contexto[1], HumanMessage) and len(contexto) == 2


async def test_si_la_guarda_cambia_el_destino_la_instruccion_pasa_a_ser_la_del_nuevo_agente() -> None:
    supervisor = SupervisorFalso(guion=[decidir("redactor", "Escribí el informe ya mismo.")])
    nodo = crear_nodo_supervisor(supervisor, max_pasos=10, max_intentos=3)
    salida = await nodo(crear_estado(), {})
    assert salida["siguiente"] == "investigador"
    assert "Escribí el informe" not in salida["instruccion"] and "Recolectá" in salida["instruccion"]
    assert salida["decisiones"][0].corregida_por_guarda is True


async def test_una_instruccion_vacia_se_reemplaza_por_la_instruccion_por_defecto() -> None:
    nodo = crear_nodo_supervisor(SupervisorFalso(guion=[decidir("investigador", "")]), max_pasos=10, max_intentos=3)
    assert "Recolectá los datos" in (await nodo(crear_estado(), {}))["instruccion"]


async def test_una_salida_invalida_del_modelo_no_rompe_el_nodo_y_sigue_el_orden_logico() -> None:
    supervisor = SupervisorFalso(guion=[OutputParserException("no es una decision")])
    salida = await crear_nodo_supervisor(supervisor, max_pasos=10, max_intentos=3)(crear_estado(), {})
    assert salida["siguiente"] == "investigador"
    registro = salida["decisiones"][0]
    assert registro.corregida_por_guarda and "no tenía la forma pedida" in (registro.razon_guarda or "")


async def test_al_cerrar_el_nodo_deja_estado_final_y_respuesta_final() -> None:
    estado = _estado(("investigador", "analista", "redactor"), informe="Informe listo.")
    salida = await crear_nodo_supervisor(SupervisorFalso(guion=[decidir("FINISH")]), max_pasos=10, max_intentos=3)(estado, {})
    assert salida["estado_final"] == "completado" and salida["respuesta_final"] == "Informe listo."
    assert salida["instruccion"] == "" and salida["intentos"] == {}


def test_si_declaro_fuera_de_alcance_no_se_analiza_aunque_haya_incidentes_detallados() -> None:
    """La bandera manda: un investigador que consulto datos pero dijo FUERA_DE_ALCANCE no habilita al analista."""
    hallazgos = hallazgos_de_datos().model_copy(update={"fuera_de_alcance": True})
    estado = crear_estado(hallazgos=hallazgos, validaciones=[_val("investigador", True, 1)])
    assert hay_que_analizar(estado) is False
    assert requisitos_cumplidos("analista", estado) is False
    assert siguiente_logico(estado) == "redactor"


async def test_al_cerrar_la_instruccion_se_vacia_aunque_el_modelo_haya_escrito_una() -> None:
    estado = _estado(("investigador", "analista", "redactor"), informe="Informe listo.")
    supervisor = SupervisorFalso(guion=[decidir("FINISH", "Instrucción que no corresponde")])
    salida = await crear_nodo_supervisor(supervisor, max_pasos=10, max_intentos=3)(estado, {})
    assert salida["instruccion"] == ""


async def test_cerrar_no_consume_un_paso() -> None:
    estado = _estado(("investigador", "analista", "redactor"), informe="Informe listo.")
    salida = await crear_nodo_supervisor(SupervisorFalso(guion=[decidir("FINISH")]), max_pasos=10, max_intentos=3)(estado, {})
    assert salida["pasos"] == 3 and salida["decisiones"][0].paso == 3
