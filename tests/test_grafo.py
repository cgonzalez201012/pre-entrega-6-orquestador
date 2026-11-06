"""El grafo completo con modelos falsos: delegacion, refinamiento, topes y aislamiento.

Las herramientas se ejecutan de verdad; lo unico simulado es la decision de cada modelo.
"""

from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.tools import BaseTool

from graph import (
    ESPECIALISTAS,
    NODO_ANALISTA,
    NODO_INVESTIGADOR,
    NODO_REDACTOR,
    NODO_SUPERVISOR,
    NODO_VALIDACION,
    construir_grafo,
    decidir_ruta,
)
from state import CLAVES_POR_NODO, FUERA_DE_ALCANCE, estado_inicial
from tests.conftest import TAREA, correr
from tests.fakes import (
    SupervisorFalso,
    decidir,
    especialistas_falsos,
    guion_analista,
    guion_investigador,
    informe,
    pedir,
    pedir_varias,
    responder,
)

INFORME_OK = informe("INC-1001", "INC-1002", "R-ART-01")


def _supervisor_ordenado() -> SupervisorFalso:
    return SupervisorFalso(
        guion=[
            decidir("investigador", "Recolectá los críticos abiertos de artículos."),
            decidir("analista", "Calculá vencimientos, brechas y ranking."),
            decidir("redactor", "Escribí el informe."),
            decidir("FINISH"),
        ]
    )


async def _recorrido(supervisor: SupervisorFalso, especialistas: Any, *, max_pasos: int = 10, max_intentos: int = 3) -> list[tuple[str, dict[str, Any]]]:
    """Lista (nodo, actualizacion) en el orden en que se ejecutaron los nodos."""
    grafo = construir_grafo(supervisor, especialistas, max_pasos=max_pasos, max_intentos=max_intentos, recursion_especialista=12)
    pasos: list[tuple[str, dict[str, Any]]] = []
    async for cambio in grafo.astream(estado_inicial(TAREA), {"recursion_limit": 3 * max_pasos + 5}, stream_mode="updates"):
        for nodo, actualizacion in cambio.items():
            pasos.append((nodo, actualizacion or {}))
    return pasos


# ---------- Estructura ----------


def test_el_grafo_tiene_los_cinco_nodos_y_las_aristas_esperadas() -> None:
    grafo = construir_grafo(_supervisor_ordenado(), especialistas_falsos(), max_pasos=10, max_intentos=3, recursion_especialista=12)
    dibujo = grafo.get_graph()
    assert {n for n in dibujo.nodes} - {"__start__", "__end__"} == {
        NODO_SUPERVISOR, NODO_INVESTIGADOR, NODO_ANALISTA, NODO_REDACTOR, NODO_VALIDACION,
    }
    aristas = {(a.source, a.target, a.conditional) for a in dibujo.edges}
    assert ("__start__", NODO_SUPERVISOR, False) in aristas
    assert (NODO_VALIDACION, NODO_SUPERVISOR, False) in aristas
    for especialista in ESPECIALISTAS:
        assert (NODO_SUPERVISOR, especialista, True) in aristas  # solo el supervisor decide
        assert (especialista, NODO_VALIDACION, False) in aristas  # y todos pasan por validacion
    assert (NODO_SUPERVISOR, "__end__", True) in aristas


def test_no_hay_rutas_fijas_entre_especialistas() -> None:
    grafo = construir_grafo(_supervisor_ordenado(), especialistas_falsos(), max_pasos=10, max_intentos=3, recursion_especialista=12)
    for a in grafo.get_graph().edges:
        assert not (a.source in ESPECIALISTAS and a.target in ESPECIALISTAS)


def test_decidir_ruta_devuelve_el_destino_que_dejo_el_supervisor() -> None:
    for destino in ("investigador", "analista", "redactor", "FINISH"):
        assert decidir_ruta({"siguiente": destino}) == destino  # type: ignore[typeddict-item]


# ---------- Camino feliz ----------


async def test_camino_feliz_delega_en_orden_y_valida_cada_aporte() -> None:
    especialistas = especialistas_falsos(guion_investigador(), guion_analista(), [INFORME_OK])
    recorrido = await _recorrido(_supervisor_ordenado(), especialistas)
    assert [nodo for nodo, _ in recorrido] == [
        "supervisor", "investigador", "validacion",
        "supervisor", "analista", "validacion",
        "supervisor", "redactor", "validacion",
        "supervisor",
    ]


async def test_camino_feliz_estado_final() -> None:
    especialistas = especialistas_falsos(guion_investigador(), guion_analista(), [INFORME_OK])
    final = await correr(_supervisor_ordenado(), especialistas)
    assert final["estado_final"] == "completado"
    assert final["respuesta_final"] == INFORME_OK.content
    assert final["pasos"] == 3  # 3 delegaciones: cerrar no consume un paso
    assert final["analisis"].ranking == ["INC-1001", "INC-1002"]
    assert [v.agente for v in final["validaciones"]] == ["investigador", "analista", "redactor"]
    assert all(v.aprobada for v in final["validaciones"])
    assert [c.agente for c in final["contribuciones"]] == ["investigador", "analista", "redactor"]
    assert [d.siguiente for d in final["decisiones"]] == ["investigador", "analista", "redactor", "FINISH"]


async def test_cada_especialista_usa_sus_herramientas_y_ninguna_de_las_del_otro() -> None:
    especialistas = especialistas_falsos(guion_investigador(), guion_analista(), [INFORME_OK])
    final = await correr(_supervisor_ordenado(), especialistas)
    assert set(final["hallazgos"].herramientas_usadas) == {"buscar_incidentes_calidad", "detalle_incidente", "consultar_regla_calidad"}
    assert set(final["analisis"].herramientas_usadas) == {"calcular_vencimiento_sla", "calcular_brecha_regla", "priorizar_incidentes"}
    llamadas = [m for m in final["messages"] if getattr(m, "tool_calls", None)]
    assert llamadas == []  # el historial del grafo no se llena con los mensajes internos de los especialistas


async def test_la_delegacion_incluye_al_menos_dos_llamadas_a_herramientas_en_secuencia() -> None:
    """Evidencia pedida por el curso: el flujo investigador -> analista encadena herramientas reales."""
    especialistas = especialistas_falsos(guion_investigador(), guion_analista(), [INFORME_OK])
    final = await correr(_supervisor_ordenado(), especialistas)
    assert len(final["hallazgos"].herramientas_usadas) + len(final["analisis"].herramientas_usadas) >= 2
    assert final["analisis"].vencimientos[0].exceso_horas == 276  # calculado con codigo, a partir de la ficha del investigador


# ---------- Rutas alternativas ----------


async def test_sin_incidentes_el_supervisor_se_salta_al_analista() -> None:
    investigador = [
        pedir("buscar_incidentes_calidad", maestro="sucursales", estado="abierto", severidad="critica"),
        responder("No hay incidentes críticos abiertos en sucursales."),
    ]
    supervisor = SupervisorFalso(guion=[decidir("investigador"), decidir("redactor"), decidir("FINISH")])
    especialistas = especialistas_falsos(investigador, [], [responder("No hay incidentes críticos abiertos en sucursales.")])
    recorrido = await _recorrido(supervisor, especialistas)
    assert NODO_ANALISTA not in [n for n, _ in recorrido]
    final = await correr(
        SupervisorFalso(guion=[decidir("investigador"), decidir("redactor"), decidir("FINISH")]),
        especialistas_falsos(investigador, [], [responder("No hay incidentes críticos abiertos en sucursales.")]),
    )
    assert final["estado_final"] == "completado" and final["analisis"] is None


async def test_si_la_consulta_esta_fuera_de_alcance_se_responde_sin_inventar_datos() -> None:
    investigador = [responder(f"{FUERA_DE_ALCANCE}: no tengo datos de empleados ni de sueldos.")]
    redactor = [responder("No puedo responder eso: el sistema solo tiene incidentes y reglas de calidad de datos.")]
    final = await correr(
        SupervisorFalso(guion=[decidir("investigador"), decidir("redactor"), decidir("FINISH")]),
        especialistas_falsos(investigador, [], redactor),
        tarea="¿Cuánto cobra el gerente de datos?",
    )
    assert final["estado_final"] == "completado" and final["hallazgos"].fuera_de_alcance
    assert final["analisis"] is None and "No puedo responder" in final["respuesta_final"]


# ---------- Refinamiento y conflictos entre agentes ----------


async def test_si_el_analista_calcula_mal_la_validacion_lo_rechaza_y_el_supervisor_pide_rehacerlo() -> None:
    mal = guion_analista(alterar={"INC-1001": {"fecha_alta": "2026-10-01"}})
    bien = guion_analista()
    supervisor = SupervisorFalso(
        guion=[
            decidir("investigador"),
            decidir("analista", "Calculá."),
            decidir("analista", "Rehacé el vencimiento de INC-1001: la fecha de alta no coincide con la ficha."),
            decidir("redactor"),
            decidir("FINISH"),
        ]
    )
    especialistas = especialistas_falsos(guion_investigador(), [*mal, *bien], [INFORME_OK])
    final = await correr(supervisor, especialistas)
    veredictos = [(v.agente, v.aprobada) for v in final["validaciones"]]
    assert veredictos == [("investigador", True), ("analista", False), ("analista", True), ("redactor", True)]
    assert final["estado_final"] == "completado" and final["intentos"]["analista"] == 2
    assert final["validaciones"][1].problemas and "INC-1001" in final["validaciones"][1].problemas[0]


async def test_la_correccion_llega_al_agente_como_un_unico_mensaje_con_el_problema_concreto() -> None:
    mal = guion_analista(alterar={"INC-1001": {"fecha_alta": "2026-10-01"}})
    especialistas = especialistas_falsos(guion_investigador(), [*mal, *guion_analista()], [INFORME_OK])
    supervisor = SupervisorFalso(
        guion=[decidir("investigador"), decidir("analista"), decidir("analista", "Rehacé INC-1001."), decidir("redactor"), decidir("FINISH")]
    )
    await correr(supervisor, especialistas)
    segundo_intento = [m for m in especialistas.recibidos if "Agente Analista" in str(m[0].content)][-3]
    humanos = [m for m in segundo_intento if isinstance(m, HumanMessage)]
    assert len(humanos) == 1  # contexto aislado: un solo mensaje de entrada
    assert "Rehacé INC-1001." in str(humanos[0].content) and "no coincide" in str(humanos[0].content)


async def test_si_un_agente_agota_sus_intentos_el_sistema_cierra_con_validacion_fallida() -> None:
    mal = guion_analista(alterar={"INC-1001": {"fecha_alta": "2026-10-01"}})
    especialistas = especialistas_falsos(guion_investigador(), [*mal, *mal, *mal], [])
    supervisor = SupervisorFalso(guion=[decidir("investigador"), decidir("analista")])  # insiste siempre con el analista
    final = await correr(supervisor, especialistas, max_intentos=3)
    assert final["estado_final"] == "validacion_fallida"
    assert final["intentos"]["analista"] == 3
    assert "no coincide" in final["respuesta_final"]
    assert final["informe"] is None  # nunca se redacto sobre calculos rechazados


# ---------- Topes: el supervisor no puede quedar en bucle ----------


async def test_un_supervisor_que_repite_siempre_lo_mismo_se_corta_en_el_tope_de_pasos() -> None:
    especialistas = especialistas_falsos(guion_investigador() * 10, [], [])
    final = await correr(SupervisorFalso(guion=[decidir("investigador")]), especialistas, max_pasos=5, max_intentos=10)
    assert final["estado_final"] == "limite_de_pasos"
    assert final["pasos"] == 5 and "límite de 5 pasos" in final["respuesta_final"]
    assert any(d.corregida_por_guarda for d in final["decisiones"])


async def test_el_tope_de_pasos_se_respeta_sin_agotar_el_recursion_limit_de_langgraph() -> None:
    """El corte lo hace el supervisor con un mensaje claro, no una GraphRecursionError."""
    especialistas = especialistas_falsos(guion_investigador() * 10, [], [])
    final = await correr(SupervisorFalso(guion=[decidir("investigador")]), especialistas, max_pasos=3, max_intentos=10)
    assert final["estado_final"] == "limite_de_pasos"


async def test_cerrar_antes_de_tiempo_o_saltarse_pasos_lo_corrige_el_codigo_y_queda_registrado() -> None:
    supervisor = SupervisorFalso(
        guion=[
            decidir("FINISH"),  # cierra sin haber hecho nada
            decidir("redactor", "Escribí ya."),  # se salta al analista
            decidir("FINISH"),  # cierra sin informe aprobado
            decidir("FINISH"),
        ]
    )
    especialistas = especialistas_falsos(guion_investigador(), guion_analista(), [INFORME_OK])
    final = await correr(supervisor, especialistas)
    efectivas = [d.siguiente for d in final["decisiones"]]
    assert efectivas == ["investigador", "analista", "redactor", "FINISH"]
    corregidas = [d for d in final["decisiones"] if d.corregida_por_guarda]
    assert len(corregidas) == 3 and all(d.razon_guarda for d in corregidas)
    assert final["estado_final"] == "completado"


async def test_una_salida_invalida_del_supervisor_no_rompe_la_corrida() -> None:
    from langchain_core.exceptions import OutputParserException

    supervisor = SupervisorFalso(
        guion=[OutputParserException("sin forma"), decidir("analista"), decidir("redactor"), decidir("FINISH")]
    )
    especialistas = especialistas_falsos(guion_investigador(), guion_analista(), [INFORME_OK])
    final = await correr(supervisor, especialistas)
    assert final["estado_final"] == "completado"
    assert final["decisiones"][0].corregida_por_guarda and final["decisiones"][0].siguiente == "investigador"


# ---------- Aislamiento de estado y de contexto ----------


async def test_cada_nodo_escribe_solo_las_claves_de_su_parcela() -> None:
    especialistas = especialistas_falsos(guion_investigador(), guion_analista(), [INFORME_OK])
    recorrido = await _recorrido(_supervisor_ordenado(), especialistas)
    assert recorrido
    for nodo, actualizacion in recorrido:
        assert set(actualizacion) <= CLAVES_POR_NODO[nodo], (nodo, set(actualizacion) - CLAVES_POR_NODO[nodo])


async def test_el_supervisor_nunca_ve_datos_crudos_de_los_especialistas() -> None:
    supervisor = _supervisor_ordenado()
    especialistas = especialistas_falsos(guion_investigador(), guion_analista(), [INFORME_OK])
    await correr(supervisor, especialistas)
    assert len(supervisor.recibidos) == 4
    for contexto in supervisor.recibidos:
        assert [type(m) for m in contexto] == [SystemMessage, HumanMessage]
        texto = "\n".join(str(m.content) for m in contexto)
        for crudo in ("registros_afectados", "Laura Perez", "sla_horas_habiles", '"fecha_alta"'):
            assert crudo not in texto


async def test_el_analista_recibe_solo_los_datos_que_necesita_y_no_la_conversacion_del_investigador() -> None:
    especialistas = especialistas_falsos(guion_investigador(), guion_analista(), [INFORME_OK])
    await correr(_supervisor_ordenado(), especialistas)
    del_analista = [m for m in especialistas.recibidos if "Agente Analista" in str(m[0].content)][0]
    assert [type(m) for m in del_analista] == [SystemMessage, HumanMessage]
    entrada = str(del_analista[1].content)
    assert "INC-1001" in entrada and "Calculá" in entrada
    assert "buscar_incidentes_calidad" not in entrada  # nada del historial interno del investigador


async def test_las_herramientas_de_cada_agente_no_se_mezclan_en_el_grafo() -> None:
    """Si el analista pidiera una herramienta del investigador, la corrida no puede completarse."""
    analista_mal_configurado = [pedir("buscar_incidentes_calidad", maestro="articulos"), responder("listo")]
    final = await correr(
        SupervisorFalso(guion=[decidir("investigador"), decidir("analista")]),
        especialistas_falsos(guion_investigador(), analista_mal_configurado * 3, []),
        max_intentos=2,
    )
    assert final["estado_final"] == "validacion_fallida"
    assert final["analisis"] is not None and final["analisis"].herramientas_usadas == []  # la llamada ajena no cuenta


def test_las_herramientas_registradas_son_basetool() -> None:
    from agents.analyst_agent import HERRAMIENTAS_ANALISTA
    from agents.research_agent import HERRAMIENTAS_INVESTIGADOR

    assert all(isinstance(h, BaseTool) for h in [*HERRAMIENTAS_INVESTIGADOR, *HERRAMIENTAS_ANALISTA])


# ---------- Refinamientos: completan lo anterior, no lo reemplazan ----------


async def test_completar_las_reglas_en_un_refinamiento_no_hace_perder_los_incidentes() -> None:
    """Reproduce el bug de la primera corrida real: el 2do pedido al investigador traia solo reglas
    y los incidentes del 1ro desaparecian; el supervisor creia que no habia nada para analizar."""
    primera = [  # lista y pide el detalle, pero se olvida de las reglas
        pedir("buscar_incidentes_calidad", maestro="articulos", estado="abierto", severidad="critica"),
        pedir_varias(("detalle_incidente", {"incidente_id": "INC-1001"}), ("detalle_incidente", {"incidente_id": "INC-1002"})),
        responder("Encontré 2 incidentes críticos."),
    ]
    segunda = [  # el refinamiento pide solo lo que falta
        pedir_varias(("consultar_regla_calidad", {"regla_id": "R-ART-01"}), ("consultar_regla_calidad", {"regla_id": "R-ART-03"})),
        responder("Consulté las dos reglas."),
    ]
    supervisor = SupervisorFalso(
        guion=[
            decidir("investigador", "Recolectá los críticos."),
            decidir("investigador", "Completá las reglas de cada incidente."),
            decidir("analista"),
            decidir("redactor"),
            decidir("FINISH"),
        ]
    )
    final = await correr(supervisor, especialistas_falsos([*primera, *segunda], guion_analista(), [INFORME_OK]))
    assert [(v.agente, v.aprobada) for v in final["validaciones"]] == [
        ("investigador", False), ("investigador", True), ("analista", True), ("redactor", True),
    ]
    assert [i.id for i in final["hallazgos"].incidentes] == ["INC-1001", "INC-1002"]
    assert final["analisis"].ranking == ["INC-1001", "INC-1002"]  # el analista SI intervino
    assert final["estado_final"] == "completado" and "INC-1001" in final["respuesta_final"]


async def test_el_supervisor_ve_los_incidentes_ya_recolectados_despues_de_un_refinamiento() -> None:
    primera = [
        pedir("buscar_incidentes_calidad", maestro="articulos", estado="abierto", severidad="critica"),
        pedir_varias(("detalle_incidente", {"incidente_id": "INC-1001"}), ("detalle_incidente", {"incidente_id": "INC-1002"})),
        responder("Encontré 2 incidentes."),
    ]
    segunda = [pedir_varias(("consultar_regla_calidad", {"regla_id": "R-ART-01"}), ("consultar_regla_calidad", {"regla_id": "R-ART-03"})), responder("Reglas listas.")]
    supervisor = SupervisorFalso(guion=[decidir("investigador"), decidir("investigador"), decidir("analista"), decidir("redactor"), decidir("FINISH")])
    await correr(supervisor, especialistas_falsos([*primera, *segunda], guion_analista(), [INFORME_OK]))
    tercero = "\n".join(str(m.content) for m in supervisor.recibidos[2])
    assert "2 incidente(s) con ficha completa" in tercero


async def test_el_analista_puede_corregir_solo_lo_que_hizo_mal() -> None:
    primera = guion_analista(alterar={"INC-1001": {"fecha_alta": "2026-10-01"}})
    correcto = guion_analista()[0].tool_calls[0]  # calcular_vencimiento_sla de INC-1001, bien armado
    segunda = [pedir("calcular_vencimiento_sla", **correcto["args"]), responder("Rehice el vencimiento de INC-1001.")]
    supervisor = SupervisorFalso(
        guion=[decidir("investigador"), decidir("analista"), decidir("analista", "Rehacé INC-1001."), decidir("redactor"), decidir("FINISH")]
    )
    final = await correr(supervisor, especialistas_falsos(guion_investigador(), [*primera, *segunda], [INFORME_OK]))
    assert [(v.agente, v.aprobada) for v in final["validaciones"]][1:3] == [("analista", False), ("analista", True)]
    assert final["estado_final"] == "completado"
    assert len(final["analisis"].vencimientos) == 2  # el de INC-1002 venia de la primera corrida
