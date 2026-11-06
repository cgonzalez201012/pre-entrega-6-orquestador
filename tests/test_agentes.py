"""Los tres especialistas: arman su contrato desde las herramientas y trabajan en contexto aislado."""

import json
from typing import Any

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage, ToolMessage

from agents.analyst_agent import (
    HERRAMIENTAS_ANALISTA,
    construir_analisis,
    crear_nodo_analista,
    datos_para_el_analista,
)
from agents.base import resultados_de, texto_de
from agents.research_agent import construir_hallazgos, crear_nodo_investigador
from agents.writer_agent import crear_nodo_redactor, datos_para_el_redactor
from state import CLAVES_POR_NODO, FUERA_DE_ALCANCE, ResultadoValidacion
from tests.conftest import TAREA, crear_estado
from tests.fakes import (
    ModeloPorPolitica,
    analisis_de_datos,
    especialistas_falsos,
    guion_analista,
    guion_investigador,
    hallazgos_de_datos,
    informe,
    pedir,
    responder,
)


def _tool(nombre: str, contenido: Any, llamada: str = "x") -> ToolMessage:
    texto = contenido if isinstance(contenido, str) else json.dumps(contenido)
    return ToolMessage(content=texto, name=nombre, tool_call_id=llamada)


# ---------- construir_hallazgos: sale de las herramientas, no del texto del modelo ----------


def test_los_hallazgos_se_arman_con_lo_que_devolvieron_las_herramientas() -> None:
    listado = {
        "maestro": "articulos",
        "fecha_corte": "2026-10-31",
        "incidentes": [
            {"id": "INC-1001", "titulo": "Sin EAN", "severidad": "critica", "estado": "abierto", "fecha_alta": "2026-09-14", "dias_desde_el_alta": 47}
        ],
    }
    detalle = {
        "id": "INC-1001", "maestro": "articulos", "titulo": "Sin EAN", "severidad": "critica", "estado": "abierto",
        "fecha_alta": "2026-09-14", "regla_id": "R-ART-01", "registros_afectados": 312,
        "responsable": "Gerente Comercial", "data_steward": "Laura Perez", "sla_horas_habiles": 4, "fecha_corte": "2026-10-31",
    }
    regla = {
        "id": "R-ART-01", "maestro": "articulos", "nombre": "EAN obligatorio", "descripcion": "d",
        "umbral_aceptacion_pct": 99.5, "medicion_actual_pct": 96.8, "cumple_umbral": False,
    }
    mensajes: list[BaseMessage] = [
        _tool("buscar_incidentes_calidad", listado),
        _tool("detalle_incidente", detalle),
        _tool("consultar_regla_calidad", regla),
    ]
    hallazgos = construir_hallazgos(mensajes, "dije cualquier cosa")
    incidente = hallazgos.incidentes[0]
    assert incidente.detallado and incidente.registros_afectados == 312 and incidente.responsable == "Gerente Comercial"
    assert hallazgos.fecha_corte is not None and hallazgos.fecha_corte.isoformat() == "2026-10-31"
    assert hallazgos.reglas[0].medicion_actual_pct == 96.8
    assert hallazgos.consultas_con_error == []
    assert hallazgos.resumen == "dije cualquier cosa"  # el texto del modelo solo viaja como resumen


def test_un_incidente_solo_listado_no_queda_detallado() -> None:
    listado = {
        "maestro": "articulos", "fecha_corte": "2026-10-31",
        "incidentes": [{"id": "INC-1003", "titulo": "t", "severidad": "alta", "estado": "abierto", "fecha_alta": "2026-10-05", "dias_desde_el_alta": 26}],
    }
    hallazgos = construir_hallazgos([_tool("buscar_incidentes_calidad", listado)], "")
    assert len(hallazgos.incidentes) == 1 and hallazgos.incidentes_detallados() == []


def test_los_errores_de_las_herramientas_quedan_registrados() -> None:
    mensajes: list[BaseMessage] = [
        _tool("detalle_incidente", {"error": "No existe el incidente 'INC-1010'.", "ayuda": "..."}),
        _tool("consultar_regla_calidad", "Error: la herramienta fallo"),  # texto que no es JSON (lo arma el ToolNode)
    ]
    hallazgos = construir_hallazgos(mensajes, "")
    assert hallazgos.incidentes == []
    assert len(hallazgos.consultas_con_error) == 2
    assert "INC-1010" in hallazgos.consultas_con_error[0]


def test_una_respuesta_con_formato_inesperado_no_rompe_el_nodo() -> None:
    hallazgos = construir_hallazgos([_tool("buscar_incidentes_calidad", {"algo": "raro"})], "")
    assert hallazgos.incidentes == []
    assert "formato inesperado" in hallazgos.consultas_con_error[0]


def test_la_marca_fuera_de_alcance_se_detecta_en_el_texto_del_modelo() -> None:
    assert construir_hallazgos([], f"{FUERA_DE_ALCANCE}: no trata de calidad de datos.").fuera_de_alcance is True
    assert construir_hallazgos([], "Encontré 2 incidentes.").fuera_de_alcance is False


def test_resultados_de_devuelve_en_orden_solo_la_herramienta_pedida() -> None:
    mensajes: list[BaseMessage] = [_tool("a", {"n": 1}), _tool("b", {"n": 2}), _tool("a", {"n": 3})]
    assert resultados_de(mensajes, "a") == [{"n": 1}, {"n": 3}]


# ---------- construir_analisis ----------


def test_el_analisis_se_arma_con_los_resultados_de_las_herramientas() -> None:
    correcto = analisis_de_datos(hallazgos_de_datos())
    mensajes: list[BaseMessage] = [
        *[_tool("calcular_vencimiento_sla", v.model_dump(mode="json")) for v in correcto.vencimientos],
        *[_tool("calcular_brecha_regla", b.model_dump(mode="json")) for b in correcto.brechas],
        _tool("priorizar_incidentes", {"ranking": correcto.ranking}),
    ]
    analisis = construir_analisis(mensajes, "listo")
    assert analisis.vencimientos == correcto.vencimientos
    assert analisis.brechas == correcto.brechas
    assert analisis.ranking == correcto.ranking == ["INC-1001", "INC-1002"]


def test_si_el_analista_repite_un_calculo_gana_el_ultimo() -> None:
    primero = {"regla_id": "R-ART-01", "medicion_pct": 90.0, "umbral_pct": 99.5, "brecha_pp": -9.5, "cumple": False}
    segundo = {"regla_id": "R-ART-01", "medicion_pct": 96.8, "umbral_pct": 99.5, "brecha_pp": -2.7, "cumple": False}
    analisis = construir_analisis([_tool("calcular_brecha_regla", primero), _tool("calcular_brecha_regla", segundo)], "")
    assert [b.medicion_pct for b in analisis.brechas] == [96.8]


def test_los_errores_del_analista_quedan_registrados() -> None:
    analisis = construir_analisis(
        [_tool("calcular_vencimiento_sla", {"error": "fechas al reves", "ayuda": "..."}), _tool("priorizar_incidentes", {"x": 1})],
        "",
    )
    assert analisis.vencimientos == [] and analisis.ranking == []
    assert len(analisis.errores) == 2


# ---------- Los nodos: contexto aislado y escritura solo en su parcela ----------


async def test_el_investigador_recibe_solo_su_instruccion_y_escribe_solo_su_parcela() -> None:
    llm = especialistas_falsos(investigador=guion_investigador())
    nodo = crear_nodo_investigador(llm, recursion_limit=12)
    estado = crear_estado(instruccion="Recolectá los críticos abiertos de artículos.", pasos=1)
    salida = await nodo(estado, {})

    assert set(salida) <= CLAVES_POR_NODO["investigador"]
    primera_llamada = llm.recibidos[0]
    assert len(primera_llamada) == 2
    assert isinstance(primera_llamada[0], SystemMessage) and isinstance(primera_llamada[1], HumanMessage)
    assert primera_llamada[1].content == "Recolectá los críticos abiertos de artículos."
    assert TAREA not in str(primera_llamada[1].content)  # no recibe la conversacion del grafo principal
    assert [i.id for i in salida["hallazgos"].incidentes_detallados()] == ["INC-1001", "INC-1002"]
    contribucion = salida["contribuciones"][0]
    assert contribucion.agente == "investigador" and contribucion.paso == 1
    assert "detalle_incidente" in contribucion.herramientas_usadas


async def test_en_un_refinamiento_el_investigador_recibe_los_problemas_marcados() -> None:
    llm = especialistas_falsos(investigador=guion_investigador())
    nodo = crear_nodo_investigador(llm, recursion_limit=12)
    rechazo = ResultadoValidacion(agente="investigador", aprobada=False, problemas=["Falta consultar la regla R-ART-03."], paso=1)
    estado = crear_estado(instruccion="Reintentá.", validacion=rechazo, pasos=2)
    await nodo(estado, {})
    entrada = str(llm.recibidos[0][1].content)
    assert "Falta consultar la regla R-ART-03." in entrada and "rechazado" in entrada


async def test_los_problemas_de_otro_agente_no_se_le_pasan_a_este() -> None:
    llm = especialistas_falsos(investigador=guion_investigador())
    nodo = crear_nodo_investigador(llm, recursion_limit=12)
    rechazo_ajeno = ResultadoValidacion(agente="analista", aprobada=False, problemas=["Falta el ranking."], paso=1)
    await nodo(crear_estado(instruccion="Recolectá.", validacion=rechazo_ajeno), {})
    assert "Falta el ranking" not in str(llm.recibidos[0][1].content)


async def test_el_analista_recibe_la_instruccion_y_los_datos_pero_no_el_resumen_ni_la_tarea() -> None:
    hallazgos = hallazgos_de_datos()
    llm = especialistas_falsos(analista=guion_analista())
    nodo = crear_nodo_analista(llm, recursion_limit=12)
    estado = crear_estado(instruccion="Calculá todo.", hallazgos=hallazgos, pasos=2)
    salida = await nodo(estado, {})

    assert set(salida) <= CLAVES_POR_NODO["analista"]
    entrada = str(llm.recibidos[0][1].content)
    assert "Calculá todo." in entrada and "INC-1001" in entrada and "R-ART-01" in entrada
    assert "resumen del investigador" not in entrada  # texto libre de otro agente: no hace falta
    assert TAREA not in entrada
    assert "responsable" not in entrada  # tampoco los campos que no necesita para calcular
    assert salida["analisis"].ranking == ["INC-1001", "INC-1002"]


async def test_el_redactor_recibe_la_consulta_y_los_datos_validados_y_escribe_el_informe() -> None:
    hallazgos = hallazgos_de_datos()
    llm = especialistas_falsos(redactor=[informe("INC-1001", "INC-1002")])
    nodo = crear_nodo_redactor(llm)
    estado = crear_estado(instruccion="Redactá.", hallazgos=hallazgos, analisis=analisis_de_datos(hallazgos), pasos=3)
    salida = await nodo(estado, {})

    assert set(salida) <= CLAVES_POR_NODO["redactor"]
    entrada = str(llm.recibidos[0][1].content)
    assert TAREA in entrada and "ranking_de_prioridad" in entrada
    assert salida["informe"].startswith("Informe de calidad de datos")
    assert salida["messages"][0].name == "redactor"


async def test_un_especialista_que_se_pasa_del_limite_de_pasos_no_rompe_el_grafo() -> None:
    """Si el ciclo ReAct interno no termina, el nodo devuelve un aporte con error (la validacion lo rechaza)."""

    def siempre_pide(_: list[BaseMessage]) -> AIMessage:
        return pedir("buscar_incidentes_calidad", maestro="articulos")

    llm = ModeloPorPolitica(politica=siempre_pide)
    nodo = crear_nodo_investigador(llm, recursion_limit=4)
    salida = await nodo(crear_estado(instruccion="Recolectá.", pasos=1), {})
    contribucion = salida["contribuciones"][0]
    assert contribucion.error is not None and "límite" in contribucion.error
    assert salida["hallazgos"].incidentes == []


# ---------- Los datos que se le pasan a cada agente ----------


def test_los_datos_del_analista_incluyen_solo_incidentes_detallados_y_sus_reglas() -> None:
    datos = json.loads(datos_para_el_analista(hallazgos_de_datos(("INC-1001",))))
    assert [i["incidente_id"] for i in datos["incidentes"]] == ["INC-1001"]
    assert [r["regla_id"] for r in datos["reglas"]] == ["R-ART-01"]
    assert set(datos["incidentes"][0]) == {
        "incidente_id", "severidad", "fecha_alta", "regla_id", "registros_afectados", "sla_horas_habiles"
    }


def test_los_datos_del_redactor_no_llevan_analisis_si_no_hubo() -> None:
    assert "analisis" not in json.loads(datos_para_el_redactor(hallazgos_de_datos(), None))


def test_texto_de_junta_los_bloques_de_texto_de_claude() -> None:
    mensaje = AIMessage(content=[{"type": "text", "text": "Hola "}, {"type": "tool_use", "id": "1", "name": "x", "input": {}}, {"type": "text", "text": "mundo"}])
    assert texto_de(mensaje) == "Hola mundo"
    assert texto_de(responder("  listo  ")) == "listo"


def test_el_analista_tiene_exactamente_tres_herramientas_de_calculo() -> None:
    assert len(HERRAMIENTAS_ANALISTA) == 3
