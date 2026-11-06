"""Traza de una corrida: contadores de uso, armado desde el estado, log y archivos."""

import json
from pathlib import Path

from langchain_core.messages import AIMessage
from langchain_core.outputs import ChatGeneration, LLMResult

from tests.conftest import correr
from tests.fakes import SupervisorFalso, decidir, especialistas_falsos, guion_analista, guion_investigador, informe
from trazas import Corrida, ContadorDeUso, a_log, corrida_desde_estado, guardar_traza

INFORME_OK = informe("INC-1001", "INC-1002", "R-ART-01")


async def _corrida_feliz() -> Corrida:
    supervisor = SupervisorFalso(
        guion=[decidir("investigador", "Recolectá."), decidir("analista", "Calculá."), decidir("redactor", "Escribí."), decidir("FINISH")]
    )
    final = await correr(supervisor, especialistas_falsos(guion_investigador(), guion_analista(), [INFORME_OK]))
    return corrida_desde_estado("escenario de prueba", final, segundos=1.234)


async def test_el_contador_suma_llamadas_y_tokens() -> None:
    contador = ContadorDeUso()
    mensaje = AIMessage(content="hola", usage_metadata={"input_tokens": 120, "output_tokens": 30, "total_tokens": 150})
    resultado = LLMResult(generations=[[ChatGeneration(message=mensaje)]])
    await contador.on_llm_end(resultado)
    await contador.on_llm_end(resultado)
    await contador.on_llm_end(LLMResult(generations=[[ChatGeneration(message=AIMessage(content="sin uso"))]]))
    assert (contador.llamadas, contador.tokens_entrada, contador.tokens_salida) == (3, 240, 60)


async def test_la_corrida_junta_decisiones_aportes_y_validaciones() -> None:
    corrida = await _corrida_feliz()
    assert corrida.estado_final == "completado" and corrida.pasos_del_supervisor == 3
    assert corrida.agentes_que_intervinieron == ["investigador", "analista", "redactor"]
    assert [d["siguiente"] for d in corrida.decisiones] == ["investigador", "analista", "redactor", "FINISH"]
    assert [v["aprobada"] for v in corrida.validaciones] == [True, True, True]
    assert corrida.decisiones_corregidas_por_guarda == 0 and corrida.segundos == 1.23


async def test_los_agentes_que_intervinieron_no_se_repiten() -> None:
    corrida = await _corrida_feliz()
    corrida.contribuciones.append(dict(corrida.contribuciones[0]))
    assert corrida.agentes_que_intervinieron == ["investigador", "analista", "redactor"]


async def test_el_log_cuenta_la_delegacion_paso_a_paso() -> None:
    texto = a_log([await _corrida_feliz()])
    assert "=== escenario de prueba ===" in texto
    for fragmento in (
        "[Paso 1] Supervisor → investigador",
        'Instrucción: "Calculá."',
        "Investigador usó: buscar_incidentes_calidad, detalle_incidente x2, consultar_regla_calidad x2",
        "Validación: APROBADA",
        "[Paso 3] Supervisor → redactor",
        "Supervisor → FINISH (cierra la tarea)",
        "Estado final: completado | delegaciones: 3",
    ):
        assert fragmento in texto, fragmento


async def test_el_log_muestra_los_rechazos_y_las_guardas() -> None:
    mal = guion_analista(alterar={"INC-1001": {"fecha_alta": "2026-10-01"}})
    supervisor = SupervisorFalso(guion=[decidir("FINISH"), decidir("analista"), decidir("analista", "Rehacé."), decidir("redactor"), decidir("FINISH")])
    final = await correr(supervisor, especialistas_falsos(guion_investigador(), [*mal, *guion_analista()], [INFORME_OK]))
    texto = a_log([corrida_desde_estado("con problemas", final)])
    assert "!! Guarda corrigió la decisión" in texto
    assert "Validación: RECHAZADA" in texto and "no coincide" in texto


async def test_guardar_traza_escribe_json_y_log(tmp_path: Path) -> None:
    corrida = await _corrida_feliz()
    ruta_json, ruta_log = guardar_traza(tmp_path / "trazas", "traza-prueba", [corrida], {"proyecto": "prueba", "max_pasos": 10})
    datos = json.loads(ruta_json.read_text(encoding="utf-8"))
    assert datos["proyecto"] == "prueba" and datos["generado_en"]
    assert datos["corridas"][0]["agentes_que_intervinieron"] == ["investigador", "analista", "redactor"]
    assert ruta_log.read_text(encoding="utf-8").startswith("=== escenario de prueba ===")
