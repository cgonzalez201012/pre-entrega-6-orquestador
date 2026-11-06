"""Herramientas del investigador (consulta) y del analista (calculo)."""

import json
from typing import Any

import pytest
from langchain_core.tools import BaseTool
from pydantic import ValidationError

from agents.analyst_agent import (
    HERRAMIENTAS_ANALISTA,
    calcular_brecha_regla,
    calcular_vencimiento_sla,
    priorizar_incidentes,
)
from agents.research_agent import (
    HERRAMIENTAS_INVESTIGADOR,
    buscar_incidentes_calidad,
    consultar_regla_calidad,
    detalle_incidente,
)
from repositorio import RepositorioCalidad


async def _usar(herramienta: BaseTool, **argumentos: Any) -> Any:
    return json.loads(await herramienta.ainvoke(argumentos))


# ---------- Investigador ----------


async def test_buscar_incidentes_criticos_abiertos_de_articulos() -> None:
    resultado = await _usar(buscar_incidentes_calidad, maestro="articulos", estado="abierto", severidad="critica")
    assert resultado["cantidad"] == 2
    assert [i["id"] for i in resultado["incidentes"]] == ["INC-1001", "INC-1002"]
    assert resultado["fecha_corte"] == "2026-10-31"


async def test_buscar_acepta_singular_y_tildes_pero_no_inventa_sinonimos() -> None:
    assert (await _usar(buscar_incidentes_calidad, maestro="Artículo"))["maestro"] == "articulos"
    error = await _usar(buscar_incidentes_calidad, maestro="productos")
    assert "productos" in error["error"] and "articulos" in error["ayuda"]


async def test_buscar_con_estado_o_severidad_invalidos_devuelve_error_con_ayuda() -> None:
    assert "ayuda" in await _usar(buscar_incidentes_calidad, maestro="articulos", estado="roto")
    assert "ayuda" in await _usar(buscar_incidentes_calidad, maestro="articulos", severidad="enorme")


async def test_el_detalle_trae_regla_registros_sla_y_fecha_de_corte() -> None:
    ficha = await _usar(detalle_incidente, incidente_id="INC-1001")
    assert ficha["regla_id"] == "R-ART-01"
    assert ficha["registros_afectados"] == 312
    assert ficha["sla_horas_habiles"] == 4
    assert ficha["fecha_corte"] == "2026-10-31"


async def test_detalle_de_un_incidente_inexistente_lista_los_ids_validos_del_maestro() -> None:
    error = await _usar(detalle_incidente, incidente_id="INC-1010")
    assert "INC-1010" in error["error"]
    assert "INC-1001" in error["ayuda"] and "articulos" in error["ayuda"]


async def test_regla_con_su_medicion_y_si_cumple() -> None:
    regla = await _usar(consultar_regla_calidad, regla_id="r-art-01")  # tolera minusculas
    assert regla["umbral_aceptacion_pct"] == 99.5
    assert regla["medicion_actual_pct"] == 96.8
    assert regla["cumple_umbral"] is False


async def test_regla_inexistente_devuelve_error_con_formato_esperado() -> None:
    error = await _usar(consultar_regla_calidad, regla_id="R-XXX-99")
    assert "R-XXX-NN" in error["ayuda"]


def test_el_repositorio_es_de_solo_lectura() -> None:
    """Minimo privilegio: el repositorio no expone ningun metodo que escriba."""
    publicos = {n for n in dir(RepositorioCalidad) if not n.startswith("_")}
    assert publicos == {"buscar_incidentes", "detalle_incidente", "regla"}


# ---------- Analista ----------


async def test_brecha_de_una_regla() -> None:
    brecha = await _usar(calcular_brecha_regla, regla_id="R-ART-01", medicion_pct=96.8, umbral_pct=99.5)
    assert brecha["brecha_pp"] == -2.7
    assert brecha["cumple"] is False


async def test_vencimiento_de_un_incidente() -> None:
    resultado = await _usar(
        calcular_vencimiento_sla,
        incidente_id="INC-1001",
        severidad="critica",
        fecha_alta="2026-09-14",
        fecha_corte="2026-10-31",
        sla_horas_habiles=4,
    )
    assert resultado["horas_habiles_transcurridas"] == 280
    assert resultado["exceso_horas"] == 276
    assert resultado["vencido"] is True


async def test_vencimiento_con_fechas_al_reves_devuelve_error_con_ayuda() -> None:
    error = await _usar(
        calcular_vencimiento_sla,
        incidente_id="INC-1001",
        severidad="critica",
        fecha_alta="2026-10-31",
        fecha_corte="2026-09-14",
        sla_horas_habiles=4,
    )
    assert "anterior" in error["error"] and "AAAA-MM-DD" in error["ayuda"]


async def test_los_argumentos_invalidos_se_rechazan_antes_de_calcular() -> None:
    with pytest.raises(ValidationError):
        await calcular_vencimiento_sla.ainvoke(
            {"incidente_id": "X", "severidad": "gravisima", "fecha_alta": "2026-09-14", "fecha_corte": "2026-10-31", "sla_horas_habiles": 4}
        )
    with pytest.raises(ValidationError):
        await calcular_vencimiento_sla.ainvoke(
            {"incidente_id": "X", "severidad": "alta", "fecha_alta": "no-es-fecha", "fecha_corte": "2026-10-31", "sla_horas_habiles": 4}
        )


async def test_priorizar_devuelve_los_ids_en_orden() -> None:
    items = [
        {"incidente_id": "B", "severidad": "critica", "vencido": True, "exceso_horas": 68, "registros_afectados": 41},
        {"incidente_id": "A", "severidad": "critica", "vencido": True, "exceso_horas": 276, "registros_afectados": 312},
    ]
    assert (await _usar(priorizar_incidentes, items=items))["ranking"] == ["A", "B"]


async def test_priorizar_exige_al_menos_un_incidente() -> None:
    with pytest.raises(ValidationError):
        await priorizar_incidentes.ainvoke({"items": []})


# ---------- Contrato de las herramientas (lo unico que ve el modelo) ----------


@pytest.mark.parametrize("herramienta", [*HERRAMIENTAS_INVESTIGADOR, *HERRAMIENTAS_ANALISTA], ids=lambda h: h.name)
def test_cada_herramienta_tiene_descripcion_larga_y_esquema_de_argumentos(herramienta: BaseTool) -> None:
    assert len(herramienta.description) > 150  # el modelo decide leyendo solo esto
    esquema = herramienta.get_input_schema().model_json_schema()
    assert esquema["properties"]
    assert all("description" in propiedad for propiedad in esquema["properties"].values())


def test_las_herramientas_de_cada_agente_estan_acotadas_y_no_se_mezclan() -> None:
    investigador = {h.name for h in HERRAMIENTAS_INVESTIGADOR}
    analista = {h.name for h in HERRAMIENTAS_ANALISTA}
    assert investigador == {"buscar_incidentes_calidad", "detalle_incidente", "consultar_regla_calidad"}
    assert analista == {"calcular_brecha_regla", "calcular_vencimiento_sla", "priorizar_incidentes"}
    assert investigador.isdisjoint(analista)
