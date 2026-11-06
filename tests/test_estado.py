"""Estado compartido: contratos Pydantic, reducers y reglas de escritura por parcela."""

from datetime import date
from typing import get_type_hints

import pytest
from pydantic import ValidationError

from state import (
    CLAVES_POR_NODO,
    Analisis,
    EstadoOrquestador,
    Hallazgos,
    IncidenteHallazgo,
    ItemPrioridad,
    estado_inicial,
    sumar_contadores,
)
from tests.fakes import analisis_de_datos, hallazgos_de_datos


def _incidente(**cambios: object) -> IncidenteHallazgo:
    datos: dict[str, object] = {
        "id": "INC-1001",
        "maestro": "articulos",
        "titulo": "Articulos sin EAN",
        "severidad": "critica",
        "estado": "abierto",
        "fecha_alta": date(2026, 9, 14),
    }
    datos.update(cambios)
    return IncidenteHallazgo(**datos)  # type: ignore[arg-type]


def test_un_incidente_solo_esta_detallado_si_tiene_regla_registros_y_sla() -> None:
    assert _incidente().detallado is False
    assert _incidente(regla_id="R-ART-01", registros_afectados=312).detallado is False
    assert _incidente(regla_id="R-ART-01", registros_afectados=312, sla_horas_habiles=4).detallado is True


def test_la_severidad_tiene_que_ser_una_de_las_cuatro() -> None:
    with pytest.raises(ValidationError):
        _incidente(severidad="gravisima")


def test_la_fecha_se_interpreta_desde_texto_iso() -> None:
    assert _incidente(fecha_alta="2026-09-14").fecha_alta == date(2026, 9, 14)
    with pytest.raises(ValidationError):
        _incidente(fecha_alta="14/09/2026")


def test_los_contratos_son_inmutables_un_agente_no_puede_pisar_lo_de_otro() -> None:
    hallazgos = hallazgos_de_datos()
    with pytest.raises(ValidationError):
        hallazgos.resumen = "otro"  # type: ignore[misc]
    with pytest.raises(ValidationError):
        hallazgos.incidentes[0].titulo = "otro"  # type: ignore[misc]


def test_hallazgos_distingue_incidentes_detallados_y_solo_listados() -> None:
    hallazgos = Hallazgos(incidentes=[_incidente(), _incidente(id="INC-1002", regla_id="R-ART-03", registros_afectados=41, sla_horas_habiles=4)])
    assert [i.id for i in hallazgos.incidentes_detallados()] == ["INC-1002"]


def test_ids_conocidos_junta_incidentes_y_reglas() -> None:
    hallazgos = hallazgos_de_datos()
    assert hallazgos.ids_conocidos() == {"INC-1001", "INC-1002", "R-ART-01", "R-ART-03"}


def test_la_prioridad_no_acepta_horas_ni_registros_negativos() -> None:
    with pytest.raises(ValidationError):
        ItemPrioridad(incidente_id="A", severidad="alta", vencido=True, exceso_horas=-1, registros_afectados=1)


def test_analisis_arranca_vacio() -> None:
    assert Analisis().ranking == []


def test_el_reducer_de_intentos_suma_sin_mutar_los_diccionarios() -> None:
    actual = {"investigador": 1}
    nuevo = {"investigador": 1, "analista": 1}
    resultado = sumar_contadores(actual, nuevo)
    assert resultado == {"investigador": 2, "analista": 1}
    assert actual == {"investigador": 1} and nuevo == {"investigador": 1, "analista": 1}


def test_el_estado_inicial_tiene_exactamente_los_campos_del_estado() -> None:
    campos = set(get_type_hints(EstadoOrquestador, include_extras=True))
    assert set(estado_inicial("tarea")) == campos


def test_el_estado_hereda_messages_de_messages_state() -> None:
    assert "messages" in get_type_hints(EstadoOrquestador, include_extras=True)
    assert [m.content for m in estado_inicial("hola")["messages"]] == ["hola"]  # type: ignore[attr-defined]


def test_cada_campo_tiene_a_lo_sumo_un_nodo_dueno_salvo_los_de_registro() -> None:
    """Aislamiento de escritura: solo los campos de registro (se anexan) los comparten varios nodos."""
    compartidos = {"messages", "contribuciones", "ultimo_agente"}
    duenos: dict[str, list[str]] = {}
    for nodo, claves in CLAVES_POR_NODO.items():
        for clave in claves - compartidos:
            duenos.setdefault(clave, []).append(nodo)
    assert all(len(nodos) == 1 for nodos in duenos.values()), duenos


def test_los_campos_de_cada_parcela_existen_en_el_estado() -> None:
    campos = set(get_type_hints(EstadoOrquestador, include_extras=True))
    for claves in CLAVES_POR_NODO.values():
        assert claves <= campos


def test_el_estado_inicial_arranca_vacio_y_apunta_al_investigador() -> None:
    estado = estado_inicial("una tarea")
    assert estado["tarea"] == "una tarea" and estado["siguiente"] == "investigador"
    assert estado["pasos"] == 0 and estado["intentos"] == {}
    assert estado["decisiones"] == [] and estado["validaciones"] == [] and estado["contribuciones"] == []
    assert estado["hallazgos"] is None and estado["analisis"] is None and estado["informe"] is None
    assert estado["estado_final"] is None and estado["respuesta_final"] is None


# ---------- Fusion de aportes (los refinamientos completan, no reemplazan) ----------


def test_fusionar_hallazgos_conserva_los_incidentes_y_suma_las_reglas() -> None:
    sin_reglas = hallazgos_de_datos(con_reglas=False).model_copy(update={"herramientas_usadas": ["buscar_incidentes_calidad", "detalle_incidente"]})
    solo_reglas = Hallazgos(
        reglas=hallazgos_de_datos().reglas,
        herramientas_usadas=["consultar_regla_calidad"],
        resumen="completé las reglas",
    )
    fusion = sin_reglas.fusionar(solo_reglas)
    assert [i.id for i in fusion.incidentes] == ["INC-1001", "INC-1002"]
    assert {r.id for r in fusion.reglas} == {"R-ART-01", "R-ART-03"}
    assert fusion.fecha_corte == sin_reglas.fecha_corte  # la fecha de corte no se pierde
    assert fusion.herramientas_usadas == ["buscar_incidentes_calidad", "detalle_incidente", "consultar_regla_calidad"]
    assert fusion.resumen == "completé las reglas"


def test_fusionar_hallazgos_un_listado_nuevo_no_borra_el_detalle_ya_recolectado() -> None:
    detallado = hallazgos_de_datos()
    solo_listado = Hallazgos(incidentes=[_incidente()])  # INC-1001 sin regla, registros ni SLA
    fusion = detallado.fusionar(solo_listado)
    inc = next(i for i in fusion.incidentes if i.id == "INC-1001")
    assert inc.detallado and inc.registros_afectados == 312


def test_fusionar_hallazgos_lo_nuevo_pisa_a_lo_viejo_en_el_mismo_id() -> None:
    viejo = Hallazgos(incidentes=[_incidente(estado="abierto")])
    nuevo = Hallazgos(incidentes=[_incidente(estado="cerrado")])
    assert viejo.fusionar(nuevo).incidentes[0].estado == "cerrado"


def test_fusionar_hallazgos_los_errores_son_los_de_la_ultima_corrida() -> None:
    viejo = Hallazgos(consultas_con_error=["detalle_incidente: No existe INC-9"])
    nuevo = Hallazgos(incidentes=[_incidente()])
    assert viejo.fusionar(nuevo).consultas_con_error == []


def test_fusionar_hallazgos_si_la_ultima_corrida_declara_fuera_de_alcance_eso_vale() -> None:
    fusion = hallazgos_de_datos().fusionar(Hallazgos(fuera_de_alcance=True, resumen="FUERA_DE_ALCANCE: no es mi tema"))
    assert fusion.fuera_de_alcance and fusion.incidentes == []


def test_fusionar_analisis_suma_por_id_y_conserva_el_ranking_si_el_nuevo_no_trae() -> None:
    hallazgos = hallazgos_de_datos()
    completo = analisis_de_datos(hallazgos)
    parcial = Analisis(vencimientos=completo.vencimientos[:1], herramientas_usadas=["calcular_vencimiento_sla"])
    fusion = completo.fusionar(parcial)
    assert len(fusion.vencimientos) == 2 and len(fusion.brechas) == 2
    assert fusion.ranking == ["INC-1001", "INC-1002"]
    assert fusion.herramientas_usadas[-1] == "calcular_vencimiento_sla"


def test_fusionar_analisis_un_ranking_nuevo_reemplaza_al_anterior() -> None:
    completo = analisis_de_datos(hallazgos_de_datos())
    assert completo.fusionar(Analisis(ranking=["INC-1002", "INC-1001"])).ranking == ["INC-1002", "INC-1001"]


def test_fusionar_hallazgos_conserva_las_reglas_ya_consultadas_cuando_se_agregan_otras() -> None:
    reglas = hallazgos_de_datos().reglas  # R-ART-01 y R-ART-03
    viejo = Hallazgos(reglas=reglas[:1], herramientas_usadas=["consultar_regla_calidad"])
    nuevo = Hallazgos(reglas=reglas[1:], herramientas_usadas=["consultar_regla_calidad"])
    fusion = viejo.fusionar(nuevo)
    assert {r.id for r in fusion.reglas} == {"R-ART-01", "R-ART-03"}
    assert fusion.herramientas_usadas == ["consultar_regla_calidad", "consultar_regla_calidad"]
