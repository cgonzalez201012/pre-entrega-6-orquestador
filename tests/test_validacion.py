"""Nodo de validacion: la rubrica que revisa cada aporte con codigo determinista."""

from datetime import date

import pytest

from state import (
    CLAVES_POR_NODO,
    Analisis,
    Contribucion,
    Hallazgos,
    IncidenteHallazgo,
)
from tests.conftest import crear_estado
from tests.fakes import analisis_de_datos, hallazgos_de_datos, informe
from validacion import (
    nodo_validacion,
    validar,
    validar_analista,
    validar_investigador,
    validar_redactor,
)


def _sin_detalle(hallazgos: Hallazgos) -> Hallazgos:
    """Los mismos incidentes pero solo listados (como si el investigador no hubiera pedido el detalle)."""
    solo_listados = [
        IncidenteHallazgo(id=i.id, maestro=i.maestro, titulo=i.titulo, severidad=i.severidad, estado=i.estado, fecha_alta=i.fecha_alta)
        for i in hallazgos.incidentes
    ]
    return hallazgos.model_copy(update={"incidentes": solo_listados, "reglas": []})


# ---------- Investigador ----------


def test_un_investigador_perfecto_se_aprueba() -> None:
    assert validar_investigador(hallazgos_de_datos()) == []


def test_sin_hallazgos_no_hay_nada_que_aprobar() -> None:
    assert "no entregó nada" in validar_investigador(None)[0]


def test_declarar_fuera_de_alcance_es_una_salida_valida() -> None:
    assert validar_investigador(Hallazgos(fuera_de_alcance=True, resumen="FUERA_DE_ALCANCE: no es mi tema")) == []


def test_responder_sin_consultar_ninguna_herramienta_se_rechaza() -> None:
    problemas = validar_investigador(Hallazgos(resumen="Hay 2 incidentes criticos, creo."))
    assert len(problemas) == 1 and "ninguna herramienta" in problemas[0] and "FUERA_DE_ALCANCE" in problemas[0]


def test_listar_incidentes_sin_pedir_ningun_detalle_se_rechaza() -> None:
    problemas = validar_investigador(_sin_detalle(hallazgos_de_datos()))
    assert any("detalle" in p for p in problemas)


def test_un_incidente_detallado_sin_su_regla_consultada_se_rechaza() -> None:
    hallazgos = hallazgos_de_datos().model_copy(update={"reglas": hallazgos_de_datos().reglas[:1]})
    assert validar_investigador(hallazgos) == ["Falta consultar la regla R-ART-03 del incidente INC-1002."]


def test_sin_fecha_de_corte_no_se_puede_calcular_y_se_rechaza() -> None:
    hallazgos = hallazgos_de_datos().model_copy(update={"fecha_corte": None})
    assert any("fecha de corte" in p for p in validar_investigador(hallazgos))


def test_si_todas_las_consultas_dieron_error_se_rechaza_con_los_errores() -> None:
    hallazgos = Hallazgos(herramientas_usadas=["detalle_incidente"], consultas_con_error=["detalle_incidente: No existe INC-9"])
    problemas = validar_investigador(hallazgos)
    assert any("INC-9" in p for p in problemas)


def test_que_no_haya_incidentes_que_cumplan_el_filtro_no_es_un_error() -> None:
    """Un filtro sin resultados es un hallazgo valido: el supervisor pasa directo al redactor."""
    assert validar_investigador(Hallazgos(herramientas_usadas=["buscar_incidentes_calidad"])) == []


# ---------- Analista ----------


def test_un_analista_perfecto_se_aprueba() -> None:
    hallazgos = hallazgos_de_datos()
    assert validar_analista(hallazgos, analisis_de_datos(hallazgos)) == []


def test_sin_analisis_o_sin_datos_se_rechaza() -> None:
    hallazgos = hallazgos_de_datos()
    assert "no entregó nada" in validar_analista(hallazgos, None)[0]
    assert "No hay incidentes detallados" in validar_analista(None, Analisis())[0]
    assert "No hay incidentes detallados" in validar_analista(_sin_detalle(hallazgos), Analisis())[0]


def test_falta_el_vencimiento_de_un_incidente() -> None:
    hallazgos = hallazgos_de_datos()
    correcto = analisis_de_datos(hallazgos)
    analisis = correcto.model_copy(update={"vencimientos": correcto.vencimientos[:1]})
    assert any("INC-1002" in p and "vencimiento" in p for p in validar_analista(hallazgos, analisis))


def test_falta_la_brecha_de_una_regla() -> None:
    hallazgos = hallazgos_de_datos()
    correcto = analisis_de_datos(hallazgos)
    analisis = correcto.model_copy(update={"brechas": correcto.brechas[:1]})
    assert any("R-ART-03" in p and "brecha" in p for p in validar_analista(hallazgos, analisis))


def test_un_vencimiento_calculado_con_una_fecha_distinta_a_la_de_los_datos_se_rechaza() -> None:
    """Pydantic no lo detecta (la forma es valida): lo detecta recalcular con los datos del investigador."""
    hallazgos = hallazgos_de_datos()
    correcto = analisis_de_datos(hallazgos)
    equivocado = correcto.vencimientos[0].model_copy(update={"fecha_alta": date(2026, 10, 1), "dias_habiles": 21})
    analisis = correcto.model_copy(update={"vencimientos": [equivocado, *correcto.vencimientos[1:]]})
    problemas = validar_analista(hallazgos, analisis)
    assert any("INC-1001" in p and "no coincide" in p for p in problemas)


def test_una_brecha_con_la_medicion_cambiada_se_rechaza() -> None:
    hallazgos = hallazgos_de_datos()
    correcto = analisis_de_datos(hallazgos)
    equivocada = correcto.brechas[0].model_copy(update={"medicion_pct": 99.0})
    analisis = correcto.model_copy(update={"brechas": [equivocada, *correcto.brechas[1:]]})
    assert any("no coincide" in p for p in validar_analista(hallazgos, analisis))


def test_calcular_un_incidente_que_no_esta_en_los_datos_se_rechaza() -> None:
    hallazgos = hallazgos_de_datos()
    correcto = analisis_de_datos(hallazgos)
    inventado = correcto.vencimientos[0].model_copy(update={"incidente_id": "INC-9999"})
    analisis = correcto.model_copy(update={"vencimientos": [*correcto.vencimientos, inventado]})
    assert any("INC-9999" in p for p in validar_analista(hallazgos, analisis))


def test_sin_ranking_se_rechaza() -> None:
    hallazgos = hallazgos_de_datos()
    analisis = analisis_de_datos(hallazgos).model_copy(update={"ranking": []})
    assert any("ranking" in p for p in validar_analista(hallazgos, analisis))


def test_un_ranking_en_otro_orden_se_rechaza_y_dice_cual_corresponde() -> None:
    hallazgos = hallazgos_de_datos()
    analisis = analisis_de_datos(hallazgos).model_copy(update={"ranking": ["INC-1002", "INC-1001"]})
    problemas = validar_analista(hallazgos, analisis)
    assert len(problemas) == 1 and "['INC-1001', 'INC-1002']" in problemas[0]


# ---------- Redactor ----------


def test_un_informe_que_cubre_todo_se_aprueba() -> None:
    hallazgos = hallazgos_de_datos()
    texto = informe("INC-1001", "INC-1002", "R-ART-01").content
    assert validar_redactor(hallazgos, analisis_de_datos(hallazgos), str(texto)) == []


def test_un_informe_vacio_se_rechaza() -> None:
    assert validar_redactor(None, None, "   ") == ["El informe está vacío."]
    assert validar_redactor(None, None, None) == ["El informe está vacío."]


def test_citar_un_id_que_no_existe_en_los_datos_se_rechaza() -> None:
    hallazgos = hallazgos_de_datos()
    problemas = validar_redactor(hallazgos, analisis_de_datos(hallazgos), "INC-1001 INC-1002 y también INC-7777 y R-ZZZ-01")
    assert any("INC-7777" in p and "R-ZZZ-01" in p for p in problemas)


def test_omitir_un_incidente_analizado_se_rechaza() -> None:
    hallazgos = hallazgos_de_datos()
    problemas = validar_redactor(hallazgos, analisis_de_datos(hallazgos), "Solo hablo de INC-1001.")
    assert any("INC-1002" in p for p in problemas)


def test_no_nombrar_el_incidente_de_mayor_prioridad_se_rechaza() -> None:
    hallazgos = hallazgos_de_datos()
    # el ranking pone primero a INC-1002; el informe cita solo INC-1001
    analisis = analisis_de_datos(hallazgos).model_copy(update={"ranking": ["INC-1002", "INC-1001"]})
    problemas = validar_redactor(hallazgos, analisis, "Hablo solo de INC-1001.")
    assert any("mayor prioridad" in p and "INC-1002" in p for p in problemas)
    assert validar_redactor(hallazgos, analisis, "INC-1002 primero y despues INC-1001.") == []


def test_si_la_consulta_quedo_fuera_de_alcance_el_informe_no_puede_citar_ids() -> None:
    hallazgos = Hallazgos(fuera_de_alcance=True)
    assert validar_redactor(hallazgos, None, "No tengo datos de empleados ni de sueldos.") == []
    assert validar_redactor(hallazgos, None, "El sueldo está en INC-1001.") != []


# ---------- validar() y el nodo ----------


def test_validar_elige_la_rubrica_del_ultimo_agente_que_entrego() -> None:
    hallazgos = hallazgos_de_datos()
    estado = crear_estado(hallazgos=hallazgos, ultimo_agente="investigador", pasos=1)
    resultado = validar(estado)
    assert resultado.agente == "investigador" and resultado.aprobada and resultado.paso == 1

    estado = crear_estado(hallazgos=hallazgos, analisis=None, ultimo_agente="analista", pasos=2)
    assert validar(estado).aprobada is False


def test_si_el_especialista_no_termino_su_aporte_se_rechaza_con_ese_motivo() -> None:
    hallazgos = hallazgos_de_datos()
    corte = Contribucion(agente="investigador", paso=1, resumen="Sin respuesta.", error="No terminó dentro del límite de 12 pasos.")
    resultado = validar(crear_estado(hallazgos=hallazgos, ultimo_agente="investigador", contribuciones=[corte], pasos=1))
    assert resultado.aprobada is False and resultado.problemas[0].startswith("No terminó")


def test_validar_sin_ningun_aporte_previo_es_un_error_de_programacion() -> None:
    with pytest.raises(ValueError, match="nada para validar"):
        validar(crear_estado())


async def test_el_nodo_escribe_solo_su_parcela() -> None:
    estado = crear_estado(hallazgos=hallazgos_de_datos(), ultimo_agente="investigador", pasos=1)
    salida = await nodo_validacion(estado)
    assert set(salida) <= CLAVES_POR_NODO["validacion"]
    assert salida["validacion"] == salida["validaciones"][0]
