"""Calculos deterministas del analista: dias habiles, vencimiento de SLA, brecha y prioridad."""

from datetime import date

import pytest

import calculos
from errores import ErrorCalculo
from state import ItemPrioridad


# ---------- dias habiles ----------


def test_una_semana_completa_tiene_cinco_dias_habiles() -> None:
    assert calculos.dias_habiles_entre(date(2026, 10, 19), date(2026, 10, 26)) == 5  # lunes a lunes


def test_el_intervalo_incluye_la_fecha_de_alta_y_excluye_la_de_corte() -> None:
    # viernes 30/10 -> lunes 2/11: solo cuenta el viernes (sabado y domingo no)
    assert calculos.dias_habiles_entre(date(2026, 10, 30), date(2026, 11, 2)) == 1


def test_el_mismo_dia_tiene_cero_dias_habiles() -> None:
    assert calculos.dias_habiles_entre(date(2026, 10, 20), date(2026, 10, 20)) == 0


def test_solo_un_fin_de_semana_tiene_cero_dias_habiles() -> None:
    assert calculos.dias_habiles_entre(date(2026, 10, 31), date(2026, 11, 2)) == 0  # sabado -> lunes


def test_fechas_al_reves_es_un_error_con_ayuda() -> None:
    with pytest.raises(ErrorCalculo) as error:
        calculos.dias_habiles_entre(date(2026, 10, 31), date(2026, 10, 1))
    assert "AAAA-MM-DD" in error.value.ayuda


# ---------- vencimiento del SLA ----------


def test_vencimiento_de_un_incidente_critico_muy_viejo() -> None:
    resultado = calculos.calcular_vencimiento("INC-1001", "critica", date(2026, 9, 14), date(2026, 10, 31), 4)
    assert resultado.dias_habiles == 35
    assert resultado.horas_habiles_transcurridas == 35 * calculos.JORNADA_HORAS == 280
    assert resultado.exceso_horas == 276
    assert resultado.vencido is True


def test_un_incidente_dentro_del_plazo_no_esta_vencido_y_no_tiene_exceso() -> None:
    resultado = calculos.calcular_vencimiento("INC-3003", "baja", date(2026, 10, 28), date(2026, 10, 31), 120)
    assert resultado.horas_habiles_transcurridas == 24
    assert resultado.exceso_horas == 0
    assert resultado.vencido is False


def test_justo_en_el_limite_del_plazo_no_esta_vencido() -> None:
    # 3 dias habiles = 24 horas = exactamente el SLA de 24 horas de severidad alta
    resultado = calculos.calcular_vencimiento("INC-X", "alta", date(2026, 10, 26), date(2026, 10, 29), 24)
    assert resultado.horas_habiles_transcurridas == 24
    assert resultado.vencido is False
    assert resultado.exceso_horas == 0


def test_una_hora_habil_despues_del_limite_ya_esta_vencido() -> None:
    resultado = calculos.calcular_vencimiento("INC-X", "alta", date(2026, 10, 26), date(2026, 10, 29), 23)
    assert resultado.vencido is True
    assert resultado.exceso_horas == 1


# ---------- brecha contra el umbral ----------


def test_brecha_negativa_cuando_la_regla_no_cumple() -> None:
    brecha = calculos.calcular_brecha("R-ART-01", 96.8, 99.5)
    assert brecha.brecha_pp == -2.7
    assert brecha.cumple is False


def test_brecha_cero_o_positiva_cuando_cumple() -> None:
    assert calculos.calcular_brecha("R-X", 99.5, 99.5).cumple is True
    assert calculos.calcular_brecha("R-ART-05", 100.0, 99.9) == calculos.calcular_brecha("R-ART-05", 100.0, 99.9)
    assert calculos.calcular_brecha("R-ART-05", 100.0, 99.9).brecha_pp == 0.1


# ---------- prioridad ----------


def _item(id_: str, severidad: str, vencido: bool, exceso: int, registros: int) -> ItemPrioridad:
    return ItemPrioridad(
        incidente_id=id_, severidad=severidad, vencido=vencido, exceso_horas=exceso, registros_afectados=registros  # type: ignore[arg-type]
    )


def test_la_severidad_manda_sobre_el_vencimiento() -> None:
    orden = calculos.ordenar_por_prioridad(
        [_item("A", "baja", True, 500, 1000), _item("B", "critica", False, 0, 1)]
    )
    assert orden == ["B", "A"]


def test_con_igual_severidad_primero_el_vencido() -> None:
    orden = calculos.ordenar_por_prioridad([_item("A", "alta", False, 0, 99), _item("B", "alta", True, 1, 1)])
    assert orden == ["B", "A"]


def test_con_igual_severidad_y_vencimiento_primero_el_que_mas_se_paso() -> None:
    orden = calculos.ordenar_por_prioridad([_item("A", "critica", True, 68, 41), _item("B", "critica", True, 276, 312)])
    assert orden == ["B", "A"]


def test_el_ultimo_desempate_son_los_registros_afectados_y_despues_el_id() -> None:
    orden = calculos.ordenar_por_prioridad(
        [_item("B", "media", True, 10, 5), _item("A", "media", True, 10, 5), _item("C", "media", True, 10, 50)]
    )
    assert orden == ["C", "A", "B"]


def test_el_orden_no_depende_del_orden_de_entrada() -> None:
    items = [_item("A", "alta", True, 5, 5), _item("B", "critica", True, 1, 1), _item("C", "baja", False, 0, 9)]
    assert calculos.ordenar_por_prioridad(items) == calculos.ordenar_por_prioridad(list(reversed(items)))
