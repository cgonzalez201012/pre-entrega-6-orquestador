"""Calculos deterministas del analista.

Son funciones puras (sin modelo, sin red): las herramientas del analista las exponen y el
nodo de validacion las vuelve a ejecutar con los datos del investigador para comprobar que
el analista no se equivoco al pasarle los valores a la herramienta.

Convenciones (supuestos documentados en el README):
- Dias habiles: de lunes a viernes, sin descontar feriados (los datos son ficticios).
- Una jornada habil son 8 horas, y los dias habiles transcurridos se cuentan desde la
  fecha de alta (inclusive) hasta la fecha de corte (exclusive).
- Prioridad: primero la severidad, despues si el SLA esta vencido, despues cuantas horas
  se paso del plazo y por ultimo cuantos registros afecta. El desempate final es el ID.
"""

from collections.abc import Sequence
from datetime import date, timedelta

from errores import ErrorCalculo
from state import BrechaRegla, ItemPrioridad, Severidad, VencimientoSLA

JORNADA_HORAS = 8
ORDEN_SEVERIDAD: dict[str, int] = {"critica": 4, "alta": 3, "media": 2, "baja": 1}


def dias_habiles_entre(desde: date, hasta: date) -> int:
    """Cuenta los dias de lunes a viernes en el intervalo [desde, hasta)."""
    if hasta < desde:
        raise ErrorCalculo(
            f"La fecha de corte ({hasta}) es anterior a la fecha de alta ({desde}).",
            "Pasá fecha_alta y fecha_corte en formato AAAA-MM-DD, con la fecha de alta primero.",
        )
    total_dias = (hasta - desde).days
    return sum(1 for n in range(total_dias) if (desde + timedelta(days=n)).weekday() < 5)


def calcular_vencimiento(
    incidente_id: str,
    severidad: Severidad,
    fecha_alta: date,
    fecha_corte: date,
    sla_horas_habiles: int,
) -> VencimientoSLA:
    """Compara las horas habiles transcurridas con el plazo (SLA) del incidente."""
    dias = dias_habiles_entre(fecha_alta, fecha_corte)
    horas = dias * JORNADA_HORAS
    return VencimientoSLA(
        incidente_id=incidente_id,
        severidad=severidad,
        fecha_alta=fecha_alta,
        fecha_corte=fecha_corte,
        dias_habiles=dias,
        horas_habiles_transcurridas=horas,
        sla_horas_habiles=sla_horas_habiles,
        exceso_horas=max(0, horas - sla_horas_habiles),
        vencido=horas > sla_horas_habiles,
    )


def calcular_brecha(regla_id: str, medicion_pct: float, umbral_pct: float) -> BrechaRegla:
    """Distancia (en puntos porcentuales) entre la medicion actual y el umbral de la regla."""
    return BrechaRegla(
        regla_id=regla_id,
        medicion_pct=medicion_pct,
        umbral_pct=umbral_pct,
        brecha_pp=round(medicion_pct - umbral_pct, 2),
        cumple=medicion_pct >= umbral_pct,
    )


def ordenar_por_prioridad(items: Sequence[ItemPrioridad]) -> list[str]:
    """Devuelve los IDs de incidente del mas urgente al menos urgente."""

    def clave(item: ItemPrioridad) -> tuple[int, int, int, int, str]:
        return (
            -ORDEN_SEVERIDAD[item.severidad],
            0 if item.vencido else 1,
            -item.exceso_horas,
            -item.registros_afectados,
            item.incidente_id,
        )

    return [item.incidente_id for item in sorted(items, key=clave)]
