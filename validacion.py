"""Nodo de validacion: la rubrica que revisa cada aporte antes de que el supervisor decida.

Despues de cada especialista corre este nodo. Es codigo determinista (sin modelo): aplica
una rubrica de controles sobre lo que el agente acaba de entregar y devuelve un veredicto
con los problemas encontrados. El supervisor lee ese veredicto para decidir si pide un
refinamiento, sigue con el proximo especialista o cierra.

Idea central (la misma leccion de las entregas anteriores): validar la FORMA no es validar
la VERDAD. Pydantic garantiza que el aporte tiene la forma correcta; estos controles
comprueban que el contenido coincide con los datos (por ejemplo, el analista no puede
haberle pasado a una herramienta una fecha distinta de la que encontro el investigador).

Rubrica por agente:

  investigador: consulto herramientas (o declaro FUERA_DE_ALCANCE); si listo incidentes,
                pidio el detalle de al menos uno; cada incidente detallado tiene su regla
                consultada y hay fecha de corte.
  analista:     calculo el vencimiento de CADA incidente detallado y la brecha de CADA
                regla, el ranking incluye exactamente esos incidentes, y todos los numeros
                coinciden con los que salen de recalcular con los datos del investigador.
  redactor:     cito los IDs de todos los incidentes analizados, nombro el de mayor
                prioridad y no cito ningun ID que no exista en los datos.
"""

import re
from typing import Any

import calculos
from state import (
    Analisis,
    EstadoOrquestador,
    Hallazgos,
    ItemPrioridad,
    NombreAgente,
    ResultadoValidacion,
    VencimientoSLA,
)

PATRON_IDS = re.compile(r"\b(?:INC-\d{4}|R-[A-Z]{3}-\d{2})\b")


def validar_investigador(hallazgos: Hallazgos | None) -> list[str]:
    """Problemas del aporte del investigador (lista vacia = aprobado)."""
    if hallazgos is None:
        return ["No hay hallazgos: el investigador no entregó nada."]
    if hallazgos.fuera_de_alcance:
        return []  # salida explicita y valida: la consulta no se puede responder con las herramientas

    problemas: list[str] = []
    if not hallazgos.herramientas_usadas:
        problemas.append(
            "No llamaste a ninguna herramienta: los datos tienen que salir de las herramientas. "
            "Si la consulta no se puede responder con ellas, respondé FUERA_DE_ALCANCE."
        )
        return problemas

    detallados = hallazgos.incidentes_detallados()
    if hallazgos.incidentes and not detallados:
        problemas.append(
            "Listaste incidentes pero no pediste el detalle de ninguno: faltan la regla, "
            "los registros afectados y el SLA. Llamá a detalle_incidente."
        )
    for incidente in detallados:
        if incidente.regla_id is not None and hallazgos.regla(incidente.regla_id) is None:
            problemas.append(f"Falta consultar la regla {incidente.regla_id} del incidente {incidente.id}.")
    if detallados and hallazgos.fecha_corte is None:
        problemas.append("No se pudo determinar la fecha de corte de los datos.")
    if not hallazgos.incidentes and hallazgos.consultas_con_error:
        problemas.append(
            "Todas tus consultas dieron error: " + " | ".join(hallazgos.consultas_con_error) + " Corregí las llamadas."
        )
    return problemas


def _items_de_prioridad(hallazgos: Hallazgos, vencimientos: dict[str, VencimientoSLA]) -> list[ItemPrioridad]:
    items: list[ItemPrioridad] = []
    for incidente in hallazgos.incidentes_detallados():
        vencimiento = vencimientos[incidente.id]
        items.append(
            ItemPrioridad(
                incidente_id=incidente.id,
                severidad=incidente.severidad,
                vencido=vencimiento.vencido,
                exceso_horas=vencimiento.exceso_horas,
                registros_afectados=incidente.registros_afectados or 0,
            )
        )
    return items


def validar_analista(hallazgos: Hallazgos | None, analisis: Analisis | None) -> list[str]:
    """Problemas del aporte del analista, recalculando todo con los datos del investigador."""
    if analisis is None:
        return ["No hay análisis: el analista no entregó nada."]
    if hallazgos is None or not hallazgos.incidentes_detallados() or hallazgos.fecha_corte is None:
        return ["No hay incidentes detallados para analizar: el investigador tiene que entregarlos primero."]

    problemas: list[str] = []
    detallados = hallazgos.incidentes_detallados()
    esperados_vencimientos: dict[str, VencimientoSLA] = {}
    for incidente in detallados:
        if incidente.sla_horas_habiles is None:
            continue  # un incidente detallado siempre lo tiene; esto solo tranquiliza al chequeo de tipos
        esperado = calculos.calcular_vencimiento(
            incidente.id, incidente.severidad, incidente.fecha_alta, hallazgos.fecha_corte, incidente.sla_horas_habiles
        )
        esperados_vencimientos[incidente.id] = esperado
        obtenido = next((v for v in analisis.vencimientos if v.incidente_id == incidente.id), None)
        if obtenido is None:
            problemas.append(f"Falta calcular el vencimiento del SLA del incidente {incidente.id}.")
        elif obtenido != esperado:
            problemas.append(
                f"El vencimiento de {incidente.id} no coincide con los datos recolectados: "
                f"calculaste {obtenido.model_dump(mode='json')} y corresponde {esperado.model_dump(mode='json')}. "
                "Usá exactamente fecha_alta, fecha_corte, severidad y sla_horas_habiles de los datos."
            )

    for regla_id in sorted({i.regla_id for i in detallados if i.regla_id}):
        regla = hallazgos.regla(regla_id)
        if regla is None:
            continue  # ya lo reprocho la validacion del investigador
        esperada = calculos.calcular_brecha(regla.id, regla.medicion_actual_pct, regla.umbral_aceptacion_pct)
        obtenida = next((b for b in analisis.brechas if b.regla_id == regla_id), None)
        if obtenida is None:
            problemas.append(f"Falta calcular la brecha de la regla {regla_id}.")
        elif obtenida != esperada:
            problemas.append(
                f"La brecha de {regla_id} no coincide con los datos: calculaste {obtenida.model_dump(mode='json')} "
                f"y corresponde {esperada.model_dump(mode='json')}. Usá la medición y el umbral exactos."
            )

    ids_detallados = {i.id for i in detallados}
    sobrantes = {v.incidente_id for v in analisis.vencimientos} - ids_detallados
    if sobrantes:
        problemas.append(f"Calculaste incidentes que no están en los datos: {', '.join(sorted(sobrantes))}.")

    ranking_esperado = calculos.ordenar_por_prioridad(_items_de_prioridad(hallazgos, esperados_vencimientos))
    if not analisis.ranking:
        problemas.append("Falta el ranking de prioridad: llamá a priorizar_incidentes con todos los incidentes.")
    elif analisis.ranking != ranking_esperado:
        problemas.append(
            f"El ranking de prioridad no coincide: obtuviste {analisis.ranking} y corresponde {ranking_esperado}. "
            "Pasá a priorizar_incidentes todos los incidentes con los valores de sus vencimientos."
        )
    return problemas


def validar_redactor(hallazgos: Hallazgos | None, analisis: Analisis | None, informe: str | None) -> list[str]:
    """Problemas del informe final."""
    if not informe or not informe.strip():
        return ["El informe está vacío."]

    problemas: list[str] = []
    citados = set(PATRON_IDS.findall(informe))
    conocidos = hallazgos.ids_conocidos() if hallazgos is not None else set()
    inventados = citados - conocidos
    if inventados:
        problemas.append(f"El informe cita IDs que no existen en los datos: {', '.join(sorted(inventados))}.")

    if hallazgos is not None and analisis is not None and not hallazgos.fuera_de_alcance:
        faltantes = {i.id for i in hallazgos.incidentes_detallados()} - citados
        if faltantes:
            problemas.append(f"El informe no cubre estos incidentes analizados: {', '.join(sorted(faltantes))}.")
        if analisis.ranking and analisis.ranking[0] not in citados:
            problemas.append(f"El informe no nombra el incidente de mayor prioridad ({analisis.ranking[0]}).")
    return problemas


def validar(estado: EstadoOrquestador) -> ResultadoValidacion:
    """Aplica la rubrica que corresponde al ultimo agente que entrego su aporte."""
    agente: NombreAgente | None = estado["ultimo_agente"]
    paso = estado["pasos"]
    if agente is None:
        raise ValueError("No hay nada para validar: ningun agente entrego su aporte todavia.")
    if agente == "investigador":
        problemas = validar_investigador(estado["hallazgos"])
    elif agente == "analista":
        problemas = validar_analista(estado["hallazgos"], estado["analisis"])
    else:
        problemas = validar_redactor(estado["hallazgos"], estado["analisis"], estado["informe"])

    # Si el especialista no termino dentro de su limite de pasos, su aporte esta incompleto.
    ultima = next((c for c in reversed(estado["contribuciones"]) if c.agente == agente), None)
    if ultima is not None and ultima.error:
        problemas.insert(0, ultima.error)
    return ResultadoValidacion(agente=agente, aprobada=not problemas, problemas=problemas, paso=paso)


async def nodo_validacion(estado: EstadoOrquestador) -> dict[str, Any]:
    """Nodo del grafo: escribe solo `validacion` y anexa a `validaciones`."""
    resultado = validar(estado)
    return {"validacion": resultado, "validaciones": [resultado]}
