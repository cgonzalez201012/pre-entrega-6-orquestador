"""Agente Analista: calcula con herramientas matematicas lo que el investigador recolecto.

Es el especialista de "analisis/computo". Recibe los datos YA recolectados (no consulta la
base) y usa tres herramientas de calculo deterministas (`calculos.py`):

- brecha de una regla contra su umbral,
- vencimiento del plazo (SLA) de un incidente en horas habiles,
- ranking de prioridad de los incidentes.

Un modelo de lenguaje no deberia hacer cuentas de cabeza: por eso cada numero sale de una
herramienta y el nodo de validacion lo recalcula con los datos del investigador para
detectar si el analista le paso mal un valor a la herramienta.
"""

import json
from datetime import date
from typing import Any

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import BaseTool, tool
from pydantic import BaseModel, Field, ValidationError

import calculos
from agents.base import (
    construir_agente_react,
    ejecutar_agente_react,
    entrada_con_correcciones,
    es_error,
    nombres_de_herramientas,
    resultados_de,
    texto_final,
)
from agents.research_agent import NodoAsincrono
from errores import ErrorCalculo
from state import (
    Analisis,
    BrechaRegla,
    Contribucion,
    EstadoOrquestador,
    Hallazgos,
    ItemPrioridad,
    NombreAgente,
    Severidad,
    VencimientoSLA,
)

NOMBRE: NombreAgente = "analista"

PROMPT_ANALISTA = """Sos el Agente Analista de un equipo de Gobierno de Datos de Mercado Austral S.A., una empresa
ficticia de retail. Recibís datos YA recolectados por el investigador y tu único trabajo es
CALCULAR con tus herramientas. No buscás datos nuevos y no inventás ninguno.

Reglas de trabajo:
1. No hagas cuentas de cabeza: todo número que informes tiene que salir de una herramienta.
2. Para CADA incidente de la lista llamá a calcular_vencimiento_sla con sus datos exactos
   (incidente_id, severidad, fecha_alta, fecha_corte y sla_horas_habiles).
3. Para CADA regla de la lista llamá a calcular_brecha_regla con la medición y el umbral exactos.
4. Después, UNA sola llamada a priorizar_incidentes con todos los incidentes, usando para cada
   uno el resultado de su vencimiento (vencido y exceso_horas), su severidad y sus
   registros_afectados.
5. Copiá los valores tal cual vienen en los datos: no los redondees ni los cambies.
6. Al terminar, respondé con un resumen breve de lo que calculaste y lo que resultó (con los IDs
   y los números que devolvieron las herramientas). Sin opiniones ni recomendaciones."""


# ---------- Herramientas ----------


def _a_texto(resultado: dict[str, Any]) -> str:
    return json.dumps(resultado, ensure_ascii=False)


def _error_a_texto(error: ErrorCalculo) -> str:
    return _a_texto({"error": error.mensaje, "ayuda": error.ayuda})


class BrechaReglaArgs(BaseModel):
    """Argumentos de calcular_brecha_regla."""

    regla_id: str = Field(description="ID de la regla, con formato R-XXX-NN (por ejemplo 'R-ART-01').")
    medicion_pct: float = Field(description="Medicion actual de la regla, en porcentaje (por ejemplo 96.8).")
    umbral_pct: float = Field(description="Umbral de aceptacion de la regla, en porcentaje (por ejemplo 99.5).")


class VencimientoSlaArgs(BaseModel):
    """Argumentos de calcular_vencimiento_sla."""

    incidente_id: str = Field(description="ID del incidente, con formato INC-NNNN.")
    severidad: Severidad = Field(description="Severidad del incidente: 'critica', 'alta', 'media' o 'baja'.")
    fecha_alta: date = Field(description="Fecha de alta del incidente, formato AAAA-MM-DD.")
    fecha_corte: date = Field(description="Fecha de corte de los datos, formato AAAA-MM-DD.")
    sla_horas_habiles: int = Field(gt=0, description="Plazo (SLA) del incidente, en horas habiles.")


class PriorizarArgs(BaseModel):
    """Argumentos de priorizar_incidentes."""

    items: list[ItemPrioridad] = Field(
        min_length=1,
        description=(
            "Un elemento por incidente, con: incidente_id, severidad, vencido (true/false), "
            "exceso_horas (horas habiles pasadas del plazo, 0 si no vencio) y registros_afectados."
        ),
    )


@tool("calcular_brecha_regla", args_schema=BrechaReglaArgs)
async def calcular_brecha_regla(regla_id: str, medicion_pct: float, umbral_pct: float) -> str:
    """Calcula cuántos puntos porcentuales le faltan (o le sobran) a una regla respecto de su umbral.

    Usala UNA vez por cada regla de calidad con los valores exactos de su medicion actual y su
    umbral de aceptacion. La brecha es medicion - umbral: negativa significa que la regla esta
    por debajo del umbral y no cumple.

    Devuelve un JSON con regla_id, medicion_pct, umbral_pct, brecha_pp y cumple.
    """
    resultado = calculos.calcular_brecha(regla_id, medicion_pct, umbral_pct)
    return _a_texto(resultado.model_dump(mode="json"))


@tool("calcular_vencimiento_sla", args_schema=VencimientoSlaArgs)
async def calcular_vencimiento_sla(
    incidente_id: str, severidad: Severidad, fecha_alta: date, fecha_corte: date, sla_horas_habiles: int
) -> str:
    """Calcula si el plazo (SLA) de un incidente esta vencido y por cuántas horas habiles.

    Usala UNA vez por cada incidente con sus datos exactos. Cuenta los dias de lunes a viernes
    entre la fecha de alta y la fecha de corte (una jornada son 8 horas habiles, sin descontar
    feriados) y los compara con el SLA del incidente.

    Devuelve un JSON con dias_habiles, horas_habiles_transcurridas, sla_horas_habiles,
    exceso_horas y vencido. Si las fechas estan al reves devuelve un JSON con 'error' y 'ayuda'.
    """
    try:
        resultado = calculos.calcular_vencimiento(incidente_id, severidad, fecha_alta, fecha_corte, sla_horas_habiles)
    except ErrorCalculo as error:
        return _error_a_texto(error)
    return _a_texto(resultado.model_dump(mode="json"))


@tool("priorizar_incidentes", args_schema=PriorizarArgs)
async def priorizar_incidentes(items: list[ItemPrioridad]) -> str:
    """Ordena los incidentes del más urgente al menos urgente.

    Usala UNA sola vez, al final, con todos los incidentes y despues de haber calculado el
    vencimiento de cada uno (necesitas vencido y exceso_horas). El criterio es: primero la
    severidad, despues si el SLA esta vencido, despues las horas pasadas del plazo y por
    ultimo los registros afectados.

    Devuelve un JSON con 'ranking': la lista de IDs de incidente, el primero es el mas urgente.
    """
    return _a_texto({"ranking": calculos.ordenar_por_prioridad(items)})


HERRAMIENTAS_ANALISTA: list[BaseTool] = [calcular_brecha_regla, calcular_vencimiento_sla, priorizar_incidentes]


# ---------- Del resultado de las herramientas al contrato `Analisis` ----------


def construir_analisis(mensajes: list[BaseMessage], resumen: str) -> Analisis:
    """Arma `Analisis` leyendo los resultados de las herramientas (no el texto del modelo)."""
    errores: list[str] = []
    brechas: dict[str, BrechaRegla] = {}
    vencimientos: dict[str, VencimientoSLA] = {}
    ranking: list[str] = []

    for resultado in resultados_de(mensajes, "calcular_brecha_regla"):
        if es_error(resultado):
            errores.append(f"calcular_brecha_regla: {resultado['error']}")
            continue
        try:
            brecha = BrechaRegla.model_validate(resultado)
        except ValidationError:
            errores.append("calcular_brecha_regla: respuesta con formato inesperado")
            continue
        brechas[brecha.regla_id] = brecha

    for resultado in resultados_de(mensajes, "calcular_vencimiento_sla"):
        if es_error(resultado):
            errores.append(f"calcular_vencimiento_sla: {resultado['error']}")
            continue
        try:
            vencimiento = VencimientoSLA.model_validate(resultado)
        except ValidationError:
            errores.append("calcular_vencimiento_sla: respuesta con formato inesperado")
            continue
        vencimientos[vencimiento.incidente_id] = vencimiento

    for resultado in resultados_de(mensajes, "priorizar_incidentes"):
        if es_error(resultado) or not isinstance(resultado.get("ranking"), list):
            errores.append("priorizar_incidentes: " + str(resultado.get("error", "respuesta con formato inesperado")))
            continue
        ranking = [str(incidente_id) for incidente_id in resultado["ranking"]]  # gana la ultima llamada valida

    return Analisis(
        brechas=list(brechas.values()),
        vencimientos=list(vencimientos.values()),
        ranking=ranking,
        errores=errores,
        herramientas_usadas=nombres_de_herramientas(mensajes, {h.name for h in HERRAMIENTAS_ANALISTA}),
        resumen=resumen,
    )


def datos_para_el_analista(hallazgos: Hallazgos) -> str:
    """Los datos que necesita el analista, y solo esos, en JSON (contexto acotado)."""
    detallados = hallazgos.incidentes_detallados()
    regla_ids = {incidente.regla_id for incidente in detallados}
    datos = {
        "fecha_corte": hallazgos.fecha_corte.isoformat() if hallazgos.fecha_corte else None,
        "incidentes": [
            {
                "incidente_id": i.id,
                "severidad": i.severidad,
                "fecha_alta": i.fecha_alta.isoformat(),
                "regla_id": i.regla_id,
                "registros_afectados": i.registros_afectados,
                "sla_horas_habiles": i.sla_horas_habiles,
            }
            for i in detallados
        ],
        "reglas": [
            {"regla_id": r.id, "medicion_pct": r.medicion_actual_pct, "umbral_pct": r.umbral_aceptacion_pct}
            for r in hallazgos.reglas
            if r.id in regla_ids
        ],
    }
    return json.dumps(datos, ensure_ascii=False, indent=2)


# ---------- Nodo del grafo ----------


def crear_nodo_analista(llm: BaseChatModel, recursion_limit: int) -> NodoAsincrono:
    """Devuelve el nodo del grafo principal que corre al analista."""
    subgrafo = construir_agente_react(llm, HERRAMIENTAS_ANALISTA, PROMPT_ANALISTA, NOMBRE)

    async def nodo_analista(estado: EstadoOrquestador, config: RunnableConfig) -> dict[str, Any]:
        hallazgos = estado["hallazgos"] or Hallazgos()
        # Contexto aislado: la instruccion y los datos del investigador. Nada del historial.
        instruccion = entrada_con_correcciones(estado["instruccion"], estado, NOMBRE)
        entrada = f"{instruccion}\n\nDATOS RECOLECTADOS POR EL INVESTIGADOR:\n{datos_para_el_analista(hallazgos)}"
        corrida = await ejecutar_agente_react(subgrafo, entrada, recursion_limit, config)
        resumen = texto_final(corrida.mensajes)
        de_esta_corrida = construir_analisis(corrida.mensajes, resumen)
        contribucion = Contribucion(
            agente=NOMBRE,
            paso=estado["pasos"],
            resumen=resumen or (corrida.error or "Sin respuesta."),
            herramientas_usadas=de_esta_corrida.herramientas_usadas,
            error=corrida.error,
        )
        # Un refinamiento completa lo calculado antes; no lo reemplaza.
        anterior = estado["analisis"]
        analisis = anterior.fusionar(de_esta_corrida) if anterior is not None else de_esta_corrida
        return {
            "analisis": analisis,
            "ultimo_agente": NOMBRE,
            "contribuciones": [contribucion],
            "messages": [AIMessage(content=contribucion.resumen, name=NOMBRE)],
        }

    return nodo_analista
