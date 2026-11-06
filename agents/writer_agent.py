"""Agente Redactor: fase de sintesis final.

No tiene herramientas: redacta el informe SOLO con los datos validados que le pasan el
investigador y el analista. Que el supervisor no redacte es a proposito: el supervisor
coordina y evalua, no ejecuta (evita el error del "supervisor todologo").
"""

import json
from typing import Any

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.runnables import RunnableConfig

from agents.base import descripcion_de_falla, entrada_con_correcciones, texto_de
from agents.research_agent import NodoAsincrono
from errores import es_error_irrecuperable
from state import Analisis, Contribucion, EstadoOrquestador, Hallazgos, NombreAgente

NOMBRE: NombreAgente = "redactor"

PROMPT_REDACTOR = """Sos el Agente Redactor de un equipo de Gobierno de Datos de Mercado Austral S.A., una empresa
ficticia de retail. Recibís datos YA recolectados y calculados por otros agentes y redactás
el informe final para quien hizo la consulta.

Reglas de trabajo:
1. Usá SOLO los datos que te pasan. No agregues números, IDs, fechas ni responsables que no estén
   en ellos, y no calcules nada nuevo.
2. Citá los IDs (INC-NNNN y R-XXX-NN) de todos los incidentes que analizaron los otros agentes y
   de sus reglas.
3. Si hay un ranking de prioridad, el primer incidente del ranking es el que hay que atacar
   primero: nombralo explícitamente en el informe y explicá por qué con los números recibidos
   (severidad, horas pasadas del plazo, brecha de la regla).
4. Si los datos dicen que la consulta quedó fuera de alcance o que no hay incidentes, decilo
   claramente y sin inventar nada; no cites ningún ID.
5. Del estado del plazo (SLA) decí solo "vencido" (con las horas de exceso recibidas) o "dentro
   del plazo". No inventes categorías intermedias como "en riesgo". Las brechas están en puntos
   porcentuales (pp), no en %.
6. Formato: español rioplatense, breve y directo. Un párrafo de conclusión y después una lista
   corta, un ítem por incidente, en orden de prioridad. Sin opiniones que no se desprendan de
   los datos."""


def datos_para_el_redactor(hallazgos: Hallazgos, analisis: Analisis | None) -> str:
    """Los datos validados que necesita el redactor, en JSON (sin historial ni mensajes)."""
    datos: dict[str, Any] = {
        "fuera_de_alcance": hallazgos.fuera_de_alcance,
        "resumen_del_investigador": hallazgos.resumen,
        "fecha_corte": hallazgos.fecha_corte.isoformat() if hallazgos.fecha_corte else None,
        "incidentes": [i.model_dump(mode="json") for i in hallazgos.incidentes_detallados()],
        "reglas": [r.model_dump(mode="json") for r in hallazgos.reglas],
    }
    if analisis is not None:
        datos["analisis"] = {
            "brechas": [b.model_dump(mode="json") for b in analisis.brechas],
            "vencimientos": [v.model_dump(mode="json") for v in analisis.vencimientos],
            "ranking_de_prioridad": analisis.ranking,
        }
    return json.dumps(datos, ensure_ascii=False, indent=2)


def crear_nodo_redactor(llm: BaseChatModel) -> NodoAsincrono:
    """Devuelve el nodo del grafo principal que corre al redactor."""

    async def nodo_redactor(estado: EstadoOrquestador, config: RunnableConfig) -> dict[str, Any]:
        hallazgos = estado["hallazgos"] or Hallazgos()
        instruccion = entrada_con_correcciones(estado["instruccion"], estado, NOMBRE)
        entrada = (
            f"CONSULTA ORIGINAL: {estado['tarea']}\n\n{instruccion}\n\n"
            f"DATOS VALIDADOS:\n{datos_para_el_redactor(hallazgos, estado['analisis'])}"
        )
        error: str | None = None
        try:
            respuesta = await llm.ainvoke(
                [SystemMessage(content=PROMPT_REDACTOR), HumanMessage(content=entrada)], config
            )
            informe = texto_de(respuesta)
        except Exception as falla:  # noqa: BLE001 - un fallo transitorio del modelo no tira abajo el grafo
            if es_error_irrecuperable(falla):
                raise
            informe, error = "", descripcion_de_falla(falla)
        contribucion = Contribucion(
            agente=NOMBRE, paso=estado["pasos"], resumen=informe or error or "Sin respuesta.", error=error
        )
        return {
            "informe": informe,
            "ultimo_agente": NOMBRE,
            "contribuciones": [contribucion],
            "messages": [AIMessage(content=contribucion.resumen, name=NOMBRE)],
        }

    return nodo_redactor
