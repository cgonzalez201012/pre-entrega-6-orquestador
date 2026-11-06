"""Grafo principal: el Supervisor y sus especialistas.

                         +--------------+
        START ---------> |  supervisor  | <----------------------+
                         +------+-------+                        |
        (arista condicional,    |                                 |
         `decidir_ruta` con     +--> investigador --+             |
         retorno `Literal`)     +--> analista -------+--> validacion
                                +--> redactor ------+
                                |
                                +--> FINISH --> END

- `supervisor` decide quien interviene. Es el unico nodo con aristas condicionales.
- Cada especialista, al terminar, pasa SIEMPRE por `validacion`, que aplica la rubrica de
  controles y deja su veredicto en el estado; despues el control vuelve al supervisor.
- El supervisor lee el veredicto y decide: pedir un refinamiento, seguir con el proximo
  especialista o cerrar.
"""

from typing import Any, Literal

from langchain_core.language_models import BaseChatModel
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph

from agents.analyst_agent import crear_nodo_analista
from agents.research_agent import crear_nodo_investigador
from agents.writer_agent import crear_nodo_redactor
from config import Configuracion
from state import EstadoOrquestador
from supervisor import crear_nodo_supervisor
from validacion import nodo_validacion

NODO_SUPERVISOR = "supervisor"
NODO_INVESTIGADOR = "investigador"
NODO_ANALISTA = "analista"
NODO_REDACTOR = "redactor"
NODO_VALIDACION = "validacion"

ESPECIALISTAS = (NODO_INVESTIGADOR, NODO_ANALISTA, NODO_REDACTOR)


def decidir_ruta(estado: EstadoOrquestador) -> Literal["investigador", "analista", "redactor", "FINISH"]:
    """Funcion de las aristas condicionales: lee el destino que dejo el supervisor en el estado.

    El tipo de retorno `Literal` enumera los unicos destinos posibles; el grafo no tiene
    ninguna ruta fija entre los especialistas.
    """
    return estado["siguiente"]


def construir_grafo(
    llm_supervisor: BaseChatModel,
    llm_especialistas: BaseChatModel,
    *,
    max_pasos: int,
    max_intentos: int,
    recursion_especialista: int,
) -> CompiledStateGraph:
    """Arma y compila el grafo con los modelos que se le pasen (reales o falsos en los tests)."""
    grafo = StateGraph(EstadoOrquestador)

    nodos: dict[str, Any] = {
        NODO_SUPERVISOR: crear_nodo_supervisor(llm_supervisor, max_pasos, max_intentos),
        NODO_INVESTIGADOR: crear_nodo_investigador(llm_especialistas, recursion_especialista),
        NODO_ANALISTA: crear_nodo_analista(llm_especialistas, recursion_especialista),
        NODO_REDACTOR: crear_nodo_redactor(llm_especialistas),
        NODO_VALIDACION: nodo_validacion,
    }
    for nombre, nodo in nodos.items():
        grafo.add_node(nombre, nodo)

    grafo.add_edge(START, NODO_SUPERVISOR)
    grafo.add_conditional_edges(
        NODO_SUPERVISOR,
        decidir_ruta,
        {
            "investigador": NODO_INVESTIGADOR,
            "analista": NODO_ANALISTA,
            "redactor": NODO_REDACTOR,
            "FINISH": END,
        },
    )
    for especialista in ESPECIALISTAS:
        grafo.add_edge(especialista, NODO_VALIDACION)
    grafo.add_edge(NODO_VALIDACION, NODO_SUPERVISOR)
    return grafo.compile(name="orquestador")


def construir_grafo_desde_config(
    llm_supervisor: BaseChatModel, llm_especialistas: BaseChatModel, config: Configuracion
) -> CompiledStateGraph:
    """Igual que `construir_grafo`, tomando los limites de la configuracion."""
    return construir_grafo(
        llm_supervisor,
        llm_especialistas,
        max_pasos=config.max_pasos,
        max_intentos=config.max_intentos_por_agente,
        recursion_especialista=config.recursion_especialista,
    )
