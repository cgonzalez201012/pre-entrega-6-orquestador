"""Pieza comun de los especialistas: un ciclo ReAct propio y utilidades para leer sus mensajes.

Cada especialista es un pequeno `StateGraph` con el mismo ciclo ReAct que use en la
entrega 5 (modelo -> herramientas -> modelo ...). No uso `create_react_agent` porque en
LangGraph 1.x esta deprecado (se mudo a `langchain.agents.create_agent`); armar el ciclo
a mano deja a la vista el grafo y no suma otra dependencia.

Cada especialista corre en un contexto AISLADO: recibe un solo mensaje (su instruccion y
los datos que necesita), nunca el historial completo del grafo principal.
"""

import json
from collections.abc import Collection, Sequence
from dataclasses import dataclass
from typing import Any

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import BaseTool
from langgraph.errors import GraphRecursionError
from langgraph.graph import END, START, MessagesState, StateGraph
from langgraph.graph.state import CompiledStateGraph
from langgraph.prebuilt import ToolNode, tools_condition

from state import EstadoOrquestador

NODO_MODELO = "modelo"
NODO_HERRAMIENTAS = "herramientas"


def construir_agente_react(
    llm: BaseChatModel, herramientas: Sequence[BaseTool], prompt_sistema: str, nombre: str
) -> CompiledStateGraph:
    """Arma el ciclo ReAct de un especialista con sus herramientas acotadas."""
    llm_con_herramientas = llm.bind_tools(list(herramientas))

    async def llamar_modelo(estado: MessagesState) -> dict[str, Any]:
        contexto = [SystemMessage(content=prompt_sistema), *estado["messages"]]
        respuesta = await llm_con_herramientas.ainvoke(contexto)
        return {"messages": [respuesta]}

    grafo = StateGraph(MessagesState)
    # Los tipos de langgraph 1.2 no reconocen una funcion async como nodo (mypy), aunque
    # es el uso documentado: por eso el ignore acotado a esta linea.
    grafo.add_node(NODO_MODELO, llamar_modelo)  # type: ignore[call-overload]
    # handle_tool_errors=True: una excepcion inesperada de una herramienta vuelve al modelo
    # como texto en vez de romper el grafo.
    grafo.add_node(NODO_HERRAMIENTAS, ToolNode(list(herramientas), handle_tool_errors=True))
    grafo.add_edge(START, NODO_MODELO)
    grafo.add_conditional_edges(NODO_MODELO, tools_condition, {"tools": NODO_HERRAMIENTAS, END: END})
    grafo.add_edge(NODO_HERRAMIENTAS, NODO_MODELO)  # el ciclo: observar y volver a razonar
    return grafo.compile(name=nombre)


@dataclass(frozen=True)
class CorridaEspecialista:
    """Resultado de correr un especialista: sus mensajes y, si fallo, por que."""

    mensajes: list[BaseMessage]
    error: str | None = None


async def ejecutar_agente_react(
    subgrafo: CompiledStateGraph,
    entrada: str,
    recursion_limit: int,
    config: RunnableConfig | None = None,
) -> CorridaEspecialista:
    """Corre un especialista con UN solo mensaje de entrada (contexto aislado).

    Si el especialista se pasa del limite de pasos devuelve el error en lugar de levantar
    la excepcion: el nodo la convierte en un aporte vacio que la validacion va a rechazar,
    y el supervisor decide que hacer.
    """
    configuracion: RunnableConfig = {**(config or {}), "recursion_limit": recursion_limit}
    try:
        salida = await subgrafo.ainvoke({"messages": [HumanMessage(content=entrada)]}, configuracion)
    except GraphRecursionError:
        return CorridaEspecialista(
            mensajes=[],
            error=f"El especialista no terminó dentro del límite de {recursion_limit} pasos.",
        )
    return CorridaEspecialista(mensajes=list(salida["messages"]))


def entrada_con_correcciones(instruccion: str, estado: EstadoOrquestador, agente: str) -> str:
    """Agrega a la instruccion los problemas que marco la validacion la ultima vez.

    Se hace en codigo (no depende de que el supervisor los copie) para que un refinamiento
    siempre le llegue al especialista con el motivo exacto del rechazo.
    """
    validacion = estado.get("validacion")
    if validacion is None or validacion.agente != agente or validacion.aprobada:
        return instruccion
    problemas = "\n".join(f"- {problema}" for problema in validacion.problemas)
    return f"{instruccion}\n\nTu aporte anterior fue rechazado por estos motivos. Corregilos:\n{problemas}"


def texto_de(mensaje: BaseMessage) -> str:
    """Texto plano de un mensaje (Claude puede devolver una lista de bloques)."""
    contenido = mensaje.content
    if isinstance(contenido, str):
        return contenido.strip()
    partes: list[str] = []
    for bloque in contenido:
        if isinstance(bloque, str):
            partes.append(bloque)
        elif isinstance(bloque, dict) and bloque.get("type") == "text":
            partes.append(str(bloque.get("text", "")))
    return "".join(partes).strip()


def texto_final(mensajes: Sequence[BaseMessage]) -> str:
    """Texto del ultimo mensaje del modelo (vacio si no hay)."""
    for mensaje in reversed(mensajes):
        if isinstance(mensaje, AIMessage):
            return texto_de(mensaje)
    return ""


def resultados_de(mensajes: Sequence[BaseMessage], herramienta: str) -> list[Any]:
    """Resultados (JSON ya interpretado) de todas las llamadas a una herramienta, en orden.

    Si una herramienta devolvio texto que no es JSON (por ejemplo el error de validacion de
    argumentos que arma el `ToolNode`) se devuelve como un dict con la clave 'error'.
    """
    resultados: list[Any] = []
    for mensaje in mensajes:
        if isinstance(mensaje, ToolMessage) and mensaje.name == herramienta:
            texto = texto_de(mensaje)
            try:
                resultados.append(json.loads(texto))
            except json.JSONDecodeError:
                resultados.append({"error": texto})
    return resultados


def nombres_de_herramientas(mensajes: Sequence[BaseMessage], propias: Collection[str]) -> list[str]:
    """Nombres de las herramientas PROPIAS que el modelo pidio, en orden y con repeticiones.

    Si el modelo pide una herramienta que el agente no tiene, el `ToolNode` la rechaza: esa
    llamada no cuenta como evidencia de trabajo.
    """
    nombres: list[str] = []
    for mensaje in mensajes:
        if isinstance(mensaje, AIMessage):
            nombres.extend(str(llamada["name"]) for llamada in mensaje.tool_calls if llamada["name"] in propias)
    return nombres


def es_error(resultado: Any) -> bool:
    """True si el resultado de una herramienta es un error (JSON con la clave 'error')."""
    return isinstance(resultado, dict) and "error" in resultado
