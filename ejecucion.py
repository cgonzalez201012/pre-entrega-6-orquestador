"""Armado de los modelos reales y ejecucion de una tarea de punta a punta.

Es lo que usan la linea de comandos (`main.py`) y el notebook de demostracion.
"""

import time
from pathlib import Path

from langchain_anthropic import ChatAnthropic
from langgraph.graph.state import CompiledStateGraph

from config import BASE_DIR, Configuracion
from graph import construir_grafo_desde_config
from state import estado_inicial
from trazas import Corrida, ContadorDeUso, corrida_desde_estado

RUTA_DIAGRAMA = BASE_DIR / "docs" / "grafo.mmd"
RUTA_DIAGRAMA_PNG = BASE_DIR / "docs" / "grafo.png"


def crear_llm(clave: str, modelo: str, max_tokens: int) -> ChatAnthropic:
    """Modelo de Anthropic.

    No se envia `temperature`: la API de Anthropic la rechaza con este modelo
    (error 400 "temperature: Extra inputs are not permitted"), asi que queda el valor del servicio.
    """
    return ChatAnthropic(  # type: ignore[call-arg]
        model=modelo,
        api_key=clave,  # type: ignore[arg-type]
        max_tokens=max_tokens,
    )


def crear_grafo(config: Configuracion) -> CompiledStateGraph:
    """Grafo con los modelos reales: uno para el supervisor y otro para los especialistas."""
    supervisor = crear_llm(config.anthropic_api_key, config.modelo_supervisor, max_tokens=1024)
    especialistas = crear_llm(config.anthropic_api_key, config.modelo_especialistas, max_tokens=2048)
    return construir_grafo_desde_config(supervisor, especialistas, config)


async def ejecutar_tarea(
    grafo: CompiledStateGraph, tarea: str, config: Configuracion, escenario: str = "consulta"
) -> Corrida:
    """Corre una tarea completa por el grafo y devuelve la corrida con su traza y sus metricas."""
    uso = ContadorDeUso()
    inicio = time.perf_counter()
    estado = await grafo.ainvoke(
        estado_inicial(tarea),
        {"recursion_limit": config.recursion_limit_grafo, "callbacks": [uso]},
    )
    return corrida_desde_estado(escenario, estado, time.perf_counter() - inicio, uso)


def diagrama_mermaid(grafo: CompiledStateGraph) -> str:
    """Texto Mermaid del grafo, generado por LangGraph a partir de los nodos y aristas reales."""
    return grafo.get_graph().draw_mermaid()


def guardar_diagrama(grafo: CompiledStateGraph, ruta_mmd: Path = RUTA_DIAGRAMA) -> tuple[Path, Path | None]:
    """Guarda el diagrama como texto Mermaid y, si hay red, tambien como PNG.

    El PNG lo genera `draw_mermaid_png()` (que usa el servicio mermaid.ink); si no se puede
    alcanzar, se guarda solo el texto y se devuelve `None` en lugar de la ruta del PNG.
    """
    ruta_mmd.parent.mkdir(parents=True, exist_ok=True)
    ruta_mmd.write_text(diagrama_mermaid(grafo), encoding="utf-8")
    ruta_png = ruta_mmd.with_suffix(".png")
    try:
        ruta_png.write_bytes(grafo.get_graph().draw_mermaid_png())
    except Exception:  # noqa: BLE001  - el PNG es opcional: cualquier falla de red o de renderizado se tolera
        return ruta_mmd, None
    return ruta_mmd, ruta_png
