"""Fixtures compartidas. Ningun test necesita claves ni red."""

from typing import Any

import pytest

from config import Configuracion
from graph import construir_grafo
from state import EstadoOrquestador, estado_inicial
from tests.fakes import ModeloPorPolitica, SupervisorFalso

TAREA = "Priorizá los incidentes críticos abiertos del maestro de artículos."


@pytest.fixture
def config() -> Configuracion:
    """Configuracion de prueba (sin clave real)."""
    return Configuracion(
        anthropic_api_key="clave-falsa-de-prueba",
        modelo_supervisor="modelo-falso",
        modelo_especialistas="modelo-falso",
        max_pasos=10,
        max_intentos_por_agente=3,
        recursion_especialista=12,
    )


def crear_estado(**cambios: Any) -> EstadoOrquestador:
    """Estado inicial con los campos que se quieran pisar (para tests unitarios)."""
    estado = estado_inicial(TAREA)
    estado.update(cambios)
    return estado  # type: ignore[return-value]


async def correr(
    supervisor: SupervisorFalso,
    especialistas: ModeloPorPolitica,
    *,
    max_pasos: int = 10,
    max_intentos: int = 3,
    tarea: str = TAREA,
) -> dict[str, Any]:
    """Arma el grafo con modelos falsos, lo corre de punta a punta y devuelve el estado final."""
    grafo = construir_grafo(
        supervisor, especialistas, max_pasos=max_pasos, max_intentos=max_intentos, recursion_especialista=12
    )
    resultado: dict[str, Any] = await grafo.ainvoke(
        estado_inicial(tarea), {"recursion_limit": 3 * max_pasos + 5}
    )
    return resultado
