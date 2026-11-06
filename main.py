"""Linea de comandos del orquestador multi-agente de Gobierno de Datos.

Ejemplos:
    python main.py "Priorizá los incidentes críticos abiertos del maestro de artículos"
    python main.py -v "Priorizá los incidentes abiertos del maestro de proveedores"
    python main.py --demo          # corrida completa de 3 escenarios; guarda la traza en trazas/
    python main.py --diagrama      # genera docs/grafo.mmd (y docs/grafo.png si hay red)
"""

import argparse
import asyncio
import sys
from collections.abc import Sequence
from dataclasses import replace

import anthropic
from langgraph.errors import GraphRecursionError

from config import BASE_DIR, Configuracion
from ejecucion import crear_grafo, ejecutar_tarea, guardar_diagrama
from errores import ErrorConfiguracion
from trazas import Corrida, a_log, guardar_traza

CARPETA_TRAZAS = BASE_DIR / "trazas"

# Escenarios de la corrida --demo. Son SOLO tareas: a quien delegar, en que orden y cuando
# cerrar lo decide el supervisor.
ESCENARIOS: list[tuple[str, str]] = [
    (
        "1. Priorización multi-agente (maestro de artículos)",
        "Necesito saber qué incidentes críticos abiertos tiene el maestro de artículos, qué regla incumple "
        "cada uno, cuánto se pasaron del plazo (SLA) y cuál conviene atacar primero. Armame un informe corto.",
    ),
    (
        "2. Otro maestro con más incidentes (proveedores)",
        "Priorizá todos los incidentes abiertos del maestro de proveedores: qué reglas incumplen, cómo está "
        "el vencimiento del SLA de cada uno y en qué orden hay que atenderlos. Quiero un informe corto.",
    ),
    (
        "3. Consulta fuera de alcance",
        "¿Cuántos empleados tiene Mercado Austral y cuánto cobra el gerente de datos?",
    ),
]


def _mostrar(corrida: Corrida, detalle: bool) -> None:
    if detalle:
        print(a_log([corrida]))
        return
    print(corrida.respuesta_final)
    guardas = (
        f", {corrida.decisiones_corregidas_por_guarda} decisión(es) corregida(s) por las guardas"
        if corrida.decisiones_corregidas_por_guarda
        else ""
    )
    print(
        f"\n[estado: {corrida.estado_final} | agentes: {', '.join(corrida.agentes_que_intervinieron) or 'ninguno'} | "
        f"delegaciones: {corrida.pasos_del_supervisor}{guardas} | "
        f"tokens: {corrida.tokens_entrada} entrada, {corrida.tokens_salida} salida | {corrida.segundos} s]"
    )


async def _una_tarea(config: Configuracion, tarea: str, detalle: bool) -> None:
    corrida = await ejecutar_tarea(crear_grafo(config), tarea, config)
    _mostrar(corrida, detalle)


async def _demo(config: Configuracion) -> None:
    """Corrida completa con el modelo real. Guarda la traza como JSON y como log."""
    grafo = crear_grafo(config)
    corridas: list[Corrida] = []
    for escenario, tarea in ESCENARIOS:
        print(f"--> {escenario}")
        corrida = await ejecutar_tarea(grafo, tarea, config, escenario)
        corridas.append(corrida)
        _mostrar(corrida, detalle=False)
        print()

    ruta_json, ruta_log = guardar_traza(
        CARPETA_TRAZAS,
        "traza-demo",
        corridas,
        {
            "proyecto": "pre-entrega-6-orquestador",
            "modelo_supervisor": config.modelo_supervisor,
            "modelo_especialistas": config.modelo_especialistas,
            "max_pasos": config.max_pasos,
            "max_intentos_por_agente": config.max_intentos_por_agente,
            "recursion_especialista": config.recursion_especialista,
        },
    )
    print(f"Traza guardada en:\n  {ruta_json}\n  {ruta_log}")


def _diagrama(config: Configuracion) -> None:
    ruta_mmd, ruta_png = guardar_diagrama(crear_grafo(config))
    print(f"Diagrama Mermaid guardado en {ruta_mmd}")
    if ruta_png is None:
        print("No pude generar el PNG (draw_mermaid_png necesita acceso a mermaid.ink); el texto Mermaid sirve igual.")
    else:
        print(f"Diagrama PNG guardado en {ruta_png}")


def _parsear(argumentos: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Orquestador multi-agente de Gobierno de Datos (LangGraph + Claude).")
    parser.add_argument("tarea", nargs="?", help="Tarea para el equipo de agentes.")
    parser.add_argument("--demo", action="store_true", help="Corrida completa de 3 escenarios; guarda la traza.")
    parser.add_argument("--diagrama", action="store_true", help="Genera el diagrama Mermaid del grafo en docs/.")
    parser.add_argument("-v", "--verbose", action="store_true", help="Muestra cada paso de la delegación.")
    return parser.parse_args(argumentos)


def _ejecutar(args: argparse.Namespace) -> int:
    if args.diagrama:
        # El diagrama no llama al modelo: se puede generar sin clave real.
        config = Configuracion.desde_entorno(requerir_clave=False)
        _diagrama(replace(config, anthropic_api_key=config.anthropic_api_key or "sin-clave"))
        return 0

    config = Configuracion.desde_entorno()
    if args.demo:
        asyncio.run(_demo(config))
    elif args.tarea:
        asyncio.run(_una_tarea(config, args.tarea, args.verbose))
    else:
        print("Indicá una tarea, --demo o --diagrama. Mirá python main.py --help.")
        return 2
    return 0


def main(argumentos: Sequence[str] | None = None) -> int:
    args = _parsear(argumentos)
    try:
        return _ejecutar(args)
    except ErrorConfiguracion as error:
        print(f"Error de configuración: {error}", file=sys.stderr)
    except anthropic.AuthenticationError:
        print("Anthropic rechazó la clave. Revisá ANTHROPIC_API_KEY en el .env.", file=sys.stderr)
    except anthropic.RateLimitError:
        print("Anthropic devolvió límite de uso (429). Esperá un rato y probá de nuevo.", file=sys.stderr)
    except anthropic.APIConnectionError:
        print("No pude conectarme con Anthropic. Revisá tu conexión a internet.", file=sys.stderr)
    except anthropic.APIStatusError as error:
        print(f"Anthropic devolvió un error ({error.status_code}): {error.message}", file=sys.stderr)
    except GraphRecursionError:
        print("El grafo se pasó de su límite de pasos sin cerrar. Subí MAX_PASOS o revisá la tarea.", file=sys.stderr)
    except KeyboardInterrupt:
        print("\nInterrumpido.", file=sys.stderr)
    return 1


if __name__ == "__main__":
    sys.exit(main())
