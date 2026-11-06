"""Traza de una corrida: convierte el estado final del grafo en pasos legibles de delegacion.

Una corrida deja en el estado final tres listas paralelas, una entrada por vuelta:

    decisiones      lo que decidio el supervisor (a quien mando, con que instruccion, por que)
    contribuciones  lo que aporto cada especialista (que herramientas uso, que resumio)
    validaciones    el veredicto de la rubrica sobre ese aporte

`corrida_desde_estado` las junta por numero de paso y `a_log` las cuenta como una historia.
"""

import json
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from langchain_core.callbacks import AsyncCallbackHandler
from langchain_core.outputs import LLMResult


class ContadorDeUso(AsyncCallbackHandler):
    """Cuenta las llamadas al modelo y los tokens de entrada y salida de TODA la corrida.

    Se pasa en `config["callbacks"]` al invocar el grafo y llega a todos los nodos, incluidos
    los especialistas (cada reintento o ronda de herramientas se paga y se cuenta).
    """

    def __init__(self) -> None:
        self.llamadas = 0
        self.tokens_entrada = 0
        self.tokens_salida = 0

    async def on_llm_end(self, response: LLMResult, **kwargs: Any) -> None:
        self.llamadas += 1
        for generaciones in response.generations:
            for generacion in generaciones:
                uso = getattr(getattr(generacion, "message", None), "usage_metadata", None)
                if uso:
                    self.tokens_entrada += int(uso.get("input_tokens", 0))
                    self.tokens_salida += int(uso.get("output_tokens", 0))


@dataclass
class Corrida:
    """Una tarea del usuario, con todo lo que hizo el equipo para resolverla."""

    escenario: str
    tarea: str
    estado_final: str
    respuesta_final: str
    pasos_del_supervisor: int
    decisiones: list[dict[str, Any]] = field(default_factory=list)
    contribuciones: list[dict[str, Any]] = field(default_factory=list)
    validaciones: list[dict[str, Any]] = field(default_factory=list)
    decisiones_corregidas_por_guarda: int = 0
    llamadas_al_modelo: int = 0
    tokens_entrada: int = 0
    tokens_salida: int = 0
    segundos: float = 0.0

    @property
    def agentes_que_intervinieron(self) -> list[str]:
        """Los agentes que trabajaron, en orden y sin repetir."""
        orden: list[str] = []
        for contribucion in self.contribuciones:
            if contribucion["agente"] not in orden:
                orden.append(contribucion["agente"])
        return orden

    def a_dict(self) -> dict[str, Any]:
        datos = asdict(self)
        datos["agentes_que_intervinieron"] = self.agentes_que_intervinieron
        return datos


def corrida_desde_estado(
    escenario: str,
    estado: Mapping[str, Any],
    segundos: float = 0.0,
    uso: ContadorDeUso | None = None,
) -> Corrida:
    """Arma la `Corrida` a partir del estado final que devuelve `grafo.ainvoke`."""
    decisiones = [d.model_dump(mode="json") for d in estado["decisiones"]]
    return Corrida(
        escenario=escenario,
        tarea=estado["tarea"],
        estado_final=estado["estado_final"] or "sin_cerrar",
        respuesta_final=estado["respuesta_final"] or "",
        pasos_del_supervisor=estado["pasos"],
        decisiones=decisiones,
        contribuciones=[c.model_dump(mode="json") for c in estado["contribuciones"]],
        validaciones=[v.model_dump(mode="json") for v in estado["validaciones"]],
        decisiones_corregidas_por_guarda=sum(1 for d in decisiones if d["corregida_por_guarda"]),
        llamadas_al_modelo=uso.llamadas if uso else 0,
        tokens_entrada=uso.tokens_entrada if uso else 0,
        tokens_salida=uso.tokens_salida if uso else 0,
        segundos=round(segundos, 2),
    )


def _herramientas(nombres: Sequence[str]) -> str:
    """['a', 'b', 'b'] -> 'a, b x2' (conserva el orden de primera aparicion)."""
    conteo = Counter(nombres)
    return ", ".join(n if conteo[n] == 1 else f"{n} x{conteo[n]}" for n in dict.fromkeys(nombres))


def a_log(corridas: Sequence[Corrida]) -> str:
    """Version en texto de las corridas: la historia de la delegacion paso a paso."""
    lineas: list[str] = []
    for corrida in corridas:
        lineas.append(f"=== {corrida.escenario} ===")
        lineas.append(f'Tarea: "{corrida.tarea}"')
        lineas.append("")
        for decision in corrida.decisiones:
            paso = decision["paso"]
            if decision["siguiente"] == "FINISH":
                lineas.append("[Cierre] Supervisor → FINISH (cierra la tarea)")
                lineas.append(f"           Motivo: {decision['motivo']}")
                if decision["razon_guarda"]:
                    lineas.append(f"           !! Guarda: {decision['razon_guarda']}")
                continue
            lineas.append(f"[Paso {paso}] Supervisor → {decision['siguiente']}")
            lineas.append(f"           Motivo: {decision['motivo']}")
            lineas.append(f'           Instrucción: "{decision["instruccion"]}"')
            if decision["razon_guarda"]:
                lineas.append(f"           !! Guarda corrigió la decisión: {decision['razon_guarda']}")
            for contribucion in (c for c in corrida.contribuciones if c["paso"] == paso):
                usadas = _herramientas(contribucion["herramientas_usadas"]) or "ninguna"
                lineas.append(f"           {contribucion['agente'].capitalize()} usó: {usadas}")
                lineas.append(f"           Resumen: {_una_linea(contribucion['resumen'])}")
                if contribucion["error"]:
                    lineas.append(f"           !! Error del especialista: {contribucion['error']}")
            for validacion in (v for v in corrida.validaciones if v["paso"] == paso):
                if validacion["aprobada"]:
                    lineas.append("           Validación: APROBADA")
                else:
                    lineas.append("           Validación: RECHAZADA")
                    lineas.extend(f"             - {_una_linea(p)}" for p in validacion["problemas"])
        lineas.append("")
        lineas.append(
            f"Estado final: {corrida.estado_final} | delegaciones: {corrida.pasos_del_supervisor} | "
            f"agentes: {', '.join(corrida.agentes_que_intervinieron) or 'ninguno'}"
        )
        lineas.append(
            f"Llamadas al modelo: {corrida.llamadas_al_modelo} | tokens: {corrida.tokens_entrada} entrada, "
            f"{corrida.tokens_salida} salida | {corrida.segundos} s"
        )
        lineas.append("Respuesta final:")
        lineas.append(corrida.respuesta_final)
        lineas.append("")
    return "\n".join(lineas)


def _una_linea(texto: str) -> str:
    return " ".join(texto.split())


def guardar_traza(
    carpeta: Path, nombre: str, corridas: Sequence[Corrida], metadatos: dict[str, Any]
) -> tuple[Path, Path]:
    """Escribe `<nombre>.json` y `<nombre>.log` en la carpeta y devuelve sus rutas."""
    carpeta.mkdir(parents=True, exist_ok=True)
    ruta_json = carpeta / f"{nombre}.json"
    ruta_log = carpeta / f"{nombre}.log"
    documento = {
        **metadatos,
        "generado_en": datetime.now(UTC).isoformat(timespec="seconds"),
        "corridas": [corrida.a_dict() for corrida in corridas],
    }
    ruta_json.write_text(json.dumps(documento, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    ruta_log.write_text(a_log(corridas), encoding="utf-8")
    return ruta_json, ruta_log
