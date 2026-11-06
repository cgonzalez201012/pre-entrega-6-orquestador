"""Modelos de chat falsos para probar el GRAFO sin claves ni red.

Ojo con lo que prueban: verifican que el grafo enruta, delega, valida, refina y corta donde
corresponde. NO prueban que Claude decida bien: eso se ve con una corrida real
(python main.py --demo), cuya traza queda en trazas/.

Las herramientas SI se ejecutan de verdad (leen los JSON de data/): lo unico simulado es la
decision del modelo.
"""

import json
from collections.abc import Callable, Sequence
from datetime import date
from itertools import count
from pathlib import Path
from typing import Any

from langchain_core.callbacks import AsyncCallbackManagerForLLMRun, CallbackManagerForLLMRun
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.runnables import Runnable, RunnableLambda
from pydantic import Field

import calculos
from config import BASE_DIR
from state import Analisis, Hallazgos, IncidenteHallazgo, ItemPrioridad, ReglaHallazgo
from supervisor import DecisionSupervisor

_ids = count(1)


# ---------- Mensajes del modelo ----------


def pedir_varias(*llamadas: tuple[str, dict[str, Any]]) -> AIMessage:
    """Mensaje del modelo que pide VARIAS herramientas a la vez (llamadas paralelas)."""
    return AIMessage(
        content="",
        tool_calls=[
            {"name": nombre, "args": argumentos, "id": f"call_{next(_ids)}", "type": "tool_call"}
            for nombre, argumentos in llamadas
        ],
    )


def pedir(herramienta: str, **argumentos: Any) -> AIMessage:
    """Mensaje del modelo que pide UNA herramienta."""
    return pedir_varias((herramienta, argumentos))


def responder(texto: str) -> AIMessage:
    """Mensaje final del modelo (sin herramientas)."""
    return AIMessage(content=texto)


# ---------- Modelo falso generico (para los especialistas) ----------


class ModeloPorPolitica(BaseChatModel):
    """Modelo cuyo siguiente mensaje lo decide una funcion de lo que recibe."""

    politica: Callable[[list[BaseMessage]], AIMessage]
    recibidos: list[list[BaseMessage]] = Field(default_factory=list)

    @property
    def _llm_type(self) -> str:
        return "falso-por-politica"

    def bind_tools(self, tools: Sequence[Any], **kwargs: Any) -> "ModeloPorPolitica":
        return self  # las herramientas no hacen falta: el modelo falso ya sabe que pedir

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        self.recibidos.append(list(messages))
        return ChatResult(generations=[ChatGeneration(message=self.politica(list(messages)))])

    async def _agenerate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: AsyncCallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        return self._generate(messages, stop)


class ErrorDeApi(Exception):
    """Simula un error de la API del modelo (con `status_code`, como los del SDK de Anthropic)."""

    def __init__(self, mensaje: str, status_code: int) -> None:
        super().__init__(mensaje)
        self.status_code = status_code


def especialistas_falsos(
    investigador: Sequence[AIMessage | Exception] = (),
    analista: Sequence[AIMessage | Exception] = (),
    redactor: Sequence[AIMessage | Exception] = (),
) -> ModeloPorPolitica:
    """Un modelo que responde segun QUIEN lo llama (lo sabe por el prompt de sistema).

    Cada rol tiene su propia cola de mensajes: se consumen en orden, a lo largo de toda la
    corrida (incluidos los refinamientos). Si un elemento de la cola es una excepcion, el
    modelo la levanta en lugar de responder (simula una falla de la API).
    """
    colas: dict[str, list[AIMessage | Exception]] = {
        "Investigador": list(investigador),
        "Analista": list(analista),
        "Redactor": list(redactor),
    }

    def politica(mensajes: list[BaseMessage]) -> AIMessage:
        sistema = str(mensajes[0].content)
        for rol, cola in colas.items():
            if f"Agente {rol}" in sistema:
                if not cola:
                    raise AssertionError(f"El modelo falso del {rol} se quedo sin mensajes del guion.")
                siguiente = cola.pop(0)
                if isinstance(siguiente, Exception):
                    raise siguiente
                return siguiente
        raise AssertionError("No se pudo identificar el rol que llamo al modelo falso.")

    return ModeloPorPolitica(politica=politica)


# ---------- Supervisor falso ----------


class SupervisorFalso(BaseChatModel):
    """Modelo del supervisor: `with_structured_output` devuelve decisiones de un guion.

    `guion` puede tener `DecisionSupervisor` o excepciones (para simular una salida invalida
    del modelo). Si se acaba, repite la ultima decision (sirve para probar los topes).
    """

    guion: list[Any]
    recibidos: list[list[BaseMessage]] = Field(default_factory=list)

    @property
    def _llm_type(self) -> str:
        return "supervisor-falso"

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        raise NotImplementedError("El supervisor solo se usa con with_structured_output.")

    def with_structured_output(self, schema: Any, **kwargs: Any) -> Runnable[Any, Any]:
        async def decidir(mensajes: list[BaseMessage]) -> Any:
            self.recibidos.append(list(mensajes))
            decision = self.guion.pop(0) if len(self.guion) > 1 else self.guion[0]
            if isinstance(decision, Exception):
                raise decision
            return decision

        return RunnableLambda(decidir)


def decidir(siguiente: str, instruccion: str = "", motivo: str = "porque si") -> DecisionSupervisor:
    return DecisionSupervisor(siguiente=siguiente, instruccion=instruccion, motivo=motivo)  # type: ignore[arg-type]


# ---------- Guiones de especialistas, armados con los datos reales de data/ ----------


def _json(nombre: str) -> dict[str, Any]:
    contenido: dict[str, Any] = json.loads((Path(BASE_DIR) / "data" / nombre).read_text(encoding="utf-8"))
    return contenido


def _incidente(incidente_id: str) -> dict[str, Any]:
    return next(i for i in _json("incidentes_calidad.json")["incidentes"] if i["id"] == incidente_id)


def _regla(regla_id: str) -> dict[str, Any]:
    return next(r for r in _json("reglas_calidad.json")["reglas"] if r["id"] == regla_id)


def guion_investigador(
    incidentes: Sequence[str] = ("INC-1001", "INC-1002"),
    maestro: str = "articulos",
    severidad: str | None = "critica",
    resumen: str = "Consulté los incidentes críticos abiertos y la ficha y la regla de cada uno.",
) -> list[AIMessage]:
    """El investigador lista, pide el detalle de cada incidente y la regla de cada uno."""
    reglas = sorted({_incidente(i)["regla_id"] for i in incidentes})
    return [
        pedir("buscar_incidentes_calidad", maestro=maestro, estado="abierto", severidad=severidad),
        pedir_varias(*[("detalle_incidente", {"incidente_id": i}) for i in incidentes]),
        pedir_varias(*[("consultar_regla_calidad", {"regla_id": r}) for r in reglas]),
        responder(resumen),
    ]


def guion_analista(
    incidentes: Sequence[str] = ("INC-1001", "INC-1002"),
    alterar: dict[str, dict[str, Any]] | None = None,
    resumen: str = "Calculé el vencimiento de cada incidente, la brecha de cada regla y el ranking.",
) -> list[AIMessage]:
    """El analista calcula todo con las herramientas.

    `alterar` pisa argumentos de las llamadas para simular que el modelo copio mal un valor
    (por ejemplo {"INC-1001": {"fecha_alta": "2026-10-01"}}).
    """
    alterar = alterar or {}
    datos = _json("incidentes_calidad.json")
    corte = date.fromisoformat(datos["fecha_corte"])
    vencimientos = []
    brechas = []
    items = []
    for incidente_id in incidentes:
        inc = _incidente(incidente_id)
        sla = datos["sla_horas_habiles"][inc["severidad"]]
        argumentos = {
            "incidente_id": incidente_id,
            "severidad": inc["severidad"],
            "fecha_alta": inc["fecha_alta"],
            "fecha_corte": datos["fecha_corte"],
            "sla_horas_habiles": sla,
            **alterar.get(incidente_id, {}),
        }
        vencimientos.append(("calcular_vencimiento_sla", argumentos))
        regla = _regla(inc["regla_id"])
        brechas.append(
            (
                "calcular_brecha_regla",
                {
                    "regla_id": regla["id"],
                    "medicion_pct": regla["medicion_actual_pct"],
                    "umbral_pct": regla["umbral_aceptacion_pct"],
                },
            )
        )
        real = calculos.calcular_vencimiento(
            incidente_id, inc["severidad"], date.fromisoformat(inc["fecha_alta"]), corte, sla
        )
        items.append(
            ItemPrioridad(
                incidente_id=incidente_id,
                severidad=inc["severidad"],
                vencido=real.vencido,
                exceso_horas=real.exceso_horas,
                registros_afectados=inc["registros_afectados"],
            ).model_dump()
        )
    unicas = {b[1]["regla_id"]: b for b in brechas}
    return [
        pedir_varias(*vencimientos, *unicas.values()),
        pedir("priorizar_incidentes", items=items),
        responder(resumen),
    ]


def informe(*ids: str) -> AIMessage:
    """Un informe que cita los IDs dados."""
    return responder("Informe de calidad de datos. Incidentes por prioridad: " + ", ".join(ids) + ".")


# ---------- Datos listos para armar estados en los tests unitarios ----------


def hallazgos_de_datos(incidentes: Sequence[str] = ("INC-1001", "INC-1002"), con_reglas: bool = True) -> Hallazgos:
    """`Hallazgos` armado directamente de data/ (equivale a lo que juntaria un investigador perfecto)."""
    datos = _json("incidentes_calidad.json")
    fichas = []
    reglas = []
    for incidente_id in incidentes:
        inc = _incidente(incidente_id)
        fichas.append(
            IncidenteHallazgo(**inc, sla_horas_habiles=datos["sla_horas_habiles"][inc["severidad"]])
        )
        if con_reglas:
            regla = _regla(inc["regla_id"])
            reglas.append(ReglaHallazgo(**regla, cumple_umbral=regla["medicion_actual_pct"] >= regla["umbral_aceptacion_pct"]))
    unicas = {r.id: r for r in reglas}
    return Hallazgos(
        fecha_corte=date.fromisoformat(datos["fecha_corte"]),
        incidentes=fichas,
        reglas=list(unicas.values()),
        herramientas_usadas=["buscar_incidentes_calidad", "detalle_incidente", "consultar_regla_calidad"],
        resumen="resumen del investigador",
    )


def analisis_de_datos(hallazgos: Hallazgos) -> Analisis:
    """`Analisis` correcto para esos hallazgos (lo que calcularia un analista perfecto)."""
    vencimientos = [
        calculos.calcular_vencimiento(i.id, i.severidad, i.fecha_alta, hallazgos.fecha_corte, i.sla_horas_habiles)  # type: ignore[arg-type]
        for i in hallazgos.incidentes_detallados()
    ]
    brechas = [
        calculos.calcular_brecha(r.id, r.medicion_actual_pct, r.umbral_aceptacion_pct)
        for r in hallazgos.reglas
    ]
    por_id = {v.incidente_id: v for v in vencimientos}
    items = [
        ItemPrioridad(
            incidente_id=i.id,
            severidad=i.severidad,
            vencido=por_id[i.id].vencido,
            exceso_horas=por_id[i.id].exceso_horas,
            registros_afectados=i.registros_afectados or 0,
        )
        for i in hallazgos.incidentes_detallados()
    ]
    return Analisis(
        brechas=brechas,
        vencimientos=vencimientos,
        ranking=calculos.ordenar_por_prioridad(items),
        herramientas_usadas=["calcular_vencimiento_sla", "calcular_brecha_regla", "priorizar_incidentes"],
        resumen="resumen del analista",
    )
