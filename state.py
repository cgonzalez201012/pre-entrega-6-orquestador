"""Estado compartido del orquestador y contratos de datos entre agentes.

Este archivo es el "contrato" del sistema multi-agente:

1. `EstadoOrquestador` es el estado del grafo (hereda `messages` de MessagesState).
2. Los modelos Pydantic (`Hallazgos`, `Analisis`, ...) definen EXACTAMENTE que dato le
   pasa un agente al siguiente, para que nadie dependa de un diccionario sin forma.

Regla de escritura: cada agente es dueno de UNA parcela del estado y solo escribe ahi.

    investigador -> hallazgos            analista -> analisis
    redactor     -> informe              validacion -> validacion, validaciones
    supervisor   -> siguiente, instruccion, pasos, intentos, decisiones,
                    estado_final, respuesta_final

Todos escriben ademas `messages` y `contribuciones` (listas que solo se anexan, nunca
se pisan), que es lo que permite rastrear quien aporto que.
"""

import operator
from datetime import date
from typing import Annotated, Literal

from langchain_core.messages import HumanMessage
from langgraph.graph import MessagesState
from pydantic import BaseModel, ConfigDict, Field

Severidad = Literal["critica", "alta", "media", "baja"]
NombreAgente = Literal["investigador", "analista", "redactor"]
Destino = Literal["investigador", "analista", "redactor", "FINISH"]
EstadoFinal = Literal["completado", "limite_de_pasos", "validacion_fallida"]

AGENTES: tuple[NombreAgente, ...] = ("investigador", "analista", "redactor")
FINISH: Literal["FINISH"] = "FINISH"

# Marca literal que usa el investigador cuando la consulta no se puede responder con sus
# herramientas. Es una salida explicita (no un texto libre) para poder detectarla en codigo.
FUERA_DE_ALCANCE = "FUERA_DE_ALCANCE"


class _Modelo(BaseModel):
    """Base de los contratos: inmutables, para que un agente no pueda mutar lo de otro."""

    model_config = ConfigDict(frozen=True)


# ---------- Parcela del investigador ----------


class IncidenteHallazgo(_Modelo):
    """Un incidente tal como lo devolvieron las herramientas del investigador."""

    id: str
    maestro: str
    titulo: str
    severidad: Severidad
    estado: str
    fecha_alta: date
    # Los siguientes campos solo vienen en el detalle del incidente.
    regla_id: str | None = None
    registros_afectados: int | None = None
    responsable: str | None = None
    data_steward: str | None = None
    sla_horas_habiles: int | None = None

    @property
    def detallado(self) -> bool:
        """True si se pidio su ficha completa (regla, registros afectados y SLA)."""
        return (
            self.regla_id is not None
            and self.registros_afectados is not None
            and self.sla_horas_habiles is not None
        )


class ReglaHallazgo(_Modelo):
    """Una regla de calidad tal como la devolvio la herramienta."""

    id: str
    maestro: str
    nombre: str
    descripcion: str
    umbral_aceptacion_pct: float
    medicion_actual_pct: float
    cumple_umbral: bool


class Hallazgos(_Modelo):
    """Lo que el investigador le entrega al resto del equipo.

    Los datos salen de los resultados de las herramientas, no de lo que el modelo
    escribio: el texto libre del investigador solo viaja en `resumen`.
    """

    fecha_corte: date | None = None
    incidentes: list[IncidenteHallazgo] = Field(default_factory=list)
    reglas: list[ReglaHallazgo] = Field(default_factory=list)
    consultas_con_error: list[str] = Field(default_factory=list)
    herramientas_usadas: list[str] = Field(default_factory=list)
    fuera_de_alcance: bool = False
    resumen: str = ""

    def incidentes_detallados(self) -> list[IncidenteHallazgo]:
        return [incidente for incidente in self.incidentes if incidente.detallado]

    def regla(self, regla_id: str) -> ReglaHallazgo | None:
        return next((regla for regla in self.reglas if regla.id == regla_id), None)

    def ids_conocidos(self) -> set[str]:
        """Todos los IDs (incidentes y reglas) que existen en los datos recolectados."""
        return {i.id for i in self.incidentes} | {r.id for r in self.reglas}

    def fusionar(self, nuevos: "Hallazgos") -> "Hallazgos":
        """Suma lo que trajo una nueva corrida del investigador a lo que ya se habia recolectado.

        Un refinamiento COMPLETA el trabajo anterior, no lo reemplaza: si el supervisor pide
        "consultá las reglas que faltan", los incidentes ya recolectados tienen que seguir ahi.
        - incidentes y reglas se juntan por ID; en un mismo ID, lo nuevo pisa campo a campo
          (pero un listado sin ficha no borra el detalle que ya se tenia);
        - las herramientas usadas se anexan;
        - los errores y el resumen son los de la ULTIMA corrida;
        - si la ultima corrida declaro la consulta fuera de alcance, eso es lo que vale.
        """
        if nuevos.fuera_de_alcance:
            return nuevos
        incidentes = {i.id: i.model_dump() for i in self.incidentes}
        for incidente in nuevos.incidentes:
            vigentes = {campo: valor for campo, valor in incidente.model_dump().items() if valor is not None}
            incidentes[incidente.id] = {**incidentes.get(incidente.id, {}), **vigentes}
        reglas = {r.id: r for r in self.reglas}
        reglas.update({r.id: r for r in nuevos.reglas})
        return Hallazgos(
            fecha_corte=nuevos.fecha_corte or self.fecha_corte,
            incidentes=[IncidenteHallazgo(**datos) for datos in incidentes.values()],
            reglas=list(reglas.values()),
            consultas_con_error=nuevos.consultas_con_error,
            herramientas_usadas=[*self.herramientas_usadas, *nuevos.herramientas_usadas],
            fuera_de_alcance=False,
            resumen=nuevos.resumen,
        )


# ---------- Parcela del analista ----------


class BrechaRegla(_Modelo):
    """Distancia entre la medicion de una regla y su umbral, en puntos porcentuales."""

    regla_id: str
    medicion_pct: float
    umbral_pct: float
    brecha_pp: float  # medicion - umbral: negativa = por debajo del umbral
    cumple: bool


class VencimientoSLA(_Modelo):
    """Cuanto tiempo habil lleva abierto un incidente frente a su plazo (SLA)."""

    incidente_id: str
    severidad: Severidad
    fecha_alta: date
    fecha_corte: date
    dias_habiles: int
    horas_habiles_transcurridas: int
    sla_horas_habiles: int
    exceso_horas: int  # horas habiles pasadas del plazo (0 si no vencio)
    vencido: bool


class ItemPrioridad(_Modelo):
    """Datos de un incidente que alcanzan para ordenarlo en el ranking de prioridad."""

    incidente_id: str
    severidad: Severidad
    vencido: bool
    exceso_horas: int = Field(ge=0)
    registros_afectados: int = Field(ge=0)


class Analisis(_Modelo):
    """Lo que el analista le entrega al redactor: numeros calculados por herramientas."""

    brechas: list[BrechaRegla] = Field(default_factory=list)
    vencimientos: list[VencimientoSLA] = Field(default_factory=list)
    ranking: list[str] = Field(default_factory=list)  # IDs de incidente, el primero es el mas urgente
    errores: list[str] = Field(default_factory=list)
    herramientas_usadas: list[str] = Field(default_factory=list)
    resumen: str = ""

    def fusionar(self, nuevo: "Analisis") -> "Analisis":
        """Suma una nueva corrida del analista a lo ya calculado (mismo criterio que `Hallazgos.fusionar`).

        Brechas y vencimientos se juntan por ID y lo nuevo pisa a lo viejo; el ranking nuevo
        reemplaza al anterior solo si trae uno; errores y resumen son los de la ultima corrida.
        """
        brechas = {b.regla_id: b for b in self.brechas}
        brechas.update({b.regla_id: b for b in nuevo.brechas})
        vencimientos = {v.incidente_id: v for v in self.vencimientos}
        vencimientos.update({v.incidente_id: v for v in nuevo.vencimientos})
        return Analisis(
            brechas=list(brechas.values()),
            vencimientos=list(vencimientos.values()),
            ranking=nuevo.ranking or self.ranking,
            errores=nuevo.errores,
            herramientas_usadas=[*self.herramientas_usadas, *nuevo.herramientas_usadas],
            resumen=nuevo.resumen,
        )


# ---------- Registro de lo que pasa en el grafo ----------


class Contribucion(_Modelo):
    """Que aporto un agente en una vuelta: sirve para rastrear quien hizo que."""

    agente: NombreAgente
    paso: int
    resumen: str
    herramientas_usadas: list[str] = Field(default_factory=list)
    error: str | None = None


class DecisionRegistrada(_Modelo):
    """Una decision del supervisor, incluida la correccion de las guardas si hubo."""

    paso: int
    siguiente: Destino
    instruccion: str
    motivo: str
    corregida_por_guarda: bool = False
    razon_guarda: str | None = None


class ResultadoValidacion(_Modelo):
    """Veredicto del nodo de validacion sobre lo ultimo que entrego un agente."""

    agente: NombreAgente
    aprobada: bool
    problemas: list[str] = Field(default_factory=list)
    paso: int


def sumar_contadores(actual: dict[str, int], nuevo: dict[str, int]) -> dict[str, int]:
    """Reducer de `intentos`: suma los contadores en un diccionario nuevo (no muta ninguno)."""
    fusionado = dict(actual)
    for clave, cantidad in nuevo.items():
        fusionado[clave] = fusionado.get(clave, 0) + cantidad
    return fusionado


class EstadoOrquestador(MessagesState):
    """Estado compartido del grafo.

    `messages` (heredado) lleva la conversacion visible: la tarea del usuario y un mensaje
    por cada aporte de un agente. Los campos de abajo llevan los datos estructurados.
    Los que tienen un reducer (`Annotated`) se ANEXAN o se SUMAN; el resto se pisa.
    """

    tarea: str
    # Parcela del supervisor
    siguiente: Destino
    instruccion: str
    pasos: int
    intentos: Annotated[dict[str, int], sumar_contadores]
    decisiones: Annotated[list[DecisionRegistrada], operator.add]
    estado_final: EstadoFinal | None
    respuesta_final: str | None
    # Parcelas de los especialistas
    hallazgos: Hallazgos | None
    analisis: Analisis | None
    informe: str | None
    # Parcela de la validacion
    validacion: ResultadoValidacion | None
    validaciones: Annotated[list[ResultadoValidacion], operator.add]
    # Quien hizo que (lo escriben todos, solo anexando)
    ultimo_agente: NombreAgente | None
    contribuciones: Annotated[list[Contribucion], operator.add]


# Que claves puede devolver cada nodo (se verifica en tests/test_estado.py y tests/test_grafo.py).
CLAVES_POR_NODO: dict[str, frozenset[str]] = {
    "supervisor": frozenset(
        {"siguiente", "instruccion", "pasos", "intentos", "decisiones", "estado_final", "respuesta_final"}
    ),
    "investigador": frozenset({"hallazgos", "ultimo_agente", "contribuciones", "messages"}),
    "analista": frozenset({"analisis", "ultimo_agente", "contribuciones", "messages"}),
    "redactor": frozenset({"informe", "ultimo_agente", "contribuciones", "messages"}),
    "validacion": frozenset({"validacion", "validaciones"}),
}


def estado_inicial(tarea: str) -> dict[str, object]:
    """Valores con los que arranca una corrida del grafo."""
    return {
        "messages": [HumanMessage(content=tarea)],
        "tarea": tarea,
        "siguiente": "investigador",  # el supervisor lo pisa en su primera decision
        "instruccion": "",
        "pasos": 0,
        "intentos": {},
        "decisiones": [],
        "validaciones": [],
        "contribuciones": [],
        "hallazgos": None,
        "analisis": None,
        "informe": None,
        "validacion": None,
        "ultimo_agente": None,
        "estado_final": None,
        "respuesta_final": None,
    }
