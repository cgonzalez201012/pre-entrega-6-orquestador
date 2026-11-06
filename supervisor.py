"""Nodo Supervisor: director de trafico del equipo.

El supervisor NO recolecta datos, NO calcula y NO redacta: lee el estado del trabajo, decide
quien interviene ahora (o si hay que cerrar) y evalua los resultados con una rubrica.

Su decision es un objeto Pydantic con `siguiente: Literal[...]`, de modo que el modelo solo
puede elegir uno de los destinos que existen en el grafo.

Despues de la decision del modelo se aplican unas GUARDAS en codigo (`aplicar_guardas`),
que no reemplazan al modelo sino que acotan los errores graves:

  - tope de pasos: contra el "supervisor infinito";
  - no se cierra (FINISH) sin un informe aprobado por la validacion;
  - no se salta un paso del que depende el siguiente (por ejemplo, analizar datos que
    todavia no fueron aprobados);
  - tope de intentos por agente: un especialista no puede ser reenviado a rehacer su
    trabajo mas de `max_intentos` veces.

Toda correccion de las guardas queda registrada en la decision (`corregida_por_guarda`).

Si el modelo no devuelve una decision valida, o la API falla por una causa transitoria, el
nodo no se cae: sigue el orden logico y deja el motivo en la decision. Los errores que
reintentar no arregla (clave invalida, modelo inexistente) suben hasta `main.py`.
"""

from typing import Any

from langchain_core.exceptions import OutputParserException
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.runnables import RunnableConfig
from pydantic import BaseModel, Field, ValidationError

from agents.base import descripcion_de_falla
from agents.research_agent import NodoAsincrono
from errores import es_error_irrecuperable
from state import (
    AGENTES,
    FINISH,
    Contribucion,
    DecisionRegistrada,
    Destino,
    EstadoFinal,
    EstadoOrquestador,
    NombreAgente,
    ResultadoValidacion,
)

MAX_CARACTERES_DE_RESUMEN = 400

PROMPT_SUPERVISOR = """Sos el Supervisor de un equipo de agentes de Gobierno de Datos de Mercado Austral S.A., una
empresa ficticia de retail. Tu rol es el de DIRECTOR DE TRÁFICO: no recolectás datos, no
calculás y no redactás. Solo decidís quién interviene ahora y evaluás los resultados.

Tu equipo:
- investigador: consulta incidentes y reglas de calidad de datos (solo lectura).
- analista: con los datos del investigador calcula la brecha de cada regla contra su umbral,
  el vencimiento del SLA de cada incidente y el ranking de prioridad.
- redactor: escribe el informe final con los datos ya validados.

En cada turno respondés con tres campos: `siguiente` (investigador, analista, redactor o
FINISH), `instruccion` (qué tiene que hacer ese agente: concreta y breve; vacía si es FINISH)
y `motivo` (una frase).

Rúbrica de decisión (validá antes de avanzar):
1. El primero en intervenir es siempre el investigador. Acotá su instrucción a lo que hay que
   recolectar según la tarea (maestro, estado, severidad). Si la tarea no trata de incidentes
   ni de reglas de calidad de datos, igual arrancás por el investigador: él la declarará fuera
   de alcance.
2. Pasás al analista solo si el trabajo del investigador está APROBADO y hay incidentes para
   analizar. Si el investigador declaró la consulta fuera de alcance o no encontró incidentes,
   pasá directo al redactor.
3. Pasás al redactor solo si el trabajo del analista está APROBADO (o si no hacía falta analizar).
4. Si la validación RECHAZÓ el último aporte de un especialista, elegilo de nuevo: `siguiente` es
   ese mismo especialista y en `instruccion` resumí qué tiene que corregir.
5. Cerrás con FINISH solamente cuando el informe del redactor está APROBADO.
6. No hagas el trabajo de los especialistas: en `instruccion` no incluyas datos, cálculos ni texto
   del informe.
7. Los pasos son limitados: si vas a quedarte sin pasos, cerrá con lo que haya."""

INSTRUCCION_POR_DEFECTO: dict[str, str] = {
    "investigador": "Recolectá los datos necesarios para resolver esta tarea: {tarea}",
    "analista": "Calculá la brecha de cada regla, el vencimiento del SLA de cada incidente y el ranking de prioridad.",
    "redactor": "Redactá el informe final con los datos validados.",
}


class DecisionSupervisor(BaseModel):
    """Lo que el modelo del supervisor tiene que devolver en cada turno."""

    siguiente: Destino = Field(description="Quién interviene ahora, o FINISH para cerrar.")
    instruccion: str = Field(default="", description="Instrucción concreta y breve para ese agente (vacía si FINISH).")
    motivo: str = Field(description="Una frase que explique la decisión.")


# ---------- Que se aprobo y que sigue ----------


def _ultima_validacion(estado: EstadoOrquestador, agente: NombreAgente) -> ResultadoValidacion | None:
    return next((v for v in reversed(estado["validaciones"]) if v.agente == agente), None)


def aprobado(estado: EstadoOrquestador, agente: NombreAgente) -> bool:
    """True si el ultimo aporte del agente fue aprobado y sigue vigente.

    Un aporte deja de estar vigente si un agente anterior volvio a entregar despues de que
    este fuera aprobado: el analista, por ejemplo, calculo sobre datos que ya cambiaron.
    """
    propia = _ultima_validacion(estado, agente)
    if propia is None or not propia.aprobada:
        return False
    for anterior in AGENTES[: AGENTES.index(agente)]:
        validacion = _ultima_validacion(estado, anterior)
        if validacion is not None and validacion.paso > propia.paso:
            return False
    return True


def hay_que_analizar(estado: EstadoOrquestador) -> bool:
    """True si hay incidentes detallados para analizar (y la consulta no quedo fuera de alcance)."""
    hallazgos = estado["hallazgos"]
    return hallazgos is not None and not hallazgos.fuera_de_alcance and bool(hallazgos.incidentes_detallados())


def siguiente_logico(estado: EstadoOrquestador) -> Destino:
    """Primer paso pendiente segun las dependencias: investigador -> analista -> redactor -> cierre."""
    if not aprobado(estado, "investigador"):
        return "investigador"
    if hay_que_analizar(estado) and not aprobado(estado, "analista"):
        return "analista"
    if not aprobado(estado, "redactor"):
        return "redactor"
    return FINISH


def requisitos_cumplidos(destino: Destino, estado: EstadoOrquestador) -> bool:
    """True si el agente elegido tiene lo que necesita para trabajar."""
    if destino == "investigador":
        return True
    if not aprobado(estado, "investigador"):
        return False
    if destino == "analista":
        return hay_que_analizar(estado)
    if destino == "redactor":
        return not hay_que_analizar(estado) or aprobado(estado, "analista")
    return False


# ---------- Guardas ----------


def aplicar_guardas(
    destino: Destino, estado: EstadoOrquestador, max_pasos: int, max_intentos: int
) -> tuple[Destino, str | None]:
    """Acota la eleccion del modelo. Devuelve (destino final, razon si hubo correccion)."""
    logico = siguiente_logico(estado)
    intentos = estado["intentos"]

    def agotado(agente: Destino) -> bool:
        return agente != FINISH and intentos.get(agente, 0) >= max_intentos

    if estado["pasos"] >= max_pasos:
        razon = None if destino == FINISH else f"Se alcanzó el límite de {max_pasos} pasos: el sistema cierra."
        return FINISH, razon

    final: Destino = destino
    razon = None
    if destino == FINISH and logico != FINISH:
        final = logico
        razon = f"Intentó cerrar sin un informe aprobado: el paso pendiente es {logico}."
    elif destino != FINISH and not requisitos_cumplidos(destino, estado):
        final = logico
        razon = f"{destino} todavía no tiene lo que necesita: el paso pendiente es {logico}."

    if agotado(final):
        if logico != final and (logico == FINISH or not agotado(logico)):
            razon = f"{final} ya usó sus {max_intentos} intentos: sigue {logico}."
            final = logico
        else:
            razon = f"{final} ya usó sus {max_intentos} intentos sin lograr un aporte aprobado: el sistema cierra."
            final = FINISH
    return final, razon


# ---------- Contexto y cierre ----------


def _recortar(texto: str) -> str:
    texto = " ".join(texto.split())
    return texto if len(texto) <= MAX_CARACTERES_DE_RESUMEN else texto[:MAX_CARACTERES_DE_RESUMEN] + "…"


def resumen_del_trabajo(estado: EstadoOrquestador, max_pasos: int, max_intentos: int) -> str:
    """Lo que ve el supervisor: la tarea y el estado de cada agente. Sin datos crudos ni mensajes."""
    lineas = [
        f"TAREA DEL USUARIO: {estado['tarea']}",
        f"DELEGACIONES USADAS: {estado['pasos']} de {max_pasos}. INTENTOS MAXIMOS POR AGENTE: {max_intentos}.",
        "",
        "ESTADO DEL TRABAJO:",
    ]
    for agente in AGENTES:
        aportes = [c for c in estado["contribuciones"] if c.agente == agente]
        if not aportes:
            lineas.append(f"- {agente}: todavía no intervino.")
            continue
        validacion = _ultima_validacion(estado, agente)
        veredicto = "sin validar"
        if validacion is not None:
            veredicto = "APROBADO" if validacion.aprobada else "RECHAZADO"
        lineas.append(f"- {agente}: intervino {len(aportes)} vez/veces. Última validación: {veredicto}.")
        lineas.append(f"  Resumen de su último aporte: {_recortar(_resumen_de(aportes[-1]))}")
        if validacion is not None and not validacion.aprobada:
            for problema in validacion.problemas:
                lineas.append(f"  Problema detectado: {_recortar(problema)}")
    if estado["hallazgos"] is not None:
        lineas.append("")
        if estado["hallazgos"].fuera_de_alcance:
            lineas.append("DATO CLAVE: el investigador declaró la consulta FUERA DE ALCANCE.")
        else:
            lineas.append(
                f"DATO CLAVE: el investigador reunió {len(estado['hallazgos'].incidentes_detallados())} "
                "incidente(s) con ficha completa."
            )
    return "\n".join(lineas)


def _resumen_de(contribucion: Contribucion) -> str:
    return contribucion.resumen


def cerrar(estado: EstadoOrquestador, max_pasos: int) -> dict[str, Any]:
    """Estado final y respuesta final de la corrida (se llama cuando el destino es FINISH)."""
    informe = estado["informe"]
    if aprobado(estado, "redactor") and informe:
        return {"estado_final": "completado", "respuesta_final": informe}

    final: EstadoFinal = "limite_de_pasos" if estado["pasos"] >= max_pasos else "validacion_fallida"
    pendientes = estado["validacion"].problemas if estado["validacion"] and not estado["validacion"].aprobada else []
    causa = (
        f"alcancé el límite de {max_pasos} pasos"
        if final == "limite_de_pasos"
        else "los controles de validación no aprobaron el trabajo de un especialista"
    )
    aviso = f"No pude completar la tarea de forma confiable: {causa}."
    if pendientes:
        aviso += " Problemas pendientes: " + " | ".join(pendientes)
    return {"estado_final": final, "respuesta_final": aviso}


# ---------- Nodo del grafo ----------


def crear_nodo_supervisor(llm: BaseChatModel, max_pasos: int, max_intentos: int) -> NodoAsincrono:
    """Devuelve el nodo del grafo principal que corre al supervisor."""
    decisor = llm.with_structured_output(DecisionSupervisor)

    async def nodo_supervisor(estado: EstadoOrquestador, config: RunnableConfig) -> dict[str, Any]:
        contexto = resumen_del_trabajo(estado, max_pasos, max_intentos)
        # Si el modelo no decide (forma invalida o fallo transitorio de la API) el supervisor no
        # se cae: sigue el orden logico y deja el motivo registrado en la decision.
        falla: str | None = None
        try:
            respuesta = await decisor.ainvoke(
                [SystemMessage(content=PROMPT_SUPERVISOR), HumanMessage(content=contexto)], config
            )
            decision = DecisionSupervisor.model_validate(respuesta)
        except (OutputParserException, ValidationError):
            falla = "La decisión del modelo no tenía la forma pedida: se siguió el orden lógico."
        except Exception as error:  # noqa: BLE001 - un fallo transitorio del modelo no tira abajo el grafo
            if es_error_irrecuperable(error):
                raise
            falla = f"{descripcion_de_falla(error)} Se siguió el orden lógico."
        if falla is not None:
            decision = DecisionSupervisor(siguiente=siguiente_logico(estado), motivo=falla)

        destino, razon = aplicar_guardas(decision.siguiente, estado, max_pasos, max_intentos)
        if falla is not None:
            razon = falla
        instruccion = decision.instruccion.strip()
        if destino == FINISH:
            instruccion = ""
        elif destino != decision.siguiente or not instruccion:
            instruccion = INSTRUCCION_POR_DEFECTO[destino].format(tarea=estado["tarea"])

        # `pasos` cuenta delegaciones a especialistas: la decision de cerrar no consume un paso.
        paso = estado["pasos"] + (0 if destino == FINISH else 1)
        registro = DecisionRegistrada(
            paso=paso,
            siguiente=destino,
            instruccion=instruccion,
            motivo=decision.motivo,
            corregida_por_guarda=razon is not None,
            razon_guarda=razon,
        )
        actualizacion: dict[str, Any] = {
            "siguiente": destino,
            "instruccion": instruccion,
            "pasos": paso,
            "decisiones": [registro],
            "intentos": {} if destino == FINISH else {destino: 1},
        }
        if destino == FINISH:
            actualizacion.update(cerrar(estado, max_pasos))
        return actualizacion

    return nodo_supervisor
