"""Configuracion del orquestador: todo sale de variables de entorno (archivo .env)."""

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

from errores import ErrorConfiguracion

BASE_DIR = Path(__file__).resolve().parent

MODELO_POR_DEFECTO = "claude-haiku-4-5"
MAX_PASOS_POR_DEFECTO = 10
MAX_INTENTOS_POR_DEFECTO = 3
RECURSION_ESPECIALISTA_POR_DEFECTO = 12

# Cada ronda del grafo principal son 3 pasos: supervisor -> especialista -> validacion.
PASOS_DEL_GRAFO_POR_RONDA = 3


@dataclass(frozen=True)
class Configuracion:
    """Valores con los que se arma el orquestador.

    - `max_pasos`: cuantas veces como maximo el supervisor puede mandar a trabajar a un
      especialista en una misma tarea (condicion de parada contra el "supervisor infinito").
    - `max_intentos_por_agente`: cuantas veces se le puede pedir su trabajo a un mismo
      especialista (1 corrida inicial + los refinamientos).
    - `recursion_especialista`: techo de pasos del ciclo ReAct interno de cada especialista.
    """

    anthropic_api_key: str
    modelo_supervisor: str
    modelo_especialistas: str
    max_pasos: int
    max_intentos_por_agente: int
    recursion_especialista: int

    @property
    def recursion_limit_grafo(self) -> int:
        """Techo de pasos del grafo principal, con margen sobre el maximo teorico.

        Con `max_pasos` envios a especialistas caben `max_pasos + 1` corridas del supervisor
        (la ultima es la que cierra) y 2 pasos mas por cada envio (especialista + validacion).
        """
        return PASOS_DEL_GRAFO_POR_RONDA * self.max_pasos + 5

    @classmethod
    def desde_entorno(cls, *, requerir_clave: bool = True, cargar_dotenv: bool = True) -> "Configuracion":
        """Lee la configuracion del entorno.

        `requerir_clave=False` sirve para comandos que no llaman al modelo
        (por ejemplo, generar el diagrama del grafo).
        """
        if cargar_dotenv:
            load_dotenv(BASE_DIR / ".env", override=False)

        clave = os.environ.get("ANTHROPIC_API_KEY", "").strip()
        if requerir_clave and not clave:
            raise ErrorConfiguracion(
                "Falta ANTHROPIC_API_KEY. Copiá .env.example a .env y pegá tu clave "
                "de Anthropic (o exportala como variable de entorno)."
            )

        modelo_base = os.environ.get("MODELO_LLM", MODELO_POR_DEFECTO).strip() or MODELO_POR_DEFECTO
        return cls(
            anthropic_api_key=clave,
            modelo_supervisor=os.environ.get("MODELO_SUPERVISOR", "").strip() or modelo_base,
            modelo_especialistas=os.environ.get("MODELO_ESPECIALISTAS", "").strip() or modelo_base,
            max_pasos=_entero("MAX_PASOS", MAX_PASOS_POR_DEFECTO, minimo=3),
            max_intentos_por_agente=_entero("MAX_INTENTOS_POR_AGENTE", MAX_INTENTOS_POR_DEFECTO, minimo=1),
            recursion_especialista=_entero(
                "RECURSION_ESPECIALISTA", RECURSION_ESPECIALISTA_POR_DEFECTO, minimo=4
            ),
        )


def _entero(nombre: str, por_defecto: int, *, minimo: int) -> int:
    """Lee un entero del entorno y valida que no sea absurdamente chico."""
    texto = os.environ.get(nombre, "").strip()
    if not texto:
        return por_defecto
    try:
        valor = int(texto)
    except ValueError as error:
        raise ErrorConfiguracion(f"{nombre} tiene que ser un numero entero (vale {texto!r}).") from error
    if valor < minimo:
        raise ErrorConfiguracion(f"{nombre} tiene que ser al menos {minimo} (vale {valor}).")
    return valor
