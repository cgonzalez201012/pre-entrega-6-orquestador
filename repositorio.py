"""Base de datos simulada de incidentes y reglas de calidad de datos.

Es el "sistema externo" que consulta el agente investigador. Lee archivos JSON de /data,
pero lo hace dentro de `asyncio.to_thread` para que cada consulta se comporte como una
operacion de entrada/salida real: no bloquea el bucle de eventos mientras el sistema corre.
"""

import asyncio
import json
import unicodedata
from datetime import date
from functools import lru_cache
from pathlib import Path
from typing import Any

from config import BASE_DIR
from errores import ErrorConsulta

MAESTROS_VALIDOS = ("articulos", "proveedores", "sucursales")
ESTADOS_VALIDOS = ("abierto", "en_curso", "cerrado", "todos")
SEVERIDADES_VALIDAS = ("critica", "alta", "media", "baja")

# Se aceptan los singulares y los nombres con tilde; NO se inventan sinonimos
# ("productos" no es un maestro): si el agente se equivoca, la herramienta se lo dice.
_ALIAS_MAESTRO = {"articulo": "articulos", "proveedor": "proveedores", "sucursal": "sucursales"}


def normalizar(texto: str) -> str:
    """Minusculas, sin tildes y sin espacios sobrantes ('Artículos ' -> 'articulos')."""
    sin_tildes = unicodedata.normalize("NFD", texto.strip().lower())
    return "".join(c for c in sin_tildes if unicodedata.category(c) != "Mn")


class RepositorioCalidad:
    """Consulta incidentes y reglas de calidad de datos (datos ficticios)."""

    def __init__(self, carpeta_datos: Path) -> None:
        self._carpeta = carpeta_datos
        self._incidentes: dict[str, Any] | None = None
        self._reglas: dict[str, Any] | None = None

    def _leer_json(self, nombre: str) -> dict[str, Any]:
        contenido: dict[str, Any] = json.loads((self._carpeta / nombre).read_text(encoding="utf-8"))
        return contenido

    async def _cargar(self) -> tuple[dict[str, Any], dict[str, Any]]:
        # Sin candado a proposito: si dos consultas llegan juntas la primera vez, los JSON
        # se leen dos veces con el mismo resultado. Un Lock quedaria atado al bucle de
        # eventos que lo uso primero y se rompe si el proceso abre otro.
        if self._incidentes is None or self._reglas is None:
            self._incidentes = await asyncio.to_thread(self._leer_json, "incidentes_calidad.json")
            self._reglas = await asyncio.to_thread(self._leer_json, "reglas_calidad.json")
        return self._incidentes, self._reglas

    async def buscar_incidentes(
        self, maestro: str, estado: str = "abierto", severidad: str | None = None
    ) -> dict[str, Any]:
        """Lista los incidentes de un maestro filtrando por estado y severidad."""
        datos, _ = await self._cargar()
        maestro_ok = self._validar_maestro(maestro)
        estado_ok = normalizar(estado)
        if estado_ok not in ESTADOS_VALIDOS:
            raise ErrorConsulta(
                f"El estado {estado!r} no existe.",
                f"Estados validos: {', '.join(ESTADOS_VALIDOS)}.",
            )
        severidad_ok = normalizar(severidad) if severidad else None
        if severidad_ok is not None and severidad_ok not in SEVERIDADES_VALIDAS:
            raise ErrorConsulta(
                f"La severidad {severidad!r} no existe.",
                f"Severidades validas: {', '.join(SEVERIDADES_VALIDAS)}.",
            )

        corte = date.fromisoformat(datos["fecha_corte"])
        encontrados: list[dict[str, Any]] = []
        for inc in datos["incidentes"]:
            if inc["maestro"] != maestro_ok:
                continue
            if estado_ok != "todos" and inc["estado"] != estado_ok:
                continue
            if severidad_ok is not None and inc["severidad"] != severidad_ok:
                continue
            encontrados.append(
                {
                    "id": inc["id"],
                    "titulo": inc["titulo"],
                    "severidad": inc["severidad"],
                    "estado": inc["estado"],
                    "fecha_alta": inc["fecha_alta"],
                    "dias_desde_el_alta": (corte - date.fromisoformat(inc["fecha_alta"])).days,
                }
            )
        return {
            "maestro": maestro_ok,
            "filtros": {"estado": estado_ok, "severidad": severidad_ok},
            "fecha_corte": datos["fecha_corte"],
            "cantidad": len(encontrados),
            "incidentes": encontrados,
        }

    async def detalle_incidente(self, incidente_id: str) -> dict[str, Any]:
        """Devuelve la ficha completa de un incidente, incluidos responsable, SLA y fecha de corte."""
        datos, _ = await self._cargar()
        buscado = incidente_id.strip().upper()
        for inc in datos["incidentes"]:
            if inc["id"] == buscado:
                ficha = dict(inc)
                ficha["sla_horas_habiles"] = datos["sla_horas_habiles"][inc["severidad"]]
                ficha["fecha_corte"] = datos["fecha_corte"]
                return ficha

        # No existe: la ayuda propone los IDs del maestro que corresponde por prefijo
        # (INC-1xxx articulos, 2xxx proveedores, 3xxx sucursales) o, si no, todos.
        prefijo = buscado[4:5] if buscado.startswith("INC-") else ""
        del_prefijo = [i["id"] for i in datos["incidentes"] if i["id"][4:5] == prefijo]
        if del_prefijo:
            maestro = next(i["maestro"] for i in datos["incidentes"] if i["id"] == del_prefijo[0])
            ayuda = f"IDs existentes del maestro de {maestro}: {', '.join(del_prefijo)}."
        else:
            todos = ", ".join(i["id"] for i in datos["incidentes"])
            ayuda = f"Los IDs tienen el formato INC-NNNN. IDs existentes: {todos}."
        raise ErrorConsulta(f"No existe el incidente {incidente_id!r}.", ayuda)

    async def regla(self, regla_id: str) -> dict[str, Any]:
        """Devuelve una regla de calidad con su umbral y su medicion actual."""
        _, reglas = await self._cargar()
        buscada = regla_id.strip().upper()
        for regla in reglas["reglas"]:
            if regla["id"] == buscada:
                ficha = dict(regla)
                ficha["cumple_umbral"] = regla["medicion_actual_pct"] >= regla["umbral_aceptacion_pct"]
                return ficha
        raise ErrorConsulta(
            f"No existe la regla {regla_id!r}.",
            "Los IDs de regla tienen el formato R-XXX-NN (por ejemplo R-ART-01). "
            "El ID de la regla de un incidente figura en su detalle (campo regla_id).",
        )

    @staticmethod
    def _validar_maestro(maestro: str) -> str:
        limpio = normalizar(maestro)
        limpio = _ALIAS_MAESTRO.get(limpio, limpio)
        if limpio not in MAESTROS_VALIDOS:
            raise ErrorConsulta(
                f"El maestro {maestro!r} no existe.",
                f"Maestros validos: {', '.join(MAESTROS_VALIDOS)}.",
            )
        return limpio


@lru_cache(maxsize=1)
def obtener_repositorio() -> RepositorioCalidad:
    """Una sola instancia por proceso: los JSON se leen una vez."""
    return RepositorioCalidad(BASE_DIR / "data")
