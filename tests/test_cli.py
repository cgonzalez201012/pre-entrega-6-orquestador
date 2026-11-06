"""Linea de comandos: errores amigables y corridas de punta a punta con modelos falsos."""

import json
from pathlib import Path

import anthropic
import httpx
import pytest

import main
from config import Configuracion
from graph import construir_grafo
from tests.fakes import SupervisorFalso, decidir, especialistas_falsos, guion_analista, guion_investigador, informe


def _grafo_falso(config: Configuracion):  # type: ignore[no-untyped-def]
    supervisor = SupervisorFalso(
        guion=[decidir("investigador"), decidir("analista"), decidir("redactor"), decidir("FINISH")]
    )
    especialistas = especialistas_falsos(guion_investigador(), guion_analista(), [informe("INC-1001", "INC-1002", "R-ART-01")])
    return construir_grafo(supervisor, especialistas, max_pasos=10, max_intentos=3, recursion_especialista=12)


@pytest.fixture(autouse=True)
def _sin_dotenv(monkeypatch: pytest.MonkeyPatch) -> None:
    """Los tests de CLI no deben leer un .env real ni heredar claves del entorno."""
    monkeypatch.setattr("config.load_dotenv", lambda *a, **k: False)
    for variable in ("ANTHROPIC_API_KEY", "MODELO_LLM", "MAX_PASOS"):
        monkeypatch.delenv(variable, raising=False)


def test_sin_argumentos_explica_que_hay_que_indicar(capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ANTHROPIC_API_KEY", "clave-falsa")
    assert main.main([]) == 2
    assert "Indicá una tarea" in capsys.readouterr().out


def test_sin_clave_el_error_es_claro_y_no_un_traceback(capsys: pytest.CaptureFixture[str]) -> None:
    assert main.main(["una tarea"]) == 1
    assert "Falta ANTHROPIC_API_KEY" in capsys.readouterr().err


def test_una_tarea_corre_el_grafo_y_muestra_el_informe_y_el_resumen(
    capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("ANTHROPIC_API_KEY", "clave-falsa")
    monkeypatch.setattr(main, "crear_grafo", _grafo_falso)
    assert main.main(["Priorizá los críticos de artículos"]) == 0
    salida = capsys.readouterr().out
    assert "INC-1001" in salida
    assert "estado: completado | agentes: investigador, analista, redactor | delegaciones: 3" in salida


def test_con_verbose_muestra_la_historia_de_la_delegacion(
    capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("ANTHROPIC_API_KEY", "clave-falsa")
    monkeypatch.setattr(main, "crear_grafo", _grafo_falso)
    assert main.main(["-v", "Priorizá los críticos de artículos"]) == 0
    salida = capsys.readouterr().out
    assert "[Paso 1] Supervisor → investigador" in salida and "Validación: APROBADA" in salida


def test_la_demo_guarda_la_traza_en_json_y_log(
    capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("ANTHROPIC_API_KEY", "clave-falsa")
    monkeypatch.setattr(main, "crear_grafo", _grafo_falso)
    monkeypatch.setattr(main, "ESCENARIOS", [("1. Prueba", "Priorizá los críticos de artículos")])
    monkeypatch.setattr(main, "CARPETA_TRAZAS", tmp_path)
    assert main.main(["--demo"]) == 0
    datos = json.loads((tmp_path / "traza-demo.json").read_text(encoding="utf-8"))
    assert datos["corridas"][0]["estado_final"] == "completado" and datos["max_pasos"] == 10
    assert (tmp_path / "traza-demo.log").exists() and "traza-demo.json" in capsys.readouterr().out


def test_el_diagrama_se_genera_sin_clave(capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    llamado: dict[str, object] = {}

    def falso(grafo: object) -> tuple[Path, None]:
        llamado["grafo"] = grafo
        return tmp_path / "grafo.mmd", None

    monkeypatch.setattr(main, "guardar_diagrama", falso)
    assert main.main(["--diagrama"]) == 0
    assert llamado["grafo"] is not None and "No pude generar el PNG" in capsys.readouterr().out


def _error_http(clase: type[anthropic.APIStatusError], estado: int) -> anthropic.APIStatusError:
    respuesta = httpx.Response(estado, request=httpx.Request("POST", "https://api.anthropic.com/v1/messages"))
    return clase("mensaje de prueba", response=respuesta, body=None)


@pytest.mark.parametrize(
    ("error", "mensaje"),
    [
        (_error_http(anthropic.AuthenticationError, 401), "rechazó la clave"),
        (_error_http(anthropic.RateLimitError, 429), "límite de uso"),
        (_error_http(anthropic.InternalServerError, 500), "error (500)"),
        (anthropic.APIConnectionError(request=httpx.Request("POST", "https://api.anthropic.com")), "conectarme"),
    ],
)
def test_los_errores_de_anthropic_se_traducen_a_mensajes_amigables(
    error: Exception, mensaje: str, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    def explotar(args: object) -> int:
        raise error

    monkeypatch.setattr(main, "_ejecutar", explotar)
    assert main.main(["tarea"]) == 1
    assert mensaje in capsys.readouterr().err


def test_si_el_grafo_se_pasa_de_recursion_se_explica_que_hacer(capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch) -> None:
    from langgraph.errors import GraphRecursionError

    def explotar(args: object) -> int:
        raise GraphRecursionError("demasiados pasos")

    monkeypatch.setattr(main, "_ejecutar", explotar)
    assert main.main(["tarea"]) == 1
    assert "MAX_PASOS" in capsys.readouterr().err


def test_los_escenarios_de_la_demo_incluyen_uno_fuera_de_alcance() -> None:
    assert len(main.ESCENARIOS) == 3
    assert any("gerente de datos" in tarea for _, tarea in main.ESCENARIOS)
