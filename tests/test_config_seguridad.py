"""Configuracion, higiene del repositorio (sin secretos) y reproducibilidad."""

import ast
import re
from pathlib import Path

import pytest

from config import BASE_DIR, Configuracion
from errores import ErrorConfiguracion

RAIZ = Path(BASE_DIR)
VARIABLES = (
    "ANTHROPIC_API_KEY", "MODELO_LLM", "MODELO_SUPERVISOR", "MODELO_ESPECIALISTAS",
    "MAX_PASOS", "MAX_INTENTOS_POR_AGENTE", "RECURSION_ESPECIALISTA",
)


@pytest.fixture(autouse=True)
def _entorno_limpio(monkeypatch: pytest.MonkeyPatch) -> None:
    for variable in VARIABLES:
        monkeypatch.delenv(variable, raising=False)


def _archivos_del_repo() -> list[Path]:
    ignorados = {".venv", "__pycache__", ".pytest_cache", ".mypy_cache", ".git", ".ipynb_checkpoints"}
    return [
        ruta for ruta in RAIZ.rglob("*")
        if ruta.is_file() and not (set(ruta.relative_to(RAIZ).parts) & ignorados) and ruta.name != ".env"
    ]


# ---------- Configuracion ----------


def test_valores_por_defecto() -> None:
    config = Configuracion.desde_entorno(cargar_dotenv=False, requerir_clave=False)
    assert (config.modelo_supervisor, config.modelo_especialistas) == ("claude-haiku-4-5", "claude-haiku-4-5")
    assert (config.max_pasos, config.max_intentos_por_agente, config.recursion_especialista) == (10, 3, 12)


def test_sin_clave_falla_con_un_mensaje_que_dice_que_hacer() -> None:
    with pytest.raises(ErrorConfiguracion, match=r"Copiá \.env\.example"):
        Configuracion.desde_entorno(cargar_dotenv=False)


def test_la_clave_se_lee_del_entorno_y_se_limpia(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ANTHROPIC_API_KEY", "  clave  ")
    assert Configuracion.desde_entorno(cargar_dotenv=False).anthropic_api_key == "clave"


def test_un_modelo_por_agente_pisa_al_modelo_general(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MODELO_LLM", "modelo-general")
    monkeypatch.setenv("MODELO_SUPERVISOR", "modelo-del-supervisor")
    config = Configuracion.desde_entorno(cargar_dotenv=False, requerir_clave=False)
    assert (config.modelo_supervisor, config.modelo_especialistas) == ("modelo-del-supervisor", "modelo-general")


@pytest.mark.parametrize(
    ("variable", "valor", "fragmento"),
    [
        ("MAX_PASOS", "dos", "numero entero"),
        ("MAX_PASOS", "2", "al menos 3"),
        ("MAX_INTENTOS_POR_AGENTE", "0", "al menos 1"),
        ("RECURSION_ESPECIALISTA", "3", "al menos 4"),
    ],
)
def test_los_limites_absurdos_se_rechazan(monkeypatch: pytest.MonkeyPatch, variable: str, valor: str, fragmento: str) -> None:
    monkeypatch.setenv(variable, valor)
    with pytest.raises(ErrorConfiguracion, match=fragmento):
        Configuracion.desde_entorno(cargar_dotenv=False, requerir_clave=False)


def test_el_recursion_limit_del_grafo_deja_margen_sobre_el_maximo_teorico() -> None:
    """Con N delegaciones hacen falta 3N + 1 pasos del grafo (supervisor, especialista, validacion y el cierre)."""
    for max_pasos in (3, 10, 25):
        config = Configuracion("k", "m", "m", max_pasos, 3, 12)
        assert config.recursion_limit_grafo > 3 * max_pasos + 1


def test_la_configuracion_es_inmutable() -> None:
    config = Configuracion("k", "m", "m", 10, 3, 12)
    with pytest.raises(AttributeError):
        config.max_pasos = 99  # type: ignore[misc]


# ---------- Secretos ----------


def test_ningun_archivo_del_repo_contiene_una_clave_de_api() -> None:
    patron = re.compile(r"sk-ant-[A-Za-z0-9_\-]{10,}|AKIA[0-9A-Z]{16}|tvly-[A-Za-z0-9]{10,}|ghp_[A-Za-z0-9]{20,}")
    for ruta in _archivos_del_repo():
        if ruta.suffix in {".png", ".pyc"}:
            continue
        assert not patron.search(ruta.read_text(encoding="utf-8", errors="ignore")), f"posible clave en {ruta}"


def test_el_gitignore_excluye_el_env_pero_no_el_ejemplo() -> None:
    reglas = (RAIZ / ".gitignore").read_text(encoding="utf-8").splitlines()
    assert ".env" in reglas and "!.env.example" in reglas
    assert ".venv/" in reglas and "__pycache__/" in reglas


def test_el_env_example_tiene_un_marcador_y_documenta_todas_las_variables() -> None:
    texto = (RAIZ / ".env.example").read_text(encoding="utf-8")
    assert "ANTHROPIC_API_KEY=pega-aca-tu-clave" in texto
    for variable in VARIABLES:
        assert variable in texto, variable


def test_no_hay_un_env_con_secretos_dentro_del_repo_publicado() -> None:
    """El .env puede existir en la maquina de Cris (esta ignorado); aca se verifica que git lo ignora."""
    assert ".env" in (RAIZ / ".gitignore").read_text(encoding="utf-8").splitlines()


# ---------- Reproducibilidad ----------


@pytest.mark.parametrize("archivo", ["requirements.txt", "requirements-notebook.txt"])
def test_las_dependencias_estan_fijadas_a_una_version_exacta(archivo: str) -> None:
    for linea in (RAIZ / archivo).read_text(encoding="utf-8").splitlines():
        linea = linea.strip()
        if not linea or linea.startswith(("#", "-r ")):
            continue
        assert re.fullmatch(r"[A-Za-z0-9_.\-]+==\d+(\.\d+)*", linea), linea


def test_requirements_incluye_lo_necesario_para_correr_pytest_con_un_solo_comando() -> None:
    texto = (RAIZ / "requirements.txt").read_text(encoding="utf-8")
    for paquete in ("langgraph", "langchain-anthropic", "pytest==", "pytest-asyncio==", "pydantic"):
        assert paquete in texto, paquete


# ---------- Calidad del codigo ----------


def _modulos_del_proyecto() -> list[Path]:
    modulos = [ruta for ruta in RAIZ.glob("*.py")] + [ruta for ruta in (RAIZ / "agents").glob("*.py")]
    return sorted(modulos)


def test_todas_las_funciones_del_proyecto_tienen_anotaciones_de_tipo() -> None:
    sin_tipos: list[str] = []
    for ruta in _modulos_del_proyecto():
        for nodo in ast.walk(ast.parse(ruta.read_text(encoding="utf-8"))):
            if isinstance(nodo, (ast.FunctionDef, ast.AsyncFunctionDef)):
                argumentos = [a for a in [*nodo.args.args, *nodo.args.kwonlyargs] if a.arg not in {"self", "cls"}]
                if nodo.returns is None or any(a.annotation is None for a in argumentos):
                    sin_tipos.append(f"{ruta.name}:{nodo.name}")
    assert sin_tipos == []


def test_todos_los_modulos_tienen_docstring() -> None:
    for ruta in _modulos_del_proyecto():
        if ruta.name == "__init__.py" and not ruta.read_text(encoding="utf-8").strip():
            continue
        assert ast.get_docstring(ast.parse(ruta.read_text(encoding="utf-8"))), ruta.name


def test_ningun_modulo_manda_temperature_al_modelo() -> None:
    """La API rechaza `temperature` con este modelo (400): no tiene que aparecer en el codigo que arma el modelo."""
    codigo = (RAIZ / "ejecucion.py").read_text(encoding="utf-8")
    llamadas = [n for n in ast.walk(ast.parse(codigo)) if isinstance(n, ast.keyword) and n.arg == "temperature"]
    assert llamadas == []


# ---------- Documentacion ----------


def test_el_diagrama_del_readme_es_el_que_genera_langgraph() -> None:
    """Si cambia el grafo y nadie regenera el diagrama, este test avisa."""
    from graph import construir_grafo
    from tests.fakes import SupervisorFalso, decidir, especialistas_falsos

    grafo = construir_grafo(
        SupervisorFalso(guion=[decidir("FINISH")]), especialistas_falsos(),
        max_pasos=10, max_intentos=3, recursion_especialista=12,
    )
    generado = grafo.get_graph().draw_mermaid().rstrip("\n")
    assert (RAIZ / "docs" / "grafo.mmd").read_text(encoding="utf-8").rstrip("\n") == generado
    assert f"```mermaid\n{generado}\n```" in (RAIZ / "README.md").read_text(encoding="utf-8")


def test_el_readme_no_deja_marcadores_sin_completar() -> None:
    texto = (RAIZ / "README.md").read_text(encoding="utf-8")
    assert "{{" not in texto and "(pendiente)" not in texto and "TODO" not in texto


def test_el_readme_referencia_archivos_que_existen() -> None:
    texto = (RAIZ / "README.md").read_text(encoding="utf-8")
    for ruta in re.findall(r"\(((?:docs|trazas|notebooks)/[^)\s]+)\)", texto):
        assert (RAIZ / ruta).exists(), ruta


# ---------- Evidencia ----------


def test_el_notebook_de_demostracion_viene_ejecutado_y_sin_secretos() -> None:
    import json

    cuaderno = json.loads((RAIZ / "notebooks" / "demo_delegacion.ipynb").read_text(encoding="utf-8"))
    codigo = [celda for celda in cuaderno["cells"] if celda["cell_type"] == "code"]
    assert len(codigo) >= 5
    for celda in codigo:
        assert celda["execution_count"] is not None and celda["outputs"], "celda sin ejecutar: " + "".join(celda["source"])[:60]
        assert not any(salida["output_type"] == "error" for salida in celda["outputs"])
    texto = "".join(
        "".join(salida.get("text", [])) for celda in codigo for salida in celda["outputs"] if salida["output_type"] == "stream"
    )
    for escenario in ("1. Priorización multi-agente", "2. Otro maestro", "3. Consulta fuera de alcance"):
        assert escenario in texto
    assert "[Cierre] Supervisor → FINISH" in texto and "Estado final: completado" in texto


def test_la_traza_real_de_la_demo_esta_en_el_repo_y_es_coherente() -> None:
    import json

    traza = json.loads((RAIZ / "trazas" / "traza-demo.json").read_text(encoding="utf-8"))
    assert [c["estado_final"] for c in traza["corridas"]] == ["completado"] * 3
    assert "analista" not in traza["corridas"][2]["agentes_que_intervinieron"]  # fuera de alcance: se saltea
    assert traza["corridas"][0]["agentes_que_intervinieron"] == ["investigador", "analista", "redactor"]
    assert (RAIZ / "trazas" / "traza-demo.log").read_text(encoding="utf-8").count("Estado final: completado") == 3
