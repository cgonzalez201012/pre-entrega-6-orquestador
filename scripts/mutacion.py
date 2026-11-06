"""Prueba de mutacion artesanal: demuestra que los tests detectan errores reales.

Por cada mutante se hace una COPIA del repositorio, se inyecta UN error en el codigo (por
ejemplo, un `>=` que pasa a `>`, o una validacion que se saltea) y se corre pytest. Si pytest
sigue en verde, el mutante "sobrevive": significa que ningun test mira ese comportamiento.

Uso (desde la raiz del repo, con el entorno virtual activado):
    python scripts/mutacion.py            # tarda unos minutos
"""

import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

RAIZ = Path(__file__).resolve().parent.parent

MUTANTES: list[tuple[str, str, str, str]] = [  # (archivo, original, mutado, descripcion)
 ("supervisor.py", 'if estado["pasos"] >= max_pasos:', 'if estado["pasos"] > max_pasos:', "tope de pasos off-by-one"),
 ("supervisor.py", 'return agente != FINISH and intentos.get(agente, 0) >= max_intentos', 'return agente != FINISH and intentos.get(agente, 0) > max_intentos', "tope de intentos off-by-one"),
 ("supervisor.py", 'if destino == FINISH and logico != FINISH:', 'if False:', "sin guarda de cierre prematuro"),
 ("supervisor.py", 'elif destino != FINISH and not requisitos_cumplidos(destino, estado):', 'elif False:', "sin guarda de dependencias"),
 ("supervisor.py", 'validacion.paso > propia.paso', 'validacion.paso < propia.paso', "vigencia del aporte invertida"),
 ("supervisor.py", 'paso = estado["pasos"] + (0 if destino == FINISH else 1)', 'paso = estado["pasos"] + 1', "cerrar consume un paso"),
 ("supervisor.py", 'not hallazgos.fuera_de_alcance and bool', 'bool', "analiza aun fuera de alcance"),
 ("supervisor.py", 'if hay_que_analizar(estado) and not aprobado(estado, "analista"):', 'if not aprobado(estado, "analista"):', "analista obligatorio siempre"),
 ("supervisor.py", 'if not aprobado(estado, "investigador"):\n        return "investigador"', 'if False:\n        return "investigador"', "orden logico sin investigador"),
 ("supervisor.py", 'instruccion = ""\n', 'pass\n', "instruccion no se vacia al cerrar"),
 ("validacion.py", 'elif analisis.ranking != ranking_esperado:', 'elif False:', "no valida el ranking"),
 ("validacion.py", 'elif obtenido != esperado:', 'elif False:', "no recalcula vencimientos"),
 ("validacion.py", 'elif obtenida != esperada:', 'elif False:', "no recalcula brechas"),
 ("validacion.py", 'if inventados:', 'if False:', "acepta IDs inventados"),
 ("validacion.py", 'if faltantes:', 'if False:', "acepta informes incompletos"),
 ("validacion.py", 'if analisis.ranking and analisis.ranking[0] not in citados:', 'if False:', "no exige citar el primero del ranking"),
 ("validacion.py", 'if not hallazgos.herramientas_usadas:', 'if False:', "acepta respuestas sin consultar herramientas"),
 ("validacion.py", 'if detallados and hallazgos.fecha_corte is None:', 'if False:', "acepta datos sin fecha de corte"),
 ("validacion.py", 'if ultima is not None and ultima.error:', 'if False:', "ignora el error del especialista"),
 ("calculos.py", 'JORNADA_HORAS = 8', 'JORNADA_HORAS = 7', "jornada de 7 horas"),
 ("calculos.py", 'weekday() < 5', 'weekday() < 6', "cuenta sabados"),
 ("calculos.py", 'vencido=horas > sla_horas_habiles', 'vencido=horas >= sla_horas_habiles', "borde del SLA"),
 ("calculos.py", 'cumple=medicion_pct >= umbral_pct', 'cumple=medicion_pct > umbral_pct', "borde del umbral"),
 ("calculos.py", '0 if item.vencido else 1', '1 if item.vencido else 0', "prioridad: vencidos al final"),
 ("calculos.py", '-item.exceso_horas,', 'item.exceso_horas,', "prioridad: menos exceso primero"),
 ("calculos.py", 'if hasta < desde:', 'if hasta <= desde:', "fechas iguales dan error"),
 ("graph.py", 'grafo.add_edge(NODO_VALIDACION, NODO_SUPERVISOR)', 'grafo.add_edge(NODO_VALIDACION, END)', "validacion corta el flujo"),
 ("graph.py", 'grafo.add_edge(especialista, NODO_VALIDACION)', 'grafo.add_edge(especialista, NODO_SUPERVISOR)', "especialistas saltean la validacion"),
 ("graph.py", '"redactor": NODO_REDACTOR,', '"redactor": NODO_ANALISTA,', "ruta redactor mal cableada"),
 ("state.py", '"intentos": {}', '"intentos": {"x": 1}', "estado inicial con intentos"),
 ("agents/base.py", 'if llamada["name"] in propias', 'if True', "cuenta herramientas ajenas"),
 ("agents/research_agent.py", 'fuera_de_alcance=resumen.strip().startswith(FUERA_DE_ALCANCE)', 'fuera_de_alcance=False', "no detecta fuera de alcance"),
 ("config.py", 'return PASOS_DEL_GRAFO_POR_RONDA * self.max_pasos + 5', 'return PASOS_DEL_GRAFO_POR_RONDA * self.max_pasos', "recursion limit sin margen"),
    ("agents/research_agent.py", "hallazgos = anteriores.fusionar(de_esta_corrida) if anteriores is not None else de_esta_corrida", "hallazgos = de_esta_corrida", "el refinamiento del investigador pisa lo anterior (bug de la corrida real)"),
    ("agents/analyst_agent.py", "analisis = anterior.fusionar(de_esta_corrida) if anterior is not None else de_esta_corrida", "analisis = de_esta_corrida", "el refinamiento del analista pisa lo anterior"),
    ("state.py", "incidentes[incidente.id] = {**incidentes.get(incidente.id, {}), **vigentes}", "incidentes[incidente.id] = vigentes", "fusion: un listado borra el detalle"),
    ("state.py", "reglas = {r.id: r for r in self.reglas}\n        reglas.update", "reglas = {}\n        reglas.update", "fusion: se pierden las reglas viejas"),
    ("state.py", "ranking=nuevo.ranking or self.ranking,", "ranking=nuevo.ranking,", "fusion: un analisis parcial borra el ranking"),
    ("state.py", "consultas_con_error=nuevos.consultas_con_error,", "consultas_con_error=[*self.consultas_con_error, *nuevos.consultas_con_error],", "fusion: errores viejos persisten"),
 ("trazas.py", 'if contribucion["agente"] not in orden:', 'if True:', "agentes repetidos en la traza"),
]



def _correr(archivo: str, original: str, mutado: str) -> str:
    """Devuelve 'detectado', 'sobrevive' o 'no_aplica' para un mutante."""
    with tempfile.TemporaryDirectory() as carpeta:
        copia = Path(carpeta) / "repo"
        shutil.copytree(
            RAIZ, copia, ignore=shutil.ignore_patterns(".venv", "venv", "__pycache__", ".pytest_cache", ".mypy_cache", ".git")
        )
        ruta = copia / archivo
        texto = ruta.read_text(encoding="utf-8")
        if original not in texto:
            return "no_aplica"
        ruta.write_text(texto.replace(original, mutado, 1), encoding="utf-8")
        resultado = subprocess.run(
            [sys.executable, "-m", "pytest", "-x", "-q", "-p", "no:cacheprovider"],
            cwd=copia, capture_output=True, text=True, timeout=300, check=False,
        )
        return "sobrevive" if resultado.returncode == 0 else "detectado"


def _suite_original_en_verde() -> bool:
    """Sin esto, un test que ya falla haria parecer que 'detecta' cualquier mutante."""
    resultado = subprocess.run(
        [sys.executable, "-m", "pytest", "-x", "-q", "-p", "no:cacheprovider"],
        cwd=RAIZ, capture_output=True, text=True, timeout=300, check=False,
    )
    return resultado.returncode == 0


def main() -> int:
    if not _suite_original_en_verde():
        print("La suite original no esta en verde: arreglala antes de medir mutantes.")
        return 2
    resultados = {m[3]: (m[0], _correr(m[0], m[1], m[2])) for m in MUTANTES}
    detectados = sum(1 for _, estado in resultados.values() if estado == "detectado")
    print(f"mutantes: {len(MUTANTES)} | detectados: {detectados}")
    for descripcion, (archivo, estado) in resultados.items():
        if estado != "detectado":
            print(f"  {estado.upper()}: {archivo} - {descripcion}")
    return 0 if detectados == len(MUTANTES) else 1


if __name__ == "__main__":
    sys.exit(main())
