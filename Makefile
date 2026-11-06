# Atajos. Todos asumen el entorno virtual activado (source .venv/bin/activate).
.PHONY: instalar test tipos todo demo diagrama

instalar:
	pip install -r requirements.txt

test:
	pytest

tipos:
	mypy .

todo: instalar test tipos

demo:
	python main.py --demo

diagrama:
	python main.py --diagrama
