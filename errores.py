"""Errores propios del proyecto, con mensajes pensados para quien ejecuta el orquestador."""


class ErrorConfiguracion(Exception):
    """Falta o es invalida una variable de entorno necesaria para arrancar."""


class ErrorConsulta(Exception):
    """Una consulta a la base simulada no se pudo resolver.

    Lleva un mensaje y una ayuda: ambos viajan de vuelta al especialista como texto
    para que pueda corregir la llamada.
    """

    def __init__(self, mensaje: str, ayuda: str) -> None:
        super().__init__(mensaje)
        self.mensaje = mensaje
        self.ayuda = ayuda


class ErrorCalculo(Exception):
    """Un calculo del analista recibio datos que no tienen sentido (por ejemplo, fechas al reves)."""

    def __init__(self, mensaje: str, ayuda: str) -> None:
        super().__init__(mensaje)
        self.mensaje = mensaje
        self.ayuda = ayuda
