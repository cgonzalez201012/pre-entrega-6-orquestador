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


def es_error_irrecuperable(error: Exception) -> bool:
    """True si reintentar no sirve: la API rechazo la solicitud o las credenciales (HTTP 400, 401, 403, 404).

    Esos errores se dejan subir hasta `main.py`, que le explica al usuario que tiene que
    cambiar algo (la clave, el nombre del modelo). Todo lo demas (corte de red, limite de uso,
    error del servidor, timeout) es transitorio: el nodo lo convierte en un aporte fallido y
    el supervisor decide si reintenta, dentro de sus topes.
    """
    return getattr(error, "status_code", None) in {400, 401, 403, 404}
