import importlib.metadata
import importlib.util


def is_package_available(name: str) -> bool:
    return importlib.util.find_spec(name) is not None


def is_fastapi_availble():
    return is_package_available("fastapi")


def is_starlette_available():
    return is_package_available("sse_starlette")


def is_uvicorn_available():
    return is_package_available("uvicorn")
