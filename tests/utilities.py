""" test utilities """
import os


def get_resource_path(name: str, extension: str) -> str:
    "get resources path"
    current_path = os.path.dirname(__file__)
    path = os.path.join(current_path, f"./resources/{name}")
    return f"{path}.{extension}"
