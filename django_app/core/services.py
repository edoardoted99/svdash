"""
Communication layer with the remote compute server.
"""

import requests
from django.conf import settings


COMPUTE_URL = settings.COMPUTE_SERVER_URL


class ComputeServerError(Exception):
    pass


def health_check() -> dict:
    """Check if compute server is reachable."""
    try:
        r = requests.get(f"{COMPUTE_URL}/health", timeout=5)
        r.raise_for_status()
        return r.json()
    except requests.exceptions.RequestException as e:
        raise ComputeServerError(f"Compute server unreachable: {e}")


def decompose_image(image_file, n_components: int = 50) -> dict:
    """
    Send image to compute server for SVD decomposition.
    Returns the full JSON response.
    """
    try:
        r = requests.post(
            f"{COMPUTE_URL}/api/decompose",
            files={"file": (image_file.name, image_file.read(), "image/jpeg")},
            params={"n_components": n_components},
            timeout=120,
        )
        r.raise_for_status()
        return r.json()
    except requests.exceptions.RequestException as e:
        raise ComputeServerError(f"Decomposition failed: {e}")


def reconstruct_image(image_id: str, n_components: int) -> dict:
    """
    Request reconstruction at a different n_components.
    Uses cached SVD on the compute server — fast.
    """
    try:
        r = requests.post(
            f"{COMPUTE_URL}/api/reconstruct",
            params={"image_id": image_id, "n_components": n_components},
            timeout=30,
        )
        r.raise_for_status()
        return r.json()
    except requests.exceptions.RequestException as e:
        raise ComputeServerError(f"Reconstruction failed: {e}")


def get_components_visual(image_id: str, channel: str = "R", top_n: int = 5) -> dict:
    """
    Get visualization of individual SVD components.
    """
    try:
        r = requests.post(
            f"{COMPUTE_URL}/api/components_visual",
            params={"image_id": image_id, "channel": channel, "top_n": top_n},
            timeout=30,
        )
        r.raise_for_status()
        return r.json()
    except requests.exceptions.RequestException as e:
        raise ComputeServerError(f"Components visual failed: {e}")