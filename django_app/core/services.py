"""
Communication layer with the remote compute server + file storage helpers.
"""

import os
import base64

import requests
from django.conf import settings


COMPUTE_URL = settings.COMPUTE_SERVER_URL


class ComputeServerError(Exception):
    pass


# ─── Compute Server API ─────────────────────────────────────────────────────

def health_check() -> dict:
    try:
        r = requests.get(f"{COMPUTE_URL}/health", timeout=5)
        r.raise_for_status()
        return r.json()
    except requests.exceptions.RequestException as e:
        raise ComputeServerError(f"Compute server unreachable: {e}")


def decompose_image(image_file, n_components: int = 50, grayscale: bool = False) -> dict:
    try:
        r = requests.post(
            f"{COMPUTE_URL}/api/decompose",
            files={"file": (image_file.name, image_file.read(), "image/jpeg")},
            params={"n_components": n_components, "grayscale": str(grayscale).lower()},
            timeout=120,
        )
        r.raise_for_status()
        return r.json()
    except requests.exceptions.RequestException as e:
        raise ComputeServerError(f"Decomposition failed: {e}")


def reconstruct_image(image_id: str, n_components: int) -> dict:
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


def get_svd_matrices_visual(image_id: str, channel: str = "R") -> dict:
    try:
        r = requests.post(
            f"{COMPUTE_URL}/api/svd_matrices_visual",
            params={"image_id": image_id, "channel": channel},
            timeout=60,
        )
        r.raise_for_status()
        return r.json()
    except requests.exceptions.RequestException as e:
        raise ComputeServerError(f"SVD matrices visual failed: {e}")


def get_multi_reconstruct(image_id: str, k_values: str = "1,5,10,25,50,100,200") -> dict:
    try:
        r = requests.post(
            f"{COMPUTE_URL}/api/multi_reconstruct",
            params={"image_id": image_id, "k_values": k_values},
            timeout=120,
        )
        r.raise_for_status()
        return r.json()
    except requests.exceptions.RequestException as e:
        raise ComputeServerError(f"Multi-reconstruct failed: {e}")


# ─── File Storage Helpers ────────────────────────────────────────────────────

def save_result_image(analysis_pk, name, b64_data):
    """Save a base64-encoded PNG to media/results/{pk}/{name}.png, return URL."""
    dir_path = os.path.join(settings.MEDIA_ROOT, "results", str(analysis_pk))
    os.makedirs(dir_path, exist_ok=True)
    file_path = os.path.join(dir_path, f"{name}.png")
    with open(file_path, "wb") as f:
        f.write(base64.b64decode(b64_data))
    return f"{settings.MEDIA_URL}results/{analysis_pk}/{name}.png"


def get_result_image_url(analysis_pk, name):
    """Get URL for a result image, or None if file doesn't exist."""
    file_path = os.path.join(
        settings.MEDIA_ROOT, "results", str(analysis_pk), f"{name}.png"
    )
    if os.path.exists(file_path):
        return f"{settings.MEDIA_URL}results/{analysis_pk}/{name}.png"
    return None
