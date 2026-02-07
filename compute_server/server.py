"""
SVDash Compute Server
=====================
FastAPI service that performs SVD/PCA image decomposition on Apple MPS.
Runs on the remote Mac, accessible via SSH tunnel.

Usage:
    uvicorn server:app --host 127.0.0.1 --port 8001 --reload
"""

import io
import base64
import time
from typing import Optional

import torch
import numpy as np
from PIL import Image
from fastapi import FastAPI, UploadFile, File, Query, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel

# ─── Setup ────────────────────────────────────────────────────────────────────

app = FastAPI(
    title="SVDash Compute Server",
    description="SVD/PCA image decomposition engine running on MPS",
    version="0.1.0",
)

DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"[SVDash Compute] Device: {DEVICE}")


# ─── Helpers ──────────────────────────────────────────────────────────────────

def load_image(file_bytes: bytes) -> np.ndarray:
    """Load image bytes into a float64 numpy array (H, W, 3)."""
    img = Image.open(io.BytesIO(file_bytes)).convert("RGB")
    return np.array(img, dtype=np.float64)


def svd_channel(channel_np: np.ndarray, device: torch.device):
    """Compute full SVD on a single channel (H, W) on the given device."""
    X = torch.tensor(channel_np, dtype=torch.float32, device=device)
    mean = X.mean(dim=0)
    X_centered = X - mean
    U, S, Vt = torch.linalg.svd(X_centered, full_matrices=False)
    return U, S, Vt, mean


def reconstruct_channel(U, S, Vt, mean, n_components: int) -> np.ndarray:
    """Reconstruct a channel from truncated SVD components."""
    U_k = U[:, :n_components]
    S_k = S[:n_components]
    Vt_k = Vt[:n_components, :]
    reconstructed = U_k @ torch.diag(S_k) @ Vt_k + mean
    return reconstructed.clamp(0, 255).cpu().numpy()


def compute_psnr(original: np.ndarray, reconstructed: np.ndarray) -> float:
    """Peak Signal-to-Noise Ratio."""
    mse = np.mean((original - reconstructed) ** 2)
    if mse == 0:
        return float("inf")
    return float(10 * np.log10(255.0**2 / mse))


def compute_ssim_channel(orig: np.ndarray, recon: np.ndarray) -> float:
    """
    Simplified SSIM for a single channel.
    Uses global statistics (not windowed) for speed.
    """
    C1 = (0.01 * 255) ** 2
    C2 = (0.03 * 255) ** 2

    mu_x = orig.mean()
    mu_y = recon.mean()
    sigma_x2 = orig.var()
    sigma_y2 = recon.var()
    sigma_xy = np.mean((orig - mu_x) * (recon - mu_y))

    numerator = (2 * mu_x * mu_y + C1) * (2 * sigma_xy + C2)
    denominator = (mu_x**2 + mu_y**2 + C1) * (sigma_x2 + sigma_y2 + C2)

    return float(numerator / denominator)


def numpy_to_base64_png(img_array: np.ndarray) -> str:
    """Encode a (H, W, 3) uint8 array to base64 PNG string."""
    img = Image.fromarray(img_array.astype(np.uint8))
    buffer = io.BytesIO()
    img.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("utf-8")


def find_k_for_threshold(S: torch.Tensor, threshold: float) -> int:
    """Find minimum number of components for a given variance threshold."""
    var = S**2
    var_cum = torch.cumsum(var, 0) / var.sum()
    indices = (var_cum >= threshold).nonzero()
    if len(indices) == 0:
        return int(S.shape[0])
    return int(indices[0][0]) + 1


# ─── Cache for SVD results (per image hash) ──────────────────────────────────
# Simple in-memory cache to avoid recomputing SVD when user changes k

_svd_cache: dict = {}


def get_or_compute_svd(image_hash: str, img_array: np.ndarray):
    """Get cached SVD or compute it."""
    if image_hash in _svd_cache:
        return _svd_cache[image_hash]

    channels = {}
    for i, name in enumerate(["R", "G", "B"]):
        U, S, Vt, mean = svd_channel(img_array[:, :, i], DEVICE)
        channels[name] = {"U": U, "S": S, "Vt": Vt, "mean": mean}

    _svd_cache[image_hash] = {
        "channels": channels,
        "img_array": img_array,
    }

    # Keep cache small — max 10 images
    if len(_svd_cache) > 10:
        oldest = next(iter(_svd_cache))
        del _svd_cache[oldest]

    return _svd_cache[image_hash]


def image_hash(file_bytes: bytes) -> str:
    """Simple hash for cache key."""
    import hashlib
    return hashlib.md5(file_bytes).hexdigest()


# ─── Endpoints ────────────────────────────────────────────────────────────────

@app.get("/health")
async def health():
    """Health check."""
    return {
        "status": "ok",
        "device": str(DEVICE),
        "mps_available": torch.backends.mps.is_available(),
        "cached_images": len(_svd_cache),
    }


@app.post("/api/decompose")
async def decompose(
    file: UploadFile = File(...),
    n_components: int = Query(50, ge=1, le=2000),
):
    """
    Full SVD decomposition of an uploaded image.

    Returns:
    - Singular values per channel
    - Cumulative variance per channel
    - Thresholds (k for 90%, 95%, 99%)
    - Reconstructed image at given n_components (base64 PNG)
    - Original image (base64 PNG)
    - Quality metrics (PSNR, SSIM)
    - Timing info
    """
    t_start = time.time()

    # Load image
    file_bytes = await file.read()
    img_array = load_image(file_bytes)
    h, w = img_array.shape[:2]
    max_components = min(h, w)

    if n_components > max_components:
        n_components = max_components

    # Compute or retrieve cached SVD
    img_id = image_hash(file_bytes)
    t_svd_start = time.time()
    cached = get_or_compute_svd(img_id, img_array)
    t_svd = time.time() - t_svd_start

    channels_data = cached["channels"]

    # Build response
    singular_values = {}
    cumulative_variance = {}
    thresholds = {}

    for name in ["R", "G", "B"]:
        S = channels_data[name]["S"]
        singular_values[name] = S.cpu().numpy().tolist()

        var = S**2
        var_cum = (torch.cumsum(var, 0) / var.sum()).cpu().numpy().tolist()
        cumulative_variance[name] = var_cum

        thresholds[name] = {
            "90": find_k_for_threshold(S, 0.90),
            "95": find_k_for_threshold(S, 0.95),
            "99": find_k_for_threshold(S, 0.99),
        }

    # Reconstruct at requested n_components
    t_recon_start = time.time()
    reconstructed = np.zeros_like(img_array)
    for i, name in enumerate(["R", "G", "B"]):
        ch = channels_data[name]
        reconstructed[:, :, i] = reconstruct_channel(
            ch["U"], ch["S"], ch["Vt"], ch["mean"], n_components
        )
    t_recon = time.time() - t_recon_start

    # Quality metrics
    psnr = compute_psnr(img_array, reconstructed)
    ssim_values = {}
    for i, name in enumerate(["R", "G", "B"]):
        ssim_values[name] = compute_ssim_channel(
            img_array[:, :, i], reconstructed[:, :, i]
        )
    ssim_avg = float(np.mean(list(ssim_values.values())))

    # Encode images
    original_b64 = numpy_to_base64_png(img_array)
    reconstructed_b64 = numpy_to_base64_png(reconstructed)

    # Difference image (amplified for visibility)
    diff = np.abs(img_array - reconstructed)
    diff_amplified = np.clip(diff * 5, 0, 255)  # 5x amplification
    diff_b64 = numpy_to_base64_png(diff_amplified)

    t_total = time.time() - t_start

    return JSONResponse(content={
        "image_id": img_id,
        "shape": {"height": h, "width": w, "channels": 3},
        "max_components": max_components,
        "n_components": n_components,
        "singular_values": singular_values,
        "cumulative_variance": cumulative_variance,
        "thresholds": thresholds,
        "metrics": {
            "psnr_db": round(psnr, 2),
            "ssim": round(ssim_avg, 4),
            "ssim_per_channel": {k: round(v, 4) for k, v in ssim_values.items()},
            "compression_ratio": round(max_components / n_components, 2),
        },
        "images": {
            "original": original_b64,
            "reconstructed": reconstructed_b64,
            "difference": diff_b64,
        },
        "timing": {
            "svd_seconds": round(t_svd, 3),
            "reconstruction_seconds": round(t_recon, 3),
            "total_seconds": round(t_total, 3),
        },
    })


@app.post("/api/reconstruct")
async def reconstruct(
    image_id: str = Query(...),
    n_components: int = Query(50, ge=1, le=2000),
):
    """
    Reconstruct a previously decomposed image at a different n_components.
    Uses cached SVD — much faster than /api/decompose.
    Used by the HTMX slider.
    """
    if image_id not in _svd_cache:
        raise HTTPException(status_code=404, detail="Image not in cache. Upload again via /api/decompose.")

    t_start = time.time()
    cached = _svd_cache[image_id]
    img_array = cached["img_array"]
    channels_data = cached["channels"]

    h, w = img_array.shape[:2]
    max_components = min(h, w)
    if n_components > max_components:
        n_components = max_components

    # Reconstruct
    reconstructed = np.zeros_like(img_array)
    for i, name in enumerate(["R", "G", "B"]):
        ch = channels_data[name]
        reconstructed[:, :, i] = reconstruct_channel(
            ch["U"], ch["S"], ch["Vt"], ch["mean"], n_components
        )

    # Metrics
    psnr = compute_psnr(img_array, reconstructed)
    ssim_values = {}
    for i, name in enumerate(["R", "G", "B"]):
        ssim_values[name] = compute_ssim_channel(
            img_array[:, :, i], reconstructed[:, :, i]
        )
    ssim_avg = float(np.mean(list(ssim_values.values())))

    # Encode
    reconstructed_b64 = numpy_to_base64_png(reconstructed)
    diff = np.abs(img_array - reconstructed)
    diff_amplified = np.clip(diff * 5, 0, 255)
    diff_b64 = numpy_to_base64_png(diff_amplified)

    t_total = time.time() - t_start

    return JSONResponse(content={
        "image_id": image_id,
        "n_components": n_components,
        "max_components": max_components,
        "metrics": {
            "psnr_db": round(psnr, 2),
            "ssim": round(ssim_avg, 4),
            "ssim_per_channel": {k: round(v, 4) for k, v in ssim_values.items()},
            "compression_ratio": round(max_components / n_components, 2),
        },
        "images": {
            "reconstructed": reconstructed_b64,
            "difference": diff_b64,
        },
        "timing": {
            "total_seconds": round(t_total, 3),
        },
    })


@app.post("/api/components_visual")
async def components_visual(
    image_id: str = Query(...),
    channel: str = Query("R", regex="^[RGB]$"),
    top_n: int = Query(5, ge=1, le=20),
):
    """
    Return visualization of individual SVD components.
    Each component is the outer product U[:,k] * S[k] * Vt[k,:].
    """
    if image_id not in _svd_cache:
        raise HTTPException(status_code=404, detail="Image not in cache.")

    cached = _svd_cache[image_id]
    ch = cached["channels"][channel]
    U, S, Vt = ch["U"], ch["S"], ch["Vt"]

    components = []
    for k in range(min(top_n, S.shape[0])):
        # Single component contribution
        comp = S[k] * U[:, k:k+1] @ Vt[k:k+1, :]
        comp_np = comp.cpu().numpy()

        # Normalize to 0-255 for visualization
        c_min, c_max = comp_np.min(), comp_np.max()
        if c_max - c_min > 0:
            comp_norm = ((comp_np - c_min) / (c_max - c_min) * 255).astype(np.uint8)
        else:
            comp_norm = np.zeros_like(comp_np, dtype=np.uint8)

        # Convert to RGB grayscale for display
        comp_rgb = np.stack([comp_norm] * 3, axis=-1)
        components.append({
            "index": k,
            "singular_value": float(S[k].cpu()),
            "image": numpy_to_base64_png(comp_rgb),
        })

    return JSONResponse(content={
        "image_id": image_id,
        "channel": channel,
        "components": components,
    })
