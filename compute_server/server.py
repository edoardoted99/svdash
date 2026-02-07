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

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ─── Setup ────────────────────────────────────────────────────────────────────

app = FastAPI(
    title="SVDash Compute Server",
    description="SVD/PCA image decomposition engine running on MPS",
    version="0.2.0",
)

DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"[SVDash Compute] Device: {DEVICE}")

# Dark scientific matplotlib style (shared)
DARK_STYLE = {
    'figure.facecolor': '#0a0e14',
    'axes.facecolor': '#141c28',
    'axes.edgecolor': '#2a3a50',
    'text.color': '#d4dce8',
    'axes.labelcolor': '#d4dce8',
    'xtick.color': '#7a8a9e',
    'ytick.color': '#7a8a9e',
    'grid.color': '#1e2a3a',
    'font.family': 'monospace',
    'font.size': 10,
}


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


def fig_to_b64(fig) -> str:
    """Render matplotlib figure to base64 PNG."""
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=120, bbox_inches='tight')
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.getvalue()).decode()


def make_error_heatmap(original: np.ndarray, reconstructed: np.ndarray) -> str:
    """Generate error heatmap (MSE per pixel) as base64 PNG with matplotlib."""
    if original.ndim == 3:
        mse = np.mean((original - reconstructed) ** 2, axis=2)
    else:
        mse = (original - reconstructed) ** 2

    with plt.rc_context(DARK_STYLE):
        fig, ax = plt.subplots(figsize=(7, 5))
        im = ax.imshow(mse, cmap='inferno', interpolation='nearest', aspect='auto')
        ax.set_title('Reconstruction Error (MSE per pixel)', fontsize=11, pad=8)
        cb = plt.colorbar(im, ax=ax, shrink=0.8, pad=0.02)
        cb.set_label('MSE', fontsize=9)
        cb.ax.tick_params(colors='#7a8a9e', labelsize=8)
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        fig.tight_layout()
    return fig_to_b64(fig)


def reconstruct_full(channels_data, img_array, n_components, grayscale=False):
    """Reconstruct full image from cached SVD. Returns (reconstructed, channel_names)."""
    if grayscale:
        ch = channels_data["L"]
        gray_recon = reconstruct_channel(ch["U"], ch["S"], ch["Vt"], ch["mean"], n_components)
        reconstructed = np.stack([gray_recon] * 3, axis=-1)
        return reconstructed, ["L"]
    else:
        reconstructed = np.zeros_like(img_array)
        for i, name in enumerate(["R", "G", "B"]):
            ch = channels_data[name]
            reconstructed[:, :, i] = reconstruct_channel(
                ch["U"], ch["S"], ch["Vt"], ch["mean"], n_components
            )
        return reconstructed, ["R", "G", "B"]


def compute_metrics(img_array, reconstructed, channel_names, grayscale=False):
    """Compute PSNR and SSIM metrics."""
    if grayscale:
        gray_orig = 0.299 * img_array[:, :, 0] + 0.587 * img_array[:, :, 1] + 0.114 * img_array[:, :, 2]
        gray_recon = reconstructed[:, :, 0].astype(np.float64)
        psnr = compute_psnr(gray_orig, gray_recon)
        ssim_avg = compute_ssim_channel(gray_orig, gray_recon)
        ssim_per_channel = {"L": round(ssim_avg, 4)}
    else:
        psnr = compute_psnr(img_array, reconstructed)
        ssim_values = {}
        for i, name in enumerate(channel_names):
            ssim_values[name] = compute_ssim_channel(
                img_array[:, :, i], reconstructed[:, :, i]
            )
        ssim_avg = float(np.mean(list(ssim_values.values())))
        ssim_per_channel = {k: round(v, 4) for k, v in ssim_values.items()}
    return round(psnr, 2), round(ssim_avg, 4), ssim_per_channel


# ─── Cache for SVD results (per image hash) ──────────────────────────────────
# Simple in-memory cache to avoid recomputing SVD when user changes k

_svd_cache: dict = {}


def get_or_compute_svd(img_hash: str, img_array: np.ndarray, grayscale: bool = False):
    """Get cached SVD or compute it."""
    cache_key = img_hash + ("_gray" if grayscale else "")
    if cache_key in _svd_cache:
        return _svd_cache[cache_key], cache_key

    channels = {}
    if grayscale:
        gray = 0.299 * img_array[:, :, 0] + 0.587 * img_array[:, :, 1] + 0.114 * img_array[:, :, 2]
        U, S, Vt, mean = svd_channel(gray, DEVICE)
        channels["L"] = {"U": U, "S": S, "Vt": Vt, "mean": mean}
    else:
        for i, name in enumerate(["R", "G", "B"]):
            U, S, Vt, mean = svd_channel(img_array[:, :, i], DEVICE)
            channels[name] = {"U": U, "S": S, "Vt": Vt, "mean": mean}

    _svd_cache[cache_key] = {
        "channels": channels,
        "img_array": img_array,
        "grayscale": grayscale,
    }

    # Keep cache small — max 10 entries
    if len(_svd_cache) > 10:
        oldest = next(iter(_svd_cache))
        del _svd_cache[oldest]

    return _svd_cache[cache_key], cache_key


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
    grayscale: bool = Query(False),
):
    """
    Full SVD decomposition of an uploaded image.
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
    img_hash_val = image_hash(file_bytes)
    t_svd_start = time.time()
    cached, cache_key = get_or_compute_svd(img_hash_val, img_array, grayscale)
    t_svd = time.time() - t_svd_start

    channels_data = cached["channels"]
    channel_names = list(channels_data.keys())

    # Build response data per channel
    singular_values = {}
    cumulative_variance = {}
    thresholds = {}

    for name in channel_names:
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
    reconstructed, _ = reconstruct_full(channels_data, img_array, n_components, grayscale)
    t_recon = time.time() - t_recon_start

    # Quality metrics
    psnr, ssim_avg, ssim_per_channel = compute_metrics(
        img_array, reconstructed, channel_names, grayscale
    )

    # Encode images
    original_b64 = numpy_to_base64_png(img_array)
    reconstructed_b64 = numpy_to_base64_png(reconstructed)

    # Difference image (amplified for visibility)
    diff = np.abs(img_array.astype(np.float64) - reconstructed.astype(np.float64))
    diff_amplified = np.clip(diff * 5, 0, 255)
    diff_b64 = numpy_to_base64_png(diff_amplified)

    # Error heatmap (MSE per pixel with colorbar)
    error_heatmap_b64 = make_error_heatmap(img_array, reconstructed)

    # Channel split images (R, G, B as grayscale)
    channel_images = {}
    for i, name in enumerate(["R", "G", "B"]):
        ch_data = img_array[:, :, i]
        ch_rgb = np.stack([ch_data] * 3, axis=-1)
        channel_images[name] = numpy_to_base64_png(ch_rgb)

    t_total = time.time() - t_start

    return JSONResponse(content={
        "image_id": cache_key,
        "grayscale": grayscale,
        "shape": {"height": h, "width": w, "channels": 1 if grayscale else 3},
        "max_components": max_components,
        "n_components": n_components,
        "singular_values": singular_values,
        "cumulative_variance": cumulative_variance,
        "thresholds": thresholds,
        "metrics": {
            "psnr_db": psnr,
            "ssim": ssim_avg,
            "ssim_per_channel": ssim_per_channel,
            "compression_ratio": round(max_components / n_components, 2),
        },
        "images": {
            "original": original_b64,
            "reconstructed": reconstructed_b64,
            "difference": diff_b64,
            "error_heatmap": error_heatmap_b64,
        },
        "channel_images": channel_images,
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
    is_gray = cached.get("grayscale", False)

    h, w = img_array.shape[:2]
    max_components = min(h, w)
    if n_components > max_components:
        n_components = max_components

    # Reconstruct
    channel_names = list(channels_data.keys())
    reconstructed, _ = reconstruct_full(channels_data, img_array, n_components, is_gray)

    # Metrics
    psnr, ssim_avg, ssim_per_channel = compute_metrics(
        img_array, reconstructed, channel_names, is_gray
    )

    # Encode
    reconstructed_b64 = numpy_to_base64_png(reconstructed)
    diff = np.abs(img_array.astype(np.float64) - reconstructed.astype(np.float64))
    diff_amplified = np.clip(diff * 5, 0, 255)
    diff_b64 = numpy_to_base64_png(diff_amplified)

    # Error heatmap
    error_heatmap_b64 = make_error_heatmap(img_array, reconstructed)

    t_total = time.time() - t_start

    return JSONResponse(content={
        "image_id": image_id,
        "n_components": n_components,
        "max_components": max_components,
        "metrics": {
            "psnr_db": psnr,
            "ssim": ssim_avg,
            "ssim_per_channel": ssim_per_channel,
            "compression_ratio": round(max_components / n_components, 2),
        },
        "images": {
            "reconstructed": reconstructed_b64,
            "difference": diff_b64,
            "error_heatmap": error_heatmap_b64,
        },
        "timing": {
            "total_seconds": round(t_total, 3),
        },
    })


@app.post("/api/components_visual")
async def components_visual(
    image_id: str = Query(...),
    channel: str = Query("R", regex="^[RGBL]$"),
    top_n: int = Query(5, ge=1, le=20),
):
    """
    Return visualization of individual SVD components.
    Each component is the outer product U[:,k] * S[k] * Vt[k,:].
    """
    if image_id not in _svd_cache:
        raise HTTPException(status_code=404, detail="Image not in cache.")

    cached = _svd_cache[image_id]
    if channel not in cached["channels"]:
        available = list(cached["channels"].keys())
        raise HTTPException(status_code=400, detail=f"Channel '{channel}' not available. Use: {available}")

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


@app.post("/api/svd_matrices_visual")
async def svd_matrices_visual(
    image_id: str = Query(...),
    channel: str = Query("R", regex="^[RGBL]$"),
):
    """
    Visualize SVD matrices as images:
    - U heatmap (first 50 columns)
    - S singular values (log scale line chart)
    - Vt heatmap (first 50 rows)
    """
    if image_id not in _svd_cache:
        raise HTTPException(status_code=404, detail="Image not in cache.")

    cached = _svd_cache[image_id]
    if channel not in cached["channels"]:
        available = list(cached["channels"].keys())
        raise HTTPException(status_code=400, detail=f"Channel '{channel}' not available. Use: {available}")

    ch = cached["channels"][channel]
    U, S, Vt = ch["U"], ch["S"], ch["Vt"]

    n_show = min(50, S.shape[0])

    # ── U heatmap ──
    with plt.rc_context(DARK_STYLE):
        fig, ax = plt.subplots(figsize=(8, 5))
        u_data = U[:, :n_show].cpu().numpy()
        im = ax.imshow(u_data, aspect='auto', cmap='inferno', interpolation='nearest')
        ax.set_title(f'U  —  first {n_show} columns  (ch: {channel})', fontsize=11, pad=8)
        ax.set_xlabel('component index k')
        ax.set_ylabel('row index i')
        cb = plt.colorbar(im, ax=ax, shrink=0.8, pad=0.02)
        cb.ax.tick_params(colors='#7a8a9e', labelsize=8)
        fig.tight_layout()
    u_b64 = fig_to_b64(fig)

    # ── S chart (log scale) ──
    with plt.rc_context(DARK_STYLE):
        fig, ax = plt.subplots(figsize=(8, 4))
        s_data = S.cpu().numpy()
        ax.semilogy(s_data, color='#00d4aa', linewidth=1.2, alpha=0.9)
        ax.fill_between(range(len(s_data)), s_data, alpha=0.08, color='#00d4aa')
        ax.axvline(x=n_show, color='#ff5555', linestyle='--', alpha=0.7,
                   linewidth=1, label=f'k={n_show}')
        ax.set_title(f'Singular values  (ch: {channel})', fontsize=11, pad=8)
        ax.set_xlabel('component index k')
        ax.set_ylabel('singular value (log)')
        ax.grid(True, alpha=0.2, linewidth=0.5)
        ax.legend(fontsize=9, facecolor='#141c28', edgecolor='#2a3a50',
                  labelcolor='#d4dce8')
        fig.tight_layout()
    s_b64 = fig_to_b64(fig)

    # ── Vt heatmap ──
    with plt.rc_context(DARK_STYLE):
        fig, ax = plt.subplots(figsize=(8, 5))
        vt_data = Vt[:n_show, :].cpu().numpy()
        im = ax.imshow(vt_data, aspect='auto', cmap='inferno', interpolation='nearest')
        ax.set_title(f'V\u1d40  —  first {n_show} rows  (ch: {channel})', fontsize=11, pad=8)
        ax.set_xlabel('column index j')
        ax.set_ylabel('component index k')
        cb = plt.colorbar(im, ax=ax, shrink=0.8, pad=0.02)
        cb.ax.tick_params(colors='#7a8a9e', labelsize=8)
        fig.tight_layout()
    vt_b64 = fig_to_b64(fig)

    return JSONResponse(content={
        "image_id": image_id,
        "channel": channel,
        "u_heatmap": u_b64,
        "s_chart": s_b64,
        "vt_heatmap": vt_b64,
    })


@app.post("/api/multi_reconstruct")
async def multi_reconstruct(
    image_id: str = Query(...),
    k_values: str = Query("1,5,10,25,50,100,200"),
):
    """
    Reconstruct at multiple k values in one call.
    Returns array of {k, image, psnr, ssim} for each value.
    """
    if image_id not in _svd_cache:
        raise HTTPException(status_code=404, detail="Image not in cache.")

    cached = _svd_cache[image_id]
    img_array = cached["img_array"]
    channels_data = cached["channels"]
    is_gray = cached.get("grayscale", False)

    h, w = img_array.shape[:2]
    max_k = min(h, w)

    k_list = []
    for s in k_values.split(","):
        s = s.strip()
        if s.isdigit():
            k = min(int(s), max_k)
            if k >= 1:
                k_list.append(k)

    results = []
    for k in k_list:
        reconstructed, ch_names = reconstruct_full(channels_data, img_array, k, is_gray)
        psnr, ssim_avg, _ = compute_metrics(img_array, reconstructed, ch_names, is_gray)
        img_b64 = numpy_to_base64_png(reconstructed)
        results.append({
            "k": k,
            "image": img_b64,
            "psnr_db": psnr,
            "ssim": ssim_avg,
        })

    return JSONResponse(content={
        "image_id": image_id,
        "results": results,
    })
