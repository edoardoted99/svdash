import json
from django.shortcuts import render, redirect, get_object_or_404
from django.http import HttpResponse
from django.contrib import messages

from .models import ImageAnalysis
from .forms import ImageUploadForm
from .services import (
    decompose_image,
    reconstruct_image,
    get_components_visual,
    health_check,
    ComputeServerError,
)


def index(request):
    """Home page with upload form."""
    form = ImageUploadForm()
    recent = ImageAnalysis.objects.filter(results_json__isnull=False)[:5]

    # Check compute server status
    server_ok = False
    try:
        health_check()
        server_ok = True
    except ComputeServerError:
        pass

    return render(request, "core/index.html", {
        "form": form,
        "recent": recent,
        "server_ok": server_ok,
    })


def upload(request):
    """Handle image upload and trigger decomposition."""
    if request.method != "POST":
        return redirect("index")

    form = ImageUploadForm(request.POST, request.FILES)
    if not form.is_valid():
        messages.error(request, "Invalid upload.")
        return redirect("index")

    analysis = form.save()

    # Send to compute server
    try:
        analysis.image.open("rb")
        results = decompose_image(analysis.image, n_components=50)
        analysis.image.close()

        # Store full results
        # Remove heavy base64 images from stored JSON (we serve them separately)
        stored_results = {k: v for k, v in results.items() if k != "images"}
        stored_results["image_id"] = results.get("image_id", "")
        analysis.results_json = stored_results

        # Extract key metrics
        shape = results.get("shape", {})
        analysis.width = shape.get("width")
        analysis.height = shape.get("height")
        analysis.max_components = results.get("max_components")
        analysis.n_components = results.get("n_components", 50)

        metrics = results.get("metrics", {})
        analysis.psnr = metrics.get("psnr_db")
        analysis.ssim = metrics.get("ssim")

        # Thresholds (average across channels)
        thresholds = results.get("thresholds", {})
        r_thresh = thresholds.get("R", {})
        analysis.k_90 = r_thresh.get("90")
        analysis.k_95 = r_thresh.get("95")
        analysis.k_99 = r_thresh.get("99")

        analysis.compute_time = results.get("timing", {}).get("total_seconds")

        # Store base64 images in session for display
        request.session[f"images_{analysis.pk}"] = results.get("images", {})

        analysis.save()
        return redirect("analysis", pk=analysis.pk)

    except ComputeServerError as e:
        messages.error(request, f"Compute server error: {e}")
        analysis.delete()
        return redirect("index")


def analysis(request, pk):
    """Analysis dashboard for a specific image."""
    obj = get_object_or_404(ImageAnalysis, pk=pk)

    # Get images from session or empty
    images = request.session.get(f"images_{pk}", {})

    return render(request, "core/analysis.html", {
        "analysis": obj,
        "images": images,
        "results": obj.results_json or {},
    })


def history(request):
    """List of all past analyses."""
    analyses = ImageAnalysis.objects.filter(results_json__isnull=False)
    return render(request, "core/history.html", {
        "analyses": analyses,
    })


def compare(request):
    """Compare two analyses side by side."""
    analyses = ImageAnalysis.objects.filter(results_json__isnull=False)
    
    id_a = request.GET.get("a")
    id_b = request.GET.get("b")
    
    analysis_a = None
    analysis_b = None
    
    if id_a:
        analysis_a = get_object_or_404(ImageAnalysis, pk=id_a)
    if id_b:
        analysis_b = get_object_or_404(ImageAnalysis, pk=id_b)

    return render(request, "core/compare.html", {
        "analyses": analyses,
        "analysis_a": analysis_a,
        "analysis_b": analysis_b,
    })


# ─── HTMX Endpoints ──────────────────────────────────────────────────────────

def htmx_reconstruct(request, pk):
    """HTMX endpoint: reconstruct at a new n_components."""
    obj = get_object_or_404(ImageAnalysis, pk=pk)
    n = int(request.GET.get("n_components", 50))

    image_id = obj.results_json.get("image_id", "")
    if not image_id:
        return HttpResponse("<p class='error'>No cached image on compute server.</p>")

    try:
        result = reconstruct_image(image_id, n)
        return render(request, "core/partials/reconstruction.html", {
            "result": result,
            "n_components": n,
        })
    except ComputeServerError as e:
        return HttpResponse(f"<p class='error'>Error: {e}</p>")


def htmx_components(request, pk):
    """HTMX endpoint: get component visualizations."""
    obj = get_object_or_404(ImageAnalysis, pk=pk)
    channel = request.GET.get("channel", "R")
    top_n = int(request.GET.get("top_n", 5))

    image_id = obj.results_json.get("image_id", "")
    if not image_id:
        return HttpResponse("<p class='error'>No cached image on compute server.</p>")

    try:
        result = get_components_visual(image_id, channel, top_n)
        return render(request, "core/partials/components.html", {
            "result": result,
            "channel": channel,
        })
    except ComputeServerError as e:
        return HttpResponse(f"<p class='error'>Error: {e}</p>")