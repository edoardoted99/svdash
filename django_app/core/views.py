import csv
import io
import json

from django.shortcuts import render, redirect, get_object_or_404
from django.http import HttpResponse, JsonResponse
from django.contrib import messages

from .models import ImageAnalysis
from .forms import ImageUploadForm
from .services import (
    decompose_image,
    reconstruct_image,
    get_components_visual,
    get_svd_matrices_visual,
    get_multi_reconstruct,
    health_check,
    save_result_image,
    get_result_image_url,
    ComputeServerError,
)


def index(request):
    """Home page with upload form."""
    form = ImageUploadForm()
    recent = ImageAnalysis.objects.filter(results_json__isnull=False)[:5]

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

    is_ajax = request.headers.get("X-Requested-With") == "XMLHttpRequest"

    form = ImageUploadForm(request.POST, request.FILES)
    if not form.is_valid():
        if is_ajax:
            return JsonResponse({"error": "Invalid upload."}, status=400)
        messages.error(request, "Invalid upload.")
        return redirect("index")

    grayscale = form.cleaned_data.get("grayscale", False)
    analysis = form.save()

    try:
        analysis.image.open("rb")
        results = decompose_image(analysis.image, n_components=50, grayscale=grayscale)
        analysis.image.close()

        # Store results (without heavy base64 images)
        stored_results = {k: v for k, v in results.items()
                         if k not in ("images", "channel_images")}
        stored_results["image_id"] = results.get("image_id", "")
        stored_results["grayscale"] = results.get("grayscale", False)
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

        thresholds = results.get("thresholds", {})
        # Use first available channel for thresholds
        first_ch = next(iter(thresholds.values()), {})
        analysis.k_90 = first_ch.get("90")
        analysis.k_95 = first_ch.get("95")
        analysis.k_99 = first_ch.get("99")

        analysis.compute_time = results.get("timing", {}).get("total_seconds")
        analysis.save()

        # Save images to disk
        images = results.get("images", {})
        for name, b64 in images.items():
            save_result_image(analysis.pk, name, b64)

        # Save channel split images
        channel_images = results.get("channel_images", {})
        for ch_name, b64 in channel_images.items():
            save_result_image(analysis.pk, f"channel_{ch_name}", b64)

        # Fetch and save U/S/Vt visualizations
        image_id = results.get("image_id", "")
        first_channel = "L" if grayscale else "R"
        if image_id:
            try:
                matrices = get_svd_matrices_visual(image_id, first_channel)
                for name in ["u_heatmap", "s_chart", "vt_heatmap"]:
                    if name in matrices:
                        save_result_image(analysis.pk, name, matrices[name])
            except ComputeServerError:
                pass

        if is_ajax:
            return JsonResponse({"redirect": f"/analysis/{analysis.pk}/"})
        return redirect("analysis", pk=analysis.pk)

    except ComputeServerError as e:
        analysis.delete()
        if is_ajax:
            return JsonResponse({"error": str(e)}, status=500)
        messages.error(request, f"Compute server error: {e}")
        return redirect("index")


def analysis(request, pk):
    """Analysis dashboard for a specific image."""
    obj = get_object_or_404(ImageAnalysis, pk=pk)
    results = obj.results_json or {}

    # Build image URLs from disk
    image_names = [
        "original", "reconstructed", "difference", "error_heatmap",
        "u_heatmap", "s_chart", "vt_heatmap",
        "channel_R", "channel_G", "channel_B",
    ]
    images = {}
    for name in image_names:
        url = get_result_image_url(pk, name)
        if url:
            images[name] = url

    # Compression stats calculation
    h = obj.height or 0
    w = obj.width or 0
    k = obj.n_components or 50
    n_channels = 1 if results.get("grayscale") else 3
    original_bytes = h * w * 3  # always RGB on disk, 8-bit per channel
    # Rank-k storage: (H*k + k + k*W) per channel, 32-bit floats
    rankk_bytes = (h * k + k + k * w) * n_channels * 4
    compression_stats = {
        "original_mb": round(original_bytes / (1024 * 1024), 2),
        "rankk_mb": round(rankk_bytes / (1024 * 1024), 2),
        "savings_pct": round((1 - rankk_bytes / original_bytes) * 100, 1) if original_bytes > 0 else 0,
        "savings_mb": round((original_bytes - rankk_bytes) / (1024 * 1024), 2),
    }

    is_grayscale = results.get("grayscale", False)

    return render(request, "core/analysis.html", {
        "analysis": obj,
        "images": images,
        "results": results,
        "results_json": results,
        "compression_stats": compression_stats,
        "is_grayscale": is_grayscale,
    })


def history(request):
    analyses = ImageAnalysis.objects.filter(results_json__isnull=False)
    return render(request, "core/history.html", {"analyses": analyses})


def compare(request):
    analyses = ImageAnalysis.objects.filter(results_json__isnull=False)
    id_a = request.GET.get("a")
    id_b = request.GET.get("b")
    analysis_a = get_object_or_404(ImageAnalysis, pk=id_a) if id_a else None
    analysis_b = get_object_or_404(ImageAnalysis, pk=id_b) if id_b else None
    return render(request, "core/compare.html", {
        "analyses": analyses,
        "analysis_a": analysis_a,
        "analysis_b": analysis_b,
    })


# ─── CSV Export ──────────────────────────────────────────────────────────────

def export_csv(request, pk):
    """Export singular values and cumulative variance as CSV."""
    obj = get_object_or_404(ImageAnalysis, pk=pk)
    results = obj.results_json or {}

    sv = results.get("singular_values", {})
    cv = results.get("cumulative_variance", {})
    channels = sorted(sv.keys())

    response = HttpResponse(content_type="text/csv")
    response["Content-Disposition"] = f'attachment; filename="svdash_sv_{pk}.csv"'

    writer = csv.writer(response)
    header = ["index"]
    for ch in channels:
        header += [f"S_{ch}", f"cumvar_{ch}"]
    writer.writerow(header)

    max_len = max((len(sv.get(ch, [])) for ch in channels), default=0)
    for i in range(max_len):
        row = [i]
        for ch in channels:
            s_vals = sv.get(ch, [])
            c_vals = cv.get(ch, [])
            row.append(s_vals[i] if i < len(s_vals) else "")
            row.append(c_vals[i] if i < len(c_vals) else "")
        writer.writerow(row)

    return response


# ─── PDF Report ──────────────────────────────────────────────────────────────

def download_pdf(request, pk):
    """Generate and download a PDF report for an analysis."""
    import os
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.units import cm
    from reportlab.lib.colors import HexColor
    from reportlab.platypus import (
        SimpleDocTemplate, Paragraph, Spacer, Image as RLImage,
        Table, TableStyle,
    )
    from reportlab.lib.styles import ParagraphStyle
    from reportlab.lib.enums import TA_CENTER

    obj = get_object_or_404(ImageAnalysis, pk=pk)
    results = obj.results_json or {}

    response = HttpResponse(content_type="application/pdf")
    response["Content-Disposition"] = f'attachment; filename="svdash_report_{pk}.pdf"'

    doc = SimpleDocTemplate(
        response, pagesize=A4,
        leftMargin=2 * cm, rightMargin=2 * cm,
        topMargin=1.5 * cm, bottomMargin=1.5 * cm,
    )

    title_style = ParagraphStyle(
        "Title", fontName="Courier-Bold", fontSize=16,
        textColor=HexColor("#333333"), spaceAfter=6,
    )
    heading_style = ParagraphStyle(
        "Heading", fontName="Courier-Bold", fontSize=11,
        textColor=HexColor("#006644"), spaceBefore=14, spaceAfter=6,
    )
    mono_style = ParagraphStyle(
        "Mono", fontName="Courier", fontSize=8,
        textColor=HexColor("#333333"), leading=11,
    )

    elements = []

    elements.append(Paragraph(f"SVDash Report — {obj}", title_style))
    elements.append(Paragraph(
        f"{obj.width}x{obj.height} px | k={obj.n_components} | "
        f"{'grayscale' if results.get('grayscale') else 'RGB'} | "
        f"computed {obj.created_at.strftime('%Y-%m-%d %H:%M')}",
        mono_style,
    ))
    elements.append(Spacer(1, 0.5 * cm))

    from django.conf import settings as django_settings

    def get_img_path(name):
        p = os.path.join(django_settings.MEDIA_ROOT, "results", str(pk), f"{name}.png")
        return p if os.path.exists(p) else None

    def add_image(path, max_w=16, max_h=10):
        img = RLImage(path)
        img_w = min(max_w * cm, img.imageWidth * 0.5)
        img_h = img_w * img.imageHeight / img.imageWidth
        if img_h > max_h * cm:
            img_h = max_h * cm
            img_w = img_h * img.imageWidth / img.imageHeight
        img.drawWidth = img_w
        img.drawHeight = img_h
        elements.append(img)

    orig_path = get_img_path("original")
    if orig_path:
        elements.append(Paragraph("ORIGINAL IMAGE", heading_style))
        add_image(orig_path)
        elements.append(Spacer(1, 0.3 * cm))

    # Metrics table
    elements.append(Paragraph("METRICS", heading_style))
    metrics = results.get("metrics", {})
    table_data = [
        ["Metric", "Value"],
        ["PSNR", f"{obj.psnr} dB"],
        ["SSIM", f"{obj.ssim}"],
        ["Compression ratio", f"{metrics.get('compression_ratio', 'N/A')}x"],
        ["k (90% variance)", str(obj.k_90)],
        ["k (95% variance)", str(obj.k_95)],
        ["k (99% variance)", str(obj.k_99)],
        ["Components used", str(obj.n_components)],
        ["Max components", str(obj.max_components)],
        ["Compute time", f"{obj.compute_time}s"],
    ]
    t = Table(table_data, colWidths=[7 * cm, 7 * cm])
    t.setStyle(TableStyle([
        ("FONTNAME", (0, 0), (-1, -1), "Courier"),
        ("FONTSIZE", (0, 0), (-1, -1), 8),
        ("FONTNAME", (0, 0), (-1, 0), "Courier-Bold"),
        ("BACKGROUND", (0, 0), (-1, 0), HexColor("#e0e0e0")),
        ("GRID", (0, 0), (-1, -1), 0.5, HexColor("#cccccc")),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("TOPPADDING", (0, 0), (-1, -1), 3),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
    ]))
    elements.append(t)
    elements.append(Spacer(1, 0.3 * cm))

    for name, label in [
        ("u_heatmap", "U MATRIX HEATMAP"),
        ("s_chart", "SINGULAR VALUES"),
        ("vt_heatmap", "Vt MATRIX HEATMAP"),
    ]:
        path = get_img_path(name)
        if path:
            elements.append(Paragraph(label, heading_style))
            img = RLImage(path)
            img.drawWidth = 15 * cm
            img.drawHeight = 15 * cm * img.imageHeight / img.imageWidth
            elements.append(img)
            elements.append(Spacer(1, 0.2 * cm))

    variance_buf = _generate_variance_chart(results)
    if variance_buf:
        elements.append(Paragraph("CUMULATIVE VARIANCE", heading_style))
        img = RLImage(variance_buf)
        img.drawWidth = 15 * cm
        img.drawHeight = 7 * cm
        elements.append(img)
        elements.append(Spacer(1, 0.2 * cm))

    for name, label in [
        ("reconstructed", f"RECONSTRUCTED (k={obj.n_components})"),
        ("difference", "DIFFERENCE MAP (5x amplified)"),
        ("error_heatmap", "ERROR HEATMAP (MSE per pixel)"),
    ]:
        path = get_img_path(name)
        if path:
            elements.append(Paragraph(label, heading_style))
            add_image(path, max_w=15)
            elements.append(Spacer(1, 0.2 * cm))

    elements.append(Spacer(1, 1 * cm))
    footer_style = ParagraphStyle(
        "Footer", fontName="Courier", fontSize=7,
        textColor=HexColor("#999999"), alignment=TA_CENTER,
    )
    elements.append(Paragraph(
        "Generated by SVDash — SVD/PCA Image Decomposition Dashboard", footer_style,
    ))

    doc.build(elements)
    return response


def _generate_variance_chart(results):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    cum_var = results.get("cumulative_variance", {})
    if not cum_var:
        return None

    fig, ax = plt.subplots(figsize=(8, 3.5))
    colors = {"R": "#ff4444", "G": "#44cc44", "B": "#4488ff", "L": "#cccccc"}
    for ch_name, data in cum_var.items():
        ax.plot(data, color=colors.get(ch_name, "#888"), linewidth=1.2,
                label=ch_name, alpha=0.8)

    for thresh, ls in [(0.90, ":"), (0.95, "--"), (0.99, "-.")]:
        ax.axhline(y=thresh, color="#888", linestyle=ls, alpha=0.4, linewidth=0.8)
        ax.text(len(next(iter(cum_var.values()))) * 0.02, thresh + 0.005,
                f"{int(thresh * 100)}%", fontsize=7, color="#888")

    ax.set_xlabel("Component index k", fontsize=9, fontfamily="monospace")
    ax.set_ylabel("Cumulative variance", fontsize=9, fontfamily="monospace")
    ax.set_title("Cumulative Variance by Channel", fontsize=10, fontfamily="monospace")
    ax.grid(True, alpha=0.3, linewidth=0.5)
    ax.legend(fontsize=8)
    ax.set_ylim(0, 1.05)
    fig.tight_layout()

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return buf


# ─── HTMX Endpoints ─────────────────────────────────────────────────────────

def htmx_reconstruct(request, pk):
    obj = get_object_or_404(ImageAnalysis, pk=pk)
    n = int(request.GET.get("n_components", 50))
    image_id = (obj.results_json or {}).get("image_id", "")
    if not image_id:
        return HttpResponse("<p class='error'>No cached image on compute server.</p>")
    try:
        result = reconstruct_image(image_id, n)
        return render(request, "core/partials/reconstruction.html", {
            "result": result, "n_components": n,
        })
    except ComputeServerError as e:
        return HttpResponse(f"<p class='error'>Error: {e}</p>")


def htmx_components(request, pk):
    obj = get_object_or_404(ImageAnalysis, pk=pk)
    channel = request.GET.get("channel", "R")
    top_n = int(request.GET.get("top_n", 5))
    image_id = (obj.results_json or {}).get("image_id", "")
    if not image_id:
        return HttpResponse("<p class='error'>No cached image on compute server.</p>")
    try:
        result = get_components_visual(image_id, channel, top_n)
        return render(request, "core/partials/components.html", {
            "result": result, "channel": channel,
        })
    except ComputeServerError as e:
        return HttpResponse(f"<p class='error'>Error: {e}</p>")


def htmx_matrices(request, pk):
    obj = get_object_or_404(ImageAnalysis, pk=pk)
    channel = request.GET.get("channel", "R")
    image_id = (obj.results_json or {}).get("image_id", "")
    if not image_id:
        return HttpResponse("<p class='error'>No cached image on compute server.</p>")
    try:
        result = get_svd_matrices_visual(image_id, channel)
        return render(request, "core/partials/matrices.html", {
            "result": result, "channel": channel,
        })
    except ComputeServerError as e:
        return HttpResponse(f"<p class='error'>Error: {e}</p>")


def htmx_multi_k(request, pk):
    """HTMX endpoint: generate multi-k comparison grid."""
    obj = get_object_or_404(ImageAnalysis, pk=pk)
    image_id = (obj.results_json or {}).get("image_id", "")
    if not image_id:
        return HttpResponse("<p class='error'>No cached image on compute server.</p>")

    max_k = obj.max_components or 200
    k_vals = [k for k in [1, 5, 10, 25, 50, 100, 200] if k <= max_k]
    k_str = ",".join(str(k) for k in k_vals)

    try:
        result = get_multi_reconstruct(image_id, k_str)
        return render(request, "core/partials/multi_k.html", {
            "items": result.get("results", []),
        })
    except ComputeServerError as e:
        return HttpResponse(f"<p class='error'>Error: {e}</p>")
