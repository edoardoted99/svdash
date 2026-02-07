from django.contrib import admin
from .models import ImageAnalysis

@admin.register(ImageAnalysis)
class ImageAnalysisAdmin(admin.ModelAdmin):
    list_display = ["title", "width", "height", "k_95", "psnr", "ssim", "created_at"]
    list_filter = ["created_at"]
    readonly_fields = ["results_json"]