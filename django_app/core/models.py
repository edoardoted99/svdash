import json
from django.db import models


class ImageAnalysis(models.Model):
    """Stores an uploaded image and its SVD decomposition results."""

    title = models.CharField(max_length=255, blank=True)
    image = models.ImageField(upload_to="uploads/%Y/%m/")
    
    # Results from compute server (stored as JSON)
    results_json = models.JSONField(null=True, blank=True)
    
    # Key metrics extracted for easy querying
    n_components = models.IntegerField(default=50)
    psnr = models.FloatField(null=True, blank=True)
    ssim = models.FloatField(null=True, blank=True)
    
    # Thresholds
    k_90 = models.IntegerField(null=True, blank=True)
    k_95 = models.IntegerField(null=True, blank=True)
    k_99 = models.IntegerField(null=True, blank=True)
    
    # Image metadata
    width = models.IntegerField(null=True, blank=True)
    height = models.IntegerField(null=True, blank=True)
    max_components = models.IntegerField(null=True, blank=True)
    
    # Timing
    compute_time = models.FloatField(null=True, blank=True)
    
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        ordering = ["-created_at"]
        verbose_name_plural = "Image analyses"

    def __str__(self):
        return self.title or f"Analysis #{self.pk}"

    @property
    def singular_values(self):
        if self.results_json:
            return self.results_json.get("singular_values", {})
        return {}

    @property
    def cumulative_variance(self):
        if self.results_json:
            return self.results_json.get("cumulative_variance", {})
        return {}

    @property
    def thresholds(self):
        return {"90%": self.k_90, "95%": self.k_95, "99%": self.k_99}