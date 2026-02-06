from django.contrib import admin
from django.urls import path, include
from drf_spectacular.views import SpectacularAPIView, SpectacularSwaggerView

urlpatterns = [
    path("admin/", admin.site.urls),
    path("api/v1/", include("api.urls")),
    path("", include("ui.urls")),
    # API documentation exposed at root /api/docs/ implies aggregating or pointing to specific schema
    # But since we moved api to v1, let's expose docs there too via api.urls
    # Or keep global docs here check api/schema
    path("api/docs/", SpectacularSwaggerView.as_view(url_name="schema"), name="swagger-ui"),
    path("api/schema/", SpectacularAPIView.as_view(), name="schema"),
]
