from django.urls import path
from . import views

urlpatterns = [
    path("", views.index, name="index"),
    path("upload/", views.upload, name="upload"),
    path("analysis/<int:pk>/", views.analysis, name="analysis"),
    path("analysis/<int:pk>/pdf/", views.download_pdf, name="download_pdf"),
    path("history/", views.history, name="history"),
    path("compare/", views.compare, name="compare"),

    # Exports
    path("analysis/<int:pk>/csv/", views.export_csv, name="export_csv"),

    # HTMX endpoints
    path("htmx/reconstruct/<int:pk>/", views.htmx_reconstruct, name="htmx_reconstruct"),
    path("htmx/components/<int:pk>/", views.htmx_components, name="htmx_components"),
    path("htmx/matrices/<int:pk>/", views.htmx_matrices, name="htmx_matrices"),
    path("htmx/multi-k/<int:pk>/", views.htmx_multi_k, name="htmx_multi_k"),
]
