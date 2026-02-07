from django.urls import path
from . import views

urlpatterns = [
    path("", views.index, name="index"),
    path("upload/", views.upload, name="upload"),
    path("analysis/<int:pk>/", views.analysis, name="analysis"),
    path("history/", views.history, name="history"),
    path("compare/", views.compare, name="compare"),
    
    # HTMX endpoints
    path("htmx/reconstruct/<int:pk>/", views.htmx_reconstruct, name="htmx_reconstruct"),
    path("htmx/components/<int:pk>/", views.htmx_components, name="htmx_components"),
]