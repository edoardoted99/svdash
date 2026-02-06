from django.urls import path
from . import views

app_name = "ui"

urlpatterns = [
    path("", views.index, name="index"),
    path("transform-action/", views.transform_action, name="transform_action"),
    path("history/", views.history_list, name="history_list"),
    path("history/<uuid:pk>/", views.history_detail, name="history_detail"),
    path("status/", views.status_partial, name="status"),
]
