from django.urls import path, include
from rest_framework.routers import DefaultRouter
from drf_spectacular.views import SpectacularAPIView, SpectacularRedocView, SpectacularSwaggerView
from .views import (
    TransformView,
    KnowledgeGraphViewSet,
    TransformationViewSet,
    OllamaStatusView
)

app_name = 'api'

router = DefaultRouter()
router.register(r'graphs', KnowledgeGraphViewSet)
router.register(r'transformations', TransformationViewSet)

urlpatterns = [
    # Router URLs
    path('', include(router.urls)),
    
    # Custom Endpoints
    path('transform/', TransformView.as_view(), name='transform'),
    path('status/', OllamaStatusView.as_view(), name='status'),
    
    # Documentation
    path('schema/', SpectacularAPIView.as_view(), name='schema'),
    path('docs/', SpectacularSwaggerView.as_view(url_name='schema'), name='swagger-ui'),
]
