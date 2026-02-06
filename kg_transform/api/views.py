from rest_framework.views import APIView
from rest_framework.viewsets import ReadOnlyModelViewSet
from rest_framework.response import Response
from rest_framework import status
from django.conf import settings

from drf_spectacular.utils import extend_schema, OpenApiResponse

from core.models import KnowledgeGraph, Transformation
from api.serializers import (
    TransformRequestSerializer, 
    TransformResponseSerializer, 
    KnowledgeGraphSerializer, 
    TransformationSerializer
)
from llm_engine.orchestrator import transform_kg
from llm_engine.provider import OllamaProvider
from kg_parser.parser import serialize_to_jsonld

from asgiref.sync import async_to_sync

class TransformView(APIView):
    """
    Endpoint to transform a Knowledge Graph using natural language prompt.
    """
    
    @extend_schema(
        request=TransformRequestSerializer,
        responses={200: TransformResponseSerializer}
    )
    def post(self, request):
        serializer = TransformRequestSerializer(data=request.data)
        if not serializer.is_valid():
            return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)
            
        validated_data = serializer.validated_data
        prompt = validated_data['prompt']
        kg_data = validated_data['kg_data']
        model = validated_data.get('model')
        
        # Call Orchestrator
        result = async_to_sync(transform_kg)(prompt, kg_data, model=model)
        
        if result.error:
            # We return 500 or 422 depending on error nature, but 422 Unprocessable Entity seems fit for LLM failure
            return Response(
                {"error": result.error, "metadata": result.metadata}, 
                status=status.HTTP_422_UNPROCESSABLE_ENTITY
            )
            
        # Construct Response
        response_data = {
            "transformation_id": result.metadata.get("transformation_id"),
            "kg_output": serialize_to_jsonld(result.kg_output),
            "diff": result.diff,
            "metadata": result.metadata
        }
        
        # We need to serialize the Diff properly as list of dicts or objects
        # The serializer expects objects or dicts. AtomicOp is a dataclass.
        # DRF serializer can handle dataclasses if we convert them or if they behave object-like.
        # But let's verify if TransformationSerializer handles AtomicOp correctly or if we need conversion.
        # Actually AtomicOpSerializer uses the model AtomicOperation. 
        # But 'result.diff' contains AtomicOp dataclasses from diff.py, not model instances.
        
        # We should map 'diff' to a format compatible with AtomicOperationSerializer or just return raw list of dicts.
        # Let's manually convert diff dataclasses to list of dicts matching response structure
        diff_list = [
            {
                "op_type": op.op_type.value,
                "target_id": op.target_id,
                "payload": op.payload
            } 
            for op in result.diff
        ]
        
        response_data["diff"] = diff_list

        return Response(response_data, status=status.HTTP_200_OK)

class KnowledgeGraphViewSet(ReadOnlyModelViewSet):
    queryset = KnowledgeGraph.objects.all().order_by('-created_at')
    serializer_class = KnowledgeGraphSerializer

class TransformationViewSet(ReadOnlyModelViewSet):
    queryset = Transformation.objects.all().order_by('-created_at')
    serializer_class = TransformationSerializer

class OllamaStatusView(APIView):
    """
    Checks the status of the configured Ollama instance.
    """
    @extend_schema(
        responses={200: OpenApiResponse(description="Ollama status and models")}
    )
    def get(self, request):
        provider = OllamaProvider()
        is_healthy = provider.health_check()
        models = []
        if is_healthy:
            models = provider.list_models()
            
        return Response({
            "healthy": is_healthy,
            "base_url": settings.OLLAMA_BASE_URL,
            "models": models
        })
