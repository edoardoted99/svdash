from rest_framework import serializers
from core.models import KnowledgeGraph, Transformation, AtomicOperation

class KnowledgeGraphSerializer(serializers.ModelSerializer):
    class Meta:
        model = KnowledgeGraph
        fields = ['id', 'created_at', 'data']

class AtomicOperationSerializer(serializers.ModelSerializer):
    class Meta:
        model = AtomicOperation
        fields = ['op_type', 'target_id', 'payload']

class TransformationSerializer(serializers.ModelSerializer):
    operations = AtomicOperationSerializer(many=True, read_only=True)
    input_kg = KnowledgeGraphSerializer(read_only=True)
    output_kg = KnowledgeGraphSerializer(read_only=True)

    class Meta:
        model = Transformation
        fields = ['id', 'created_at', 'prompt', 'model_used', 'input_kg', 'output_kg', 'operations']

class TransformRequestSerializer(serializers.Serializer):
    prompt = serializers.CharField(required=True)
    kg_data = serializers.DictField(required=True, help_text="The input Knowledge Graph in JSON-LD format")
    model = serializers.CharField(required=False, default=None)

class TransformResponseSerializer(serializers.Serializer):
    transformation_id = serializers.UUIDField()
    kg_output = serializers.DictField()
    diff = AtomicOperationSerializer(many=True)
    metadata = serializers.DictField(required=False)
