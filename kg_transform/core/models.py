import uuid
from django.db import models

class KnowledgeGraph(models.Model):
    """
    Stores the full Knowledge Graph as JSON.
    """
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    created_at = models.DateTimeField(auto_now_add=True)
    data = models.JSONField(help_text="The Knowledge Graph in valid JSON-LD format")

    def __str__(self):
        return f"KG {self.id} ({self.created_at})"

class Transformation(models.Model):
    """
    Represents a transformation event from an input KG to an output KG using an LLM.
    """
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    created_at = models.DateTimeField(auto_now_add=True)
    prompt = models.TextField(help_text="The user instruction for this transformation")
    input_kg = models.ForeignKey(KnowledgeGraph, related_name='transformations_as_input', on_delete=models.CASCADE)
    output_kg = models.ForeignKey(KnowledgeGraph, related_name='transformations_as_output', on_delete=models.CASCADE)
    model_used = models.CharField(max_length=255, help_text="The LLM model used (e.g., gemma3)")

    def __str__(self):
        return f"Transformation {self.id} on {self.created_at}"

class AtomicOperation(models.Model):
    """
    Represents a single atomic change operation (ADD_NODE, REMOVE_EDGE, etc.).
    """
    transformation = models.ForeignKey(Transformation, related_name='operations', on_delete=models.CASCADE)
    op_type = models.CharField(max_length=50, help_text="Type of operation (from diff.OpType)")
    target_id = models.CharField(max_length=1024, help_text="ID of the node or edge being modified")
    payload = models.JSONField(null=True, blank=True, help_text="Additional data for the operation (new values, etc.)")

    def __str__(self):
        return f"{self.op_type} on {self.target_id}"
