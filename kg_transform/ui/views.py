from django.shortcuts import render
from django.http import HttpResponse
from django.views.decorators.http import require_POST
from django.utils.decorators import method_decorator # Not needed for func views
from asgiref.sync import async_to_sync
import json
import logging

from llm_engine.orchestrator import transform_kg
from kg_parser.parser import serialize_to_jsonld

logger = logging.getLogger(__name__)

def index(request):
    """
    Renders the main UI page.
    """
    return render(request, "ui/index.html")

@require_POST
def transform_action(request):
    """
    Handles HTMX POST request to transform KG.
    """
    prompt = request.POST.get("prompt")
    kg_data_raw = request.POST.get("kg_data_raw")
    model = request.POST.get("model")
    if not model: 
        model = None
    

    
    logger.info(f"Transform Request - Prompt: {bool(prompt)}, KG Data: {bool(kg_data_raw)}, Model: {model}")
    
    if not prompt:
        return HttpResponse("<div class='bg-red-500 text-white p-4 rounded'>Error: Missing system prompt</div>", status=400)
        
    if not kg_data_raw:
        return HttpResponse("<div class='bg-red-500 text-white p-4 rounded'>Error: Missing Knowledge Graph data</div>", status=400)

    try:
        kg_data = json.loads(kg_data_raw)
    except json.JSONDecodeError as e:
        logger.error(f"JSON Decode Error: {e}")
        return HttpResponse(f"<div class='bg-red-500 text-white p-4 rounded'>Invalid JSON input: {e}</div>", status=400)

    # Call Orchestrator (Async wrapped in Sync)
    try:
        result = async_to_sync(transform_kg)(prompt, kg_data, model=model)
    except Exception as e:
        logger.exception("Orchestrator failed")
        return HttpResponse(f"<div class='bg-red-500 text-white p-4 rounded'>Orchestrator Error: {e}</div>", status=500)

    if result.error:
        # Render error state
        return HttpResponse(f"""
            <div class='bg-red-900/50 border border-red-500 text-red-100 p-4 rounded'>
                <h3 class='font-bold'><i class='fa-solid fa-triangle-exclamation'></i> Transformation Failed</h3>
                <p>{result.error}</p>
                <div class='text-xs mt-2 opacity-75'>See logs for details over {result.metadata.get('retries', 0)} retries.</div>
            </div>
        """)

    # Success - Render Partial
    context = {
        "kg_output_json": json.dumps(serialize_to_jsonld(result.kg_output)), # For JS
        "diff": result.diff,
        "metadata": result.metadata,
        "transformation_id": result.metadata.get("transformation_id")
    }
    
    return render(request, "ui/partials/result.html", context)

def history_list(request):
    """
    Lists past transformations.
    """
    from core.models import Transformation
    transformations = Transformation.objects.all().order_by('-created_at')
    return render(request, "ui/history.html", {"transformations": transformations})

def history_detail(request, pk):
    """
    Shows detail of a specific transformation.
    """
    from core.models import Transformation
    from django.shortcuts import get_object_or_404
    
    transformation = get_object_or_404(Transformation, pk=pk)
    
    # Reconstruct Result Context
    diff_ops = transformation.operations.all()
    # We need to format diff_ops to match what result.html expects (AtomicOp objects or dicts)
    # The template expects dicts or objects with op_type, target_id, payload.
    # Our AtomicOperation model has these fields.
    
    # However, AtomicOperation.op_type is a string in DB, template checks string equality so that's fine.
    
    context = {
        "transformation": transformation,
        "kg_output_json": json.dumps(transformation.output_kg.data),
        "diff": diff_ops,
        "metadata": {
            "model": transformation.model_used,
            # We don't store token counts in Transformation model currently, only in orchestrator result metadata.
            # If we wanted them, we should have added fields to Transformation model.
            # For now, we leave them blank or generic.
        },
        "transformation_id": str(transformation.id)
    }
    return render(request, "ui/detail.html", context)

def status_partial(request):
    """
    Returns HTML partial for Ollama status.
    """
    from llm_engine.provider import OllamaProvider
    from django.conf import settings
    
    provider = OllamaProvider()
    is_healthy = provider.health_check()
    models = []
    if is_healthy:
        models = provider.list_models()
        
    context = {
        "is_healthy": is_healthy,
        "models": models,
        "base_url": settings.OLLAMA_BASE_URL
    }
    return render(request, "ui/partials/status.html", context)
