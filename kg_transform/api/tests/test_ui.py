import pytest
from django.urls import reverse
from rest_framework import status
from unittest.mock import patch, AsyncMock
from core.models import Transformation, KnowledgeGraph

@pytest.mark.django_db
class TestUI:
    def test_index_view(self, client):
        url = reverse('ui:index')
        response = client.get(url)
        assert response.status_code == 200
        assert 'Semantica KG' in response.content.decode()

    def test_history_list_view(self, client):
        url = reverse('ui:history_list')
        response = client.get(url)
        assert response.status_code == 200
        assert 'Transformation History' in response.content.decode()

    def test_history_detail_view(self, client):
        # Setup data
        in_kg = KnowledgeGraph.objects.create(data={"@id": "in"})
        out_kg = KnowledgeGraph.objects.create(data={"@id": "out"})
        t = Transformation.objects.create(
            prompt="test",
            input_kg=in_kg,
            output_kg=out_kg,
            model_used="test-model"
        )
        
        url = reverse('ui:history_detail', args=[t.id])
        response = client.get(url)
        assert response.status_code == 200
        assert str(t.id) in response.content.decode()

    def test_transform_action_success(self, client):
        url = reverse('ui:transform_action')
        payload = {
            "prompt": "Test Prompt",
            "kg_data_raw": '{"@context": "https://schema.org", "@graph": []}',
            "model": "gemma3"
        }
        
        # Mock orchestrator
        from llm_engine.orchestrator import TransformResult
        from kg_parser.parser import InternalKG
        
        mock_result = TransformResult(
            kg_output=InternalKG(nodes=[], edges=[]),
            diff=[],
            metadata={"transformation_id": "123", "model": "gemma3"}
        )
        
        # We need to patch transform_kg which is imported in views.py
        # But wait, views.py uses async_to_sync(transform_kg). 
        # Patching 'llm_engine.orchestrator.transform_kg' should work if we patch the object itself 
        # OR patch where it is imported.
        
        # However, async_to_sync wraps the function.
        # Let's try patching the function in ui.views if possible or the original.
        
        with patch('ui.views.transform_kg', new=AsyncMock(return_value=mock_result)) as mock_tf:
            # Note: async_to_sync(mock_tf) will be called. 
            # AsyncMock is awaitable, so async_to_sync handles it.
            
            response = client.post(url, payload)
            
            assert response.status_code == 200
            assert 'kg-output' in response.content.decode() # Check for result partial content
