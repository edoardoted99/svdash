import pytest
import json
from unittest.mock import patch, AsyncMock
from rest_framework import status
from django.urls import reverse
from core.models import KnowledgeGraph, Transformation, AtomicOperation
from llm_engine.provider import LLMResponse

@pytest.mark.django_db
class TestE2EAPI:
    
    @pytest.fixture
    def mock_ollama_response(self):
        return LLMResponse(
            content=json.dumps({
                "@context": "https://schema.org",
                "@graph": [
                    {"@id": "http://example.org/alice", "@type": "Person", "name": "Alice Cooper"},
                    {"@id": "http://example.org/acme", "@type": "Organization", "name": "Acme Corp"},
                    {"@id": "http://example.org/alice", "worksFor": {"@id": "http://example.org/acme"}}
                ]
            }),
            model="test-model",
            prompt_tokens=10,
            completion_tokens=20
        )

    def test_health_check(self, client):
        """
        GET /api/v1/status/
        """
        url = reverse('status')
        with patch('llm_engine.provider.OllamaProvider.health_check', return_value=True), \
             patch('llm_engine.provider.OllamaProvider.list_models', return_value=['gemma3']):
            response = client.get(url)
            assert response.status_code == status.HTTP_200_OK
            assert response.data['healthy'] is True
            assert 'gemma3' in response.data['models']

    @pytest.mark.asyncio  # Mark checking async view logic if needed, but client is sync wrapper
    async def test_transform_flow_async(self, client, mock_ollama_response):
        # Note: Django test client + async views requires special handling in some versions.
        # But 'client' (APIClient) usually wraps call synchronously. 
        # However, since view is `async def post`, better verify async client support or use AsyncClient from django
        # But prompt says "pytest fixtures". pytest-django provides `async_client`.
        pass 

    def test_transform_flow(self, client, mock_ollama_response):
        """
        POST /api/v1/transform/ -> DB -> GET /api/v1/graphs/
        """
        url = reverse('transform')
        
        kg_input = {
            "@context": "https://schema.org",
            "@graph": [
                {"@id": "http://example.org/alice", "@type": "Person", "name": "Alice"},
                {"@id": "http://example.org/acme", "@type": "Organization", "name": "Acme Corp"}
            ]
        }
        
        payload = {
            "prompt": "Update Alice name to Alice Cooper and add relation worksFor Acme",
            "kg_data": kg_input
        }

        # Mock OllamaProvider.call
        # Since orchestrator imports OllamaProvider class, we patch where it is used.
        # Orchestrator uses: provider = OllamaProvider() ... await provider.call()
        
        with patch('llm_engine.orchestrator.OllamaProvider') as MockProviderCls:
            mock_instance = MockProviderCls.return_value
            mock_instance.call = AsyncMock(return_value=mock_ollama_response)
            
            # Use 'client' which handles async views by running loop in sync thread in recent Django?
            # Or failures might occur. Let's try standard client first.
            response = client.post(url, payload, content_type='application/json')
            
            assert response.status_code == status.HTTP_200_OK, f"Response: {response.data}"
            
            data = response.data
            assert "transformation_id" in data
            assert data["transformation_id"] is not None
            assert len(data["diff"]) > 0
            
            # Verify DB persistence
            trans_id = data["transformation_id"]
            assert Transformation.objects.filter(id=trans_id).exists()
            assert AtomicOperation.objects.filter(transformation_id=trans_id).count() == len(data["diff"])
            assert KnowledgeGraph.objects.count() >= 2 # Input + Output

            # Verify GET /api/v1/graphs/{id} (Output KG)
            t_obj = Transformation.objects.get(id=trans_id)
            output_kg_id = t_obj.output_kg.id
            
            url_kg = reverse('knowledgegraph-detail', args=[output_kg_id])
            resp_kg = client.get(url_kg)
            assert resp_kg.status_code == status.HTTP_200_OK
            assert resp_kg.data["id"] == str(output_kg_id)

            # Verify GET /api/v1/transformations/{id}
            url_trans = reverse('transformation-detail', args=[trans_id])
            resp_trans = client.get(url_trans)
            assert resp_trans.status_code == status.HTTP_200_OK
            assert len(resp_trans.data["operations"]) == len(data["diff"])
