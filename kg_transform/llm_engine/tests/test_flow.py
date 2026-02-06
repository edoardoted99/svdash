import pytest
import json
from unittest.mock import AsyncMock, patch
from llm_engine.orchestrator import transform_kg, TransformResult
from llm_engine.provider import LLMResponse

@pytest.fixture
def mock_kg_input():
    return {
        "@context": "https://schema.org",
        "@graph": [
            {"@id": "http://example.org/alice", "@type": "Person", "name": "Alice"}
        ]
    }

@pytest.mark.asyncio
async def test_transform_kg_success(mock_kg_input):
    # Mock Provider
    mock_response = LLMResponse(
        content=json.dumps({
            "@context": "https://schema.org",
            "@graph": [
                {"@id": "http://example.org/alice", "@type": "Person", "name": "Alice Cooper"}
            ]
        }),
        model="test-model"
    )
    
    with patch('llm_engine.orchestrator.OllamaProvider') as MockProvider:
        instance = MockProvider.return_value
        instance.call = AsyncMock(return_value=mock_response)
        
        # Mock Validator (use real one but it uses local file, so it's fine)
        # Or mock it to speed up. Let's rely on real validator if possible or mock it for speed/isolation.
        # Given we have the file, real validator is better integration test.
        # But wait, validate_kg is in the loop.
        
        result = await transform_kg("Change name to Alice Cooper", mock_kg_input)
        
        assert result.error is None
        assert result.kg_output is not None
        
        # Check diff
        # We expect 1 atomic op: UPDATE_PROP or potentially REMOVE/ADD depending on how pyld handled it
        # Actually in this simple case, ids match, so UPDATE_PROP expected.
        assert len(result.diff) > 0
        assert result.diff[0].op_type.name == "UPDATE_PROP"
        assert result.diff[0].payload['value'] == "Alice Cooper"

@pytest.mark.asyncio
async def test_transform_kg_retry(mock_kg_input):
    # First response invalid, second valid
    invalid_response = LLMResponse(content="I cannot do that.", model="test-model")
    valid_response = LLMResponse(
        content=json.dumps({
             "@context": "https://schema.org",
            "@graph": [{"@id": "http://example.org/alice", "@type": "Person", "name": "Alice"}]
        }), 
        model="test-model"
    )

    with patch('llm_engine.orchestrator.OllamaProvider') as MockProvider:
        instance = MockProvider.return_value
        instance.call = AsyncMock(side_effect=[invalid_response, valid_response])
        
        result = await transform_kg("No-op", mock_kg_input)
        
        assert result.error is None
        assert instance.call.call_count == 2 # Called twice
