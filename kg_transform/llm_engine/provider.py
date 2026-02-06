import asyncio
import httpx
from dataclasses import dataclass
from typing import List, Dict, Any, Optional
from django.conf import settings

@dataclass
class LLMResponse:
    content: str
    model: str
    prompt_tokens: int = 0
    completion_tokens: int = 0

class OllamaProvider:
    """
    Provider for interacting with the Ollama API (OpenAI compatible).
    """
    def __init__(self):
        # Allow override from settings or env, default fallback
        self.base_url = getattr(settings, "OLLAMA_BASE_URL", "http://localhost:11434")
        self.default_model = getattr(settings, "DEFAULT_MODEL", "gemma3")
        self.timeout = getattr(settings, "OLLAMA_TIMEOUT", 120.0)
        # Ensure base URL doesn't end with slash
        if self.base_url.endswith("/"):
            self.base_url = self.base_url[:-1]

    async def call(self, messages: List[Dict[str, str]], model: Optional[str] = None) -> LLMResponse:
        """
        Asynchronously calls the Ollama chat completion API.
        """
        target_model = model or self.default_model
        url = f"{self.base_url}/v1/chat/completions"
        
        payload = {
            "model": target_model,
            "messages": messages,
            "stream": False,
             # Could add temperature, etc from settings
        }

        async with httpx.AsyncClient(timeout=self.timeout) as client:
            try:
                response = await client.post(url, json=payload)
                response.raise_for_status()
                data = response.json()
                
                content = data["choices"][0]["message"]["content"]
                usage = data.get("usage", {})
                
                return LLMResponse(
                    content=content,
                    model=data.get("model", target_model),
                    prompt_tokens=usage.get("prompt_tokens", 0),
                    completion_tokens=usage.get("completion_tokens", 0)
                )
            except httpx.RequestError as e:
                raise RuntimeError(f"Ollama connection error: {e}")
            except httpx.HTTPStatusError as e:
                 raise RuntimeError(f"Ollama API error {e.response.status_code}: {e.response.text}")
            except (KeyError, IndexError) as e:
                raise RuntimeError(f"Unexpected response format from Ollama: {e}")

    def call_sync(self, messages: List[Dict[str, str]], model: Optional[str] = None) -> LLMResponse:
        """
        Synchronous wrapper for the call method.
        """
        return asyncio.run(self.call(messages, model))

    def health_check(self) -> bool:
        """
        Checks if the Ollama instance is reachable.
        """
        url = f"{self.base_url}/api/tags" # Standard Ollama endpoint
        try:
             with httpx.Client(timeout=5.0) as client:
                response = client.get(url)
                return response.status_code == 200
        except Exception:
            return False

    def list_models(self) -> List[str]:
        """
        Returns a list of available model names.
        """
        url = f"{self.base_url}/api/tags"
        try:
            with httpx.Client(timeout=5.0) as client:
                response = client.get(url)
                response.raise_for_status()
                data = response.json()
                return [model["name"] for model in data.get("models", [])]
        except Exception as e:
            print(f"Error listing models: {e}")
            return []
