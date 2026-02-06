"""Ollama HTTP client using direct API calls."""

import httpx
from django.conf import settings
from typing import AsyncIterator


class OllamaClient:
    """Client for Ollama API via direct HTTP calls."""

    def __init__(self, base_url: str | None = None, model: str | None = None):
        self.base_url = (base_url or settings.OLLAMA_BASE_URL).rstrip("/")
        self.model = model or settings.DEFAULT_MODEL

    def generate(
        self,
        prompt: str,
        model: str | None = None,
        system: str | None = None,
        stream: bool = False,
        **options,
    ) -> dict:
        """Generate a completion (synchronous)."""
        payload = {
            "model": model or self.model,
            "prompt": prompt,
            "stream": stream,
        }
        if system:
            payload["system"] = system
        if options:
            payload["options"] = options

        with httpx.Client(timeout=120.0) as client:
            response = client.post(f"{self.base_url}/api/generate", json=payload)
            response.raise_for_status()
            return response.json()

    async def agenerate(
        self,
        prompt: str,
        model: str | None = None,
        system: str | None = None,
        stream: bool = False,
        **options,
    ) -> dict:
        """Generate a completion (async)."""
        payload = {
            "model": model or self.model,
            "prompt": prompt,
            "stream": stream,
        }
        if system:
            payload["system"] = system
        if options:
            payload["options"] = options

        async with httpx.AsyncClient(timeout=120.0) as client:
            response = await client.post(f"{self.base_url}/api/generate", json=payload)
            response.raise_for_status()
            return response.json()

    def chat(
        self,
        messages: list[dict],
        model: str | None = None,
        stream: bool = False,
        **options,
    ) -> dict:
        """Chat completion (synchronous)."""
        payload = {
            "model": model or self.model,
            "messages": messages,
            "stream": stream,
        }
        if options:
            payload["options"] = options

        with httpx.Client(timeout=120.0) as client:
            response = client.post(f"{self.base_url}/api/chat", json=payload)
            response.raise_for_status()
            return response.json()

    async def achat(
        self,
        messages: list[dict],
        model: str | None = None,
        stream: bool = False,
        **options,
    ) -> dict:
        """Chat completion (async)."""
        payload = {
            "model": model or self.model,
            "messages": messages,
            "stream": stream,
        }
        if options:
            payload["options"] = options

        async with httpx.AsyncClient(timeout=120.0) as client:
            response = await client.post(f"{self.base_url}/api/chat", json=payload)
            response.raise_for_status()
            return response.json()

    def list_models(self) -> dict:
        """List available models."""
        with httpx.Client(timeout=30.0) as client:
            response = client.get(f"{self.base_url}/api/tags")
            response.raise_for_status()
            return response.json()

    def embeddings(
        self,
        input_text: str | list[str],
        model: str | None = None,
    ) -> dict:
        """Generate embeddings."""
        payload = {
            "model": model or self.model,
            "input": input_text,
        }
        with httpx.Client(timeout=60.0) as client:
            response = client.post(f"{self.base_url}/api/embed", json=payload)
            response.raise_for_status()
            return response.json()

    def is_available(self) -> bool:
        """Check if Ollama server is available."""
        try:
            with httpx.Client(timeout=5.0) as client:
                response = client.get(f"{self.base_url}/api/tags")
                return response.status_code == 200
        except httpx.RequestError:
            return False


# Default client instance
def get_client(base_url: str | None = None, model: str | None = None) -> OllamaClient:
    """Get an Ollama client instance."""
    return OllamaClient(base_url=base_url, model=model)
