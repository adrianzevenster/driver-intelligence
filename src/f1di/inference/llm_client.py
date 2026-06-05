from __future__ import annotations

import httpx


class OpenAICompatibleLLMClient:
    """Thin client for vLLM/TGI/Ollama/OpenAI-compatible chat-completions servers."""

    def __init__(self, base_url: str, api_key: str = "local", model: str = "local-model") -> None:
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.model = model

    async def complete_json(self, system: str, user: str, timeout_s: float = 3.0) -> str:
        async with httpx.AsyncClient(timeout=timeout_s) as client:
            res = await client.post(
                f"{self.base_url}/v1/chat/completions",
                headers={"authorization": f"Bearer {self.api_key}"},
                json={
                    "model": self.model,
                    "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
                    "temperature": 0.1,
                    "response_format": {"type": "json_object"},
                },
            )
            res.raise_for_status()
            return res.json()["choices"][0]["message"]["content"]
