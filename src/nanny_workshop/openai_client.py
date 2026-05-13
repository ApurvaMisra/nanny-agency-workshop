"""OpenAI client with on-disk response caching for workshop reliability."""

import hashlib
import json
from pathlib import Path
from openai import OpenAI


class CachedOpenAI:
    """Wraps openai.OpenAI; caches completions and embeddings on disk by key.

    Disable caching by passing use_cache=False to a call. Cache files are JSON
    keyed by sha256 of the request payload.
    """

    def __init__(self, cache_dir: str | Path, api_key: str | None = None):
        self._cache_dir = Path(cache_dir)
        self._cache_dir.mkdir(parents=True, exist_ok=True)
        self._client = OpenAI(api_key=api_key)

    def _cache_path(self, kind: str, payload: dict) -> Path:
        key = hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()
        return self._cache_dir / f"{kind}_{key}.json"

    def complete(
        self,
        model: str,
        messages: list[dict],
        temperature: float = 0.0,
        response_format: dict | None = None,
        use_cache: bool = True,
    ) -> str:
        payload = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "response_format": response_format,
        }
        cache_file = self._cache_path("chat", payload)
        if use_cache and cache_file.exists():
            return json.loads(cache_file.read_text())["content"]

        kwargs = {"model": model, "messages": messages, "temperature": temperature}
        if response_format is not None:
            kwargs["response_format"] = response_format
        resp = self._client.chat.completions.create(**kwargs)
        content = resp.choices[0].message.content or ""
        cache_file.write_text(json.dumps({"content": content}))
        return content

    def embed(
        self,
        model: str,
        text: str,
        use_cache: bool = True,
    ) -> list[float]:
        payload = {"model": model, "text": text}
        cache_file = self._cache_path("embed", payload)
        if use_cache and cache_file.exists():
            return json.loads(cache_file.read_text())["embedding"]

        resp = self._client.embeddings.create(model=model, input=text)
        vec = resp.data[0].embedding
        cache_file.write_text(json.dumps({"embedding": vec}))
        return vec
