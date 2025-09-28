# utils/llm_core.py
import os
import time
from typing import Dict, Iterable, List, Optional
from openai import OpenAI

try:
    import tiktoken
except Exception:
    tiktoken = None

_BACKENDS = {}

def register(name):
    def _wrap(cls):
        _BACKENDS[name] = cls
        return cls
    return _wrap

def get_backend(name: str, **kwargs):
    name = (name or "openai").lower()
    if name not in _BACKENDS:
        raise ValueError(f"Unknown backend: {name}")
    return _BACKENDS[name](**kwargs)

def build_messages(history: List[Dict], system_prompt: str = "") -> List[Dict]:
    messages: List[Dict] = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    for item in history:
        role = item.get("role")
        content = item.get("content", "")
        if role in {"user", "assistant", "system"}:
            messages.append({"role": role, "content": content})
    return messages

def _encoding_for_model(model: str):
    if tiktoken is None:
        return None
    try:
        return tiktoken.encoding_for_model(model)
    except Exception:
        return tiktoken.get_encoding("cl100k_base") if tiktoken else None

def estimate_tokens(messages: List[Dict], model: str = "gpt-4o-mini") -> int:
    enc = _encoding_for_model(model)
    if enc is None:
        total_chars = sum(len(str(m.get("content", ""))) for m in messages)
        return max(1, total_chars // 4)
    count = 0
    for m in messages:
        count += len(enc.encode(str(m.get("content", ""))))
    return max(1, count)

@register("openai")
class OpenAIBackend:
    def __init__(self, api_key: Optional[str] = None, max_retries: int = 3, retry_backoff: float = 1.0):
        key = api_key or os.getenv("OPENAI_API_KEY")
        self.max_retries = max_retries
        self.retry_backoff = retry_backoff
        if not key:
            self.client = None
            self._missing_key = True
        else:
            self.client = OpenAI(api_key=key)
            self._missing_key = False

    def _retry_sleep(self, attempt: int):
        time.sleep(self.retry_backoff * (2 ** (attempt - 1)))

    def generate(self, messages: List[Dict], model: str = "gpt-4o-mini", temperature: float = 0.7, stream: bool = True) -> Iterable[str]:
        if self._missing_key or self.client is None:
            raise RuntimeError("OpenAI API key is required but missing.")
        attempt = 1
        while True:
            try:
                if stream:
                    with self.client.chat.completions.stream(model=model, messages=messages, temperature=temperature) as s:
                        for event in s:
                            if event.type == "content.delta":
                                delta = event.delta or ""
                                if isinstance(delta, str) and delta:
                                    yield delta
                            elif event.type == "error":
                                raise RuntimeError(getattr(event, "error", "Unknown error"))
                        yield ""
                else:
                    resp = self.client.chat.completions.create(model=model, messages=messages, temperature=temperature)
                    yield resp.choices[0].message.content or ""
                break
            except Exception as e:
                if attempt < self.max_retries:
                    self._retry_sleep(attempt)
                    attempt += 1
                    continue
                raise RuntimeError(f"OpenAI backend failed after {attempt} attempts: {e}")
