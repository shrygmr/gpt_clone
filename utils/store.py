# utils/store.py
from __future__ import annotations
import io
import json
import time
from typing import Any, Dict, List

try:
    import orjson as _json
    def _dumps(obj: Any) -> bytes:
        return _json.dumps(obj)
    def _loads(b: bytes) -> Any:
        return _json.loads(b)
except Exception:
    def _dumps(obj: Any) -> bytes:
        return json.dumps(obj, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    def _loads(b: bytes) -> Any:
        return json.loads(b.decode("utf-8"))

def serialize_session(messages: List[Dict], feedback: Dict[int, str]) -> bytes:
    payload = {
        "messages": messages,
        "feedback": feedback,
        "updated_at": int(time.time()),
        "version": 1,
    }
    return _dumps(payload)

def save_session_to_path(path: str, messages: List[Dict], feedback: Dict[int, str]) -> None:
    data = serialize_session(messages, feedback)
    with open(path, "wb") as f:
        f.write(data)

def load_session_from_fileobj(file_obj: io.BufferedIOBase) -> Dict:
    raw = file_obj.read()
    return _loads(raw)
