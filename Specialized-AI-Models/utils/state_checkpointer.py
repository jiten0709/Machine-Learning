import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional
from datetime import datetime, timezone
from pydantic import BaseModel

from .logging_setup import get_logger

class StateCheckpointer:
    """
    Persists individual stage results to a single static JSON file.
    Each stage key maps to its Pydantic model's .model_dump() / .dict().
    """

    def __init__(self, directory: Path, filename: str = "checkpoint.json", logger: Optional[logging.Logger] = None) -> None:
        self.dir = directory
        self.filename = filename
        self.dir.mkdir(parents=True, exist_ok=True)
        self.logger = logger or get_logger("StateCheckpointer")

    def _path(self) -> Path:
        return self.dir / self.filename

    def _load_raw(self) -> Dict:
        p = self._path()
        if not p.exists():
            return {}
        return json.loads(p.read_text(encoding="utf-8"))

    def _save_raw(self, data: Dict) -> None:
        p = self._path()
        p.write_text(json.dumps(data, default=str, indent=2), encoding="utf-8")
        self.logger.debug(f"Checkpoint saved → {p.name}")

    def exists(self, key: str) -> bool:
        return key in self._load_raw()

    def save(self, key: str, model: BaseModel) -> None:
        data = self._load_raw()
        data[key] = model.model_dump(mode="json")
        data["__updated_at__"] = datetime.now(timezone.utc).isoformat()
        self._save_raw(data)
        self.logger.info(f"📨 Stage '{key}' checkpointed in {self.filename}")

    def load(self, key: str, model_cls: type) -> Optional[Any]:
        data = self._load_raw()
        if key not in data:
            return None
        try:
            instance = model_cls.model_validate(data[key]) 
            self.logger.info(f"♻️ Loaded checkpoint; resuming from stage '{key}'")
            return instance
        except Exception as e:
            self.logger.warning(f"⚠️ Could not restore '{key}': {e}")
            return None

    def save_raw_key(self, key: str, value: Any) -> None:
        data = self._load_raw()
        data[key] = value
        self._save_raw(data)

    def load_raw_key(self, key: str) -> Optional[Any]:
        data = self._load_raw()
        if key in data:
            self.logger.info(f"♻️ Loaded raw checkpoint; resuming from stage '{key}'")
            return data[key]
        return None

    def list_sessions(self) -> List[str]:
        p = self._path()
        return [self.filename] if p.exists() else []