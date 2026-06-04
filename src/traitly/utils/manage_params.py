# traitly/utils/manage_json.py
import os
import json
from typing import Optional, Dict

def _import_params(
    json_path: Optional[str] = None,
    config: Optional[Dict] = None,
):
    if json_path is not None and os.path.exists(json_path):
        with open(json_path, "r", encoding="utf-8") as f:
            params = json.load(f)
    elif config is not None:
        params = config
    else:
        params = {}

    return params

def _get_params(params: Dict, section: str) -> Dict:
    return params.get(section, {}) or {}

def _clean_params(d: Dict) -> Dict:
    """Remove None values to avoid overriding defaults."""
    return {k: v for k, v in d.items() if v is not None}
