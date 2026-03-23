# src/model/json_utils.py
import json, re
from typing import Any, Dict, Optional

ALLOWED_GRADES = {"<grade_1>", "<grade_2>", "<grade_3>", "<grade_4>", "<grade_5>"}

def extract_last_json(text: str) -> Optional[Dict[str, Any]]:
    s = text.rfind("{")
    e = text.rfind("}")
    if s == -1 or e == -1 or e <= s:
        return None
    chunk = text[s:e+1]
    try:
        return json.loads(chunk)
    except Exception:
        return None

def normalize_self_score(v: Any) -> str:
    v = (v or "")
    if not isinstance(v, str):
        v = str(v)
    v = v.strip()
    if v in ALLOWED_GRADES:
        return v
    m = re.search(r"<grade_[1-5]>", v)
    if m:
        return m.group(0)
    return "<grade_3>"

