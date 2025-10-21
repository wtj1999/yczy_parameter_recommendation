import os
import re
import json
import numpy as np
from typing import List, Optional, Tuple

from fastapi import FastAPI, HTTPException
from fastapi.concurrency import run_in_threadpool
from pydantic import BaseModel

from catboost import CatBoostRegressor

# ===== 配置 =====
MODELS_DIR = os.getenv("MODELS_DIR", "./models_by_file")  # 模型目录根（每个模型子目录在此下）
FEATURE_ORDER = ["ryczy004", "ryczy205"]  # 默认 feature 顺序

# ===== Pydantic 模型 =====
class InputFeatures(BaseModel):
    technics_line_name: str
    device_code: str
    ryczy407: int
    ryczy004: float
    ryczy205: float

class PredictionResponse(BaseModel):
    recommendation: float
    model_used: Optional[str] = None

class BatchInput(BaseModel):
    instances: List[InputFeatures]

class BatchResponse(BaseModel):
    recommendations: List[float]


# ===== FastAPI app =====
app = FastAPI(title="Parameter Recommendation API", version="1.0.0")

_model_cache = {}
_metadata_cache = {}

_invalid_chars_re = re.compile(r'[\\/*?:"<>|]')
def _safe_name(s: str) -> str:
    s2 = _invalid_chars_re.sub("_", str(s)).strip()
    return s2 if s2 else "unknown"

def _construct_model_dir(technics_line_name: str, device_code: str, ryczy407: int) -> str:
    """
     merged_{safe_line}_{safe_device}_{safe_ryczy}
    """
    safe_line = _safe_name(technics_line_name)
    safe_device = _safe_name(device_code)
    safe_ryczy = _safe_name(str(ryczy407))
    dirname = f"merged_{safe_line}_{safe_device}_{safe_ryczy}"
    return os.path.join(MODELS_DIR, dirname)

def _load_model_and_metadata(model_dir: str):
    model_path = os.path.join(model_dir, "catboost_model.cbm")
    meta_path = os.path.join(model_dir, "metadata.json")

    if not os.path.exists(model_path):
        return None, None

    model = CatBoostRegressor()
    try:
        model.load_model(model_path)
    except Exception as e:
        raise RuntimeError(f"加载模型失败: {model_path} -> {e}")

    feature_order = None
    if os.path.exists(meta_path):
        try:
            with open(meta_path, "r", encoding="utf-8") as f:
                meta = json.load(f)
            feature_order = meta.get("feature_cols") or None
        except Exception:
            feature_order = None

    return model, feature_order

def get_model_for_request(technics_line_name: str, device_code: str, ryczy407: int):
    """
    直接根据三元键拼目录并加载模型。
    若未找到对应模型，返回 404。
    """
    model_dir = _construct_model_dir(technics_line_name, device_code, ryczy407)
    safe_line = _safe_name(technics_line_name)
    safe_device = _safe_name(device_code)
    safe_ryczy = _safe_name(str(ryczy407))
    cache_key = (safe_line, safe_device, safe_ryczy)

    if cache_key in _model_cache:
        return _model_cache[cache_key], _metadata_cache.get(cache_key), model_dir

    model, feature_order = _load_model_and_metadata(model_dir)
    if model is None:
        raise HTTPException(status_code=404, detail=f"未找到对应模型目录: {model_dir}")

    _model_cache[cache_key] = model
    _metadata_cache[cache_key] = feature_order or FEATURE_ORDER
    return model, _metadata_cache[cache_key], model_dir


@app.get("/")
def root():
    return {"status": "ok", "models_dir": MODELS_DIR, "default_feature_order": FEATURE_ORDER}


@app.post("/recommend", response_model=PredictionResponse)
async def recommend_single(payload: InputFeatures):
    """
    单条样本
    请求体示例:
    {
      "technics_line_name": "123Ah_XZ2-02-02-ZY-YC-01",
      "device_code": "XZ2-02-02",
      "ryczy407": 1,
      "ryczy004": 1.23,
      "ryczy205": 4.56
    }
    """
    try:
        model, feature_order, model_dir = get_model_for_request(
            payload.technics_line_name, payload.device_code, payload.ryczy407
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"加载模型失败: {e}")

    feature_order = feature_order or FEATURE_ORDER
    try:
        feature_values = []
        for fname in feature_order:
            if not hasattr(payload, fname):
                raise HTTPException(status_code=400, detail=f"请求体中缺少特征字段: {fname}")
            feature_values.append(getattr(payload, fname))
        X = np.array([feature_values], dtype=float)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"构造特征向量失败: {e}")

    try:
        preds = await run_in_threadpool(model.predict, X)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"模型预测失败: {e}")

    pred_val = round(float(preds[0]), 4)
    return PredictionResponse(recommendation=pred_val, model_used=model_dir)

