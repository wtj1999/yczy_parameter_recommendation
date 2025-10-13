from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional
from catboost import CatBoostRegressor
import numpy as np
import os
from starlette.concurrency import run_in_threadpool

# ===== 配置 =====
MODEL_PATH = os.getenv("MODEL_PATH", "./models/catboost_model.cbm")
FEATURE_ORDER = ["ryczy004", "ryczy205"]

class InputFeatures(BaseModel):
    ryczy004: float
    ryczy205: float

class PredictionResponse(BaseModel):
    recommendation: float

class BatchInput(BaseModel):
    instances: List[InputFeatures]

class BatchResponse(BaseModel):
    recommendations: List[float]

# ===== FastAPI app =====
app = FastAPI(title="CatBoost Recommendation API", version="1.0.0")
_model: Optional[CatBoostRegressor] = None

@app.on_event("startup")
def load_model():
    global _model
    if not os.path.exists(MODEL_PATH):
        raise RuntimeError(f"模型文件找不到: {MODEL_PATH}")
    m = CatBoostRegressor()
    m.load_model(MODEL_PATH)
    _model = m

@app.get("/")
def root():
    return {"status": "ok", "model_path": MODEL_PATH, "features": FEATURE_ORDER}

@app.post("/recommend", response_model=PredictionResponse)
async def recommend_single(payload: InputFeatures):
    """
    单条样本。请求体:
    {
      "ryczy004": 1.23,
      "ryczy205": 4.56
    }
    """
    if _model is None:
        raise HTTPException(status_code=500, detail="模型未加载")

    X = np.array([[payload.ryczy004, payload.ryczy205]])
    try:
        preds = await run_in_threadpool(_model.predict, X)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"推荐失败：{e}")
    pred_val = round(float(preds[0]), 4)
    return PredictionResponse(recommendation=pred_val)

@app.post("/recommend_batch", response_model=BatchResponse)
async def recommend_batch(payload: BatchInput):
    """
    批量预测。请求体:
    {
      "instances": [
        {"ryczy004": 1.23, "ryczy205": 4.56},
        {"ryczy004": 2.34, "ryczy205": 5.67}
      ]
    }
    """
    if _model is None:
        raise HTTPException(status_code=500, detail="模型未加载")
    X = np.array([[i.ryczy004, i.ryczy205] for i in payload.instances])
    try:
        preds = await run_in_threadpool(_model.predict, X)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"批量推荐失败：{e}")
    return BatchResponse(recommendations=[round(float(x), 4) for x in preds.tolist()])
