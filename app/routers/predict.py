from fastapi import APIRouter, HTTPException
from app.schemas import LaptopFeatures
from app.model import model_wrapper

router = APIRouter()

@router.post("/predict")
async def predict_price(features: LaptopFeatures):
    try:
        price = model_wrapper.predict(features.dict())
        return {"predicted_price": round(price, 2)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
