from fastapi import FastAPI
from app.routers import predict

app = FastAPI(title="Laptop Price Predictor API")
app.include_router(predict.router, prefix="/api")

@app.get("/")
async def root():
    return {"messege": "Welcome to the Laptop Price Predictor API!"}

