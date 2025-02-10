import os
import time
from contextlib import asynccontextmanager

import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from keras import Model
from pydantic import BaseModel

# Configuration and Initialization
from src.config import Config
from src.helpers.logger_config import LoggerConfig
from src.helpers.model_utils import ModelUtils
from src.x_ray_inference import XRayInference

# Set CUDA_VISIBLE_DEVICES based on the Config
if Config.DEVICE == 'CPU':
    os.environ['CUDA_VISIBLE_DEVICES'] = ''

logger = LoggerConfig().get_logger('CXR_APP', Config.CXR_APP_LOG_DIR)

# Global variables for the model and inference
model: Model
xray_inference: XRayInference


@asynccontextmanager
async def app_lifespan(app_instance: FastAPI):
    """
    Context manager to handle application startup and shutdown.

    Parameters:
    app_instance (FastAPI): The FastAPI application instance.
    """
    global model, xray_inference
    try:
        logger.info('Initializing model...')
        model = ModelUtils.load_model(Config.CKPT_PATH if Config.IS_CHECKPOINT else Config.MODEL_PATH)
        xray_inference = XRayInference(model=model)
        logger.info('Model loaded successfully')
        yield
    finally:
        logger.info('Shutting down application...')


app = FastAPI(lifespan=app_lifespan)

# CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=Config.ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=['POST'],
    allow_headers=['Content-Type'],
)


# Pydantic Model
class InferenceRequest(BaseModel):
    """
    Defines the request body for prediction.
    """
    cxr_base64: str


@app.exception_handler(Exception)
async def global_exception_handler(exc: Exception):
    """
    Handles unexpected errors gracefully.
    """
    logger.error(f'Unhandled error: {str(exc)}')
    return JSONResponse(
        status_code=500,
        content={'success': False, 'error': 'Internal Server Error'}
    )


@app.post('/predict')
async def predict(request: InferenceRequest) -> JSONResponse:
    """
    Inference endpoint for base64-encoded X-ray image.
    """
    logger.info('Received prediction request')
    start_time = time.time()
    try:
        heatmap_base64, predictions, uncertainties = await xray_inference.run_inference(request.cxr_base64)
        duration = time.time() - start_time

        logger.info(f'Prediction completed successfully in {duration:.2f}s')

        return JSONResponse(
            status_code=200,
            content={
                'success': True,
                'heatmap': heatmap_base64,
                'prediction_mean': predictions,
                'prediction_variance': uncertainties,
                'inference_duration': duration
            }
        )
    except Exception as e:
        logger.error(f'Prediction error: {str(e)}')
        raise HTTPException(status_code=500, detail='Prediction failed')


if __name__ == '__main__':
    logger.info(f'Starting prediction app (workers={Config.WORKERS})...')
    uvicorn.run(Config.APP, host=Config.HOST, port=Config.PORT, workers=Config.WORKERS)
