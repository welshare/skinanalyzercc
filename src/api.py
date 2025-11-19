"""
FastAPI REST API for the skin analyzer service.

Endpoints:
    POST /analyze - Full skin analysis
    POST /check - Quick suitability check
    POST /analyze/region - Analyze specific region
    GET /health - Health check
"""

import tempfile
import os
import logging
import sys
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, File, UploadFile, HTTPException, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import cv2
import numpy as np

from .skin_analyzer import SkinAnalyzer

# Configure logging with timestamps - must be done before any other imports that use logging
LOGGING_CONFIG = {
    "version": 1,
    "disable_existing_loggers": False,
    "formatters": {
        "default": {
            "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            "datefmt": "%Y-%m-%d %H:%M:%S",
        },
        "uvicorn": {
            "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            "datefmt": "%Y-%m-%d %H:%M:%S",
        },
    },
    "handlers": {
        "default": {
            "formatter": "default",
            "class": "logging.StreamHandler",
            "stream": "ext://sys.stdout",
        },
    },
    "root": {
        "level": "INFO",
        "handlers": ["default"],
    },
    "loggers": {
        "uvicorn": {
            "handlers": ["default"],
            "level": "INFO",
            "propagate": False,
        },
        "uvicorn.error": {
            "handlers": ["default"],
            "level": "INFO",
            "propagate": False,
        },
        "uvicorn.access": {
            "handlers": ["default"],
            "level": "INFO",
            "propagate": False,
        },
        "src": {
            "handlers": ["default"],
            "level": "INFO",
            "propagate": False,
        },
    },
}

# Apply logging configuration
import logging.config
logging.config.dictConfig(LOGGING_CONFIG)

logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Skin Analyzer API",
    description="Facial skin attribute analysis service",
    version="0.1.0"
)

# Add CORS middleware to allow cross-origin requests
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins - configure specific origins in production
    allow_credentials=False,  # Cannot be True when allow_origins=["*"]
    allow_methods=["*"],  # Allow all methods (GET, POST, etc.)
    allow_headers=["*"],  # Allow all headers
)

# Initialize analyzer (singleton)
analyzer = None


def get_analyzer() -> SkinAnalyzer:
    """Get or create the analyzer instance."""
    global analyzer
    if analyzer is None:
        logger.info("Creating new SkinAnalyzer instance...")
        try:
            analyzer = SkinAnalyzer()
            logger.info("SkinAnalyzer instance created successfully")
        except Exception as e:
            logger.error(f"Failed to create SkinAnalyzer instance: {e}", exc_info=True)
            raise
    else:
        logger.debug("Reusing existing SkinAnalyzer instance")
    return analyzer


@app.on_event("startup")
async def startup_event():
    """Log application startup."""
    logger.info("=" * 60)
    logger.info("Starting Skin Analyzer API")
    logger.info("Version: 0.1.0")
    logger.info("Initializing application components...")
    
    # Pre-initialize analyzer to log any initialization issues
    try:
        analyzer_instance = get_analyzer()
        logger.info("Skin Analyzer initialized successfully")
        logger.info("Application startup complete")
        logger.info("=" * 60)
    except Exception as e:
        logger.error(f"Failed to initialize Skin Analyzer: {e}", exc_info=True)
        raise


@app.on_event("shutdown")
async def shutdown_event():
    """Log application shutdown."""
    logger.info("=" * 60)
    logger.info("Shutting down Skin Analyzer API")
    logger.info("Application shutdown complete")
    logger.info("=" * 60)


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    logger.info("Health check requested")
    return {
        "status": "healthy",
        "service": "skin-analyzer",
        "version": "0.1.0"
    }


@app.post("/analyze")
async def analyze_image(
    file: UploadFile = File(...),
    verbose: bool = Form(default=False)
):
    """
    Perform full skin analysis on uploaded image.

    Args:
        file: Image file (JPEG, PNG)
        verbose: Include detailed metrics in response

    Returns:
        Complete analysis results
    """
    logger.info(f"Received analyze request: filename={file.filename}, verbose={verbose}")
    
    # Validate file type
    if not file.content_type.startswith("image/"):
        logger.warning(f"Invalid file type: {file.content_type}")
        raise HTTPException(400, "File must be an image")

    # Read image
    contents = await file.read()
    logger.debug(f"Read image file: {len(contents)} bytes")
    nparr = np.frombuffer(contents, np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    if image is None:
        logger.error("Could not decode image")
        raise HTTPException(400, "Could not decode image")

    logger.info(f"Image decoded: {image.shape[1]}x{image.shape[0]} pixels")

    # Analyze
    skin_analyzer = get_analyzer()
    logger.info("Starting skin analysis...")
    result = skin_analyzer.analyze(image)

    if not result["success"]:
        error_msg = result.get("error", "Analysis failed")
        logger.error(f"Analysis failed: {error_msg}")
        raise HTTPException(422, error_msg)

    logger.info(f"Analysis completed successfully: face_detected={result.get('has_face', False)}")

    # Simplify response if not verbose
    if not verbose:
        result = {
            "success": True,
            "has_face": result["has_face"],
            "face_detection": {
                "confidence": result["face_detection"]["confidence"]
            },
            "summary": result["summary"],
            "skin_tone": result["tone_analysis"]["tone"],
            "undertone": result["tone_analysis"]["undertone"],
            "texture_score": result["texture_analysis"]["smoothness_score"],
            "evenness_score": result["pigmentation_analysis"]["evenness_score"],
            "spot_count": result["pigmentation_analysis"]["total_spot_count"]
        }

    return JSONResponse(content=skin_analyzer._make_serializable(result))


@app.post("/check")
async def check_image(file: UploadFile = File(...)):
    """
    Quick check if image is suitable for analysis.

    Args:
        file: Image file

    Returns:
        Suitability check results
    """
    logger.info(f"Received check request: filename={file.filename}")
    
    if not file.content_type.startswith("image/"):
        logger.warning(f"Invalid file type: {file.content_type}")
        raise HTTPException(400, "File must be an image")

    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    if image is None:
        logger.error("Could not decode image")
        raise HTTPException(400, "Could not decode image")

    skin_analyzer = get_analyzer()
    logger.info("Performing quick check...")
    result = skin_analyzer.quick_check(image)
    
    logger.info(f"Quick check completed: suitable={result.get('is_suitable_for_analysis', False)}")

    return result


@app.post("/analyze/region")
async def analyze_region(
    file: UploadFile = File(...),
    region: str = Form(...)
):
    """
    Analyze specific facial region.

    Args:
        file: Image file
        region: Region name (forehead, left_cheek, right_cheek, nose, chin)

    Returns:
        Region-specific analysis results
    """
    logger.info(f"Received region analysis request: filename={file.filename}, region={region}")
    
    valid_regions = ["forehead", "left_cheek", "right_cheek", "nose", "chin"]
    if region not in valid_regions:
        logger.warning(f"Invalid region requested: {region}")
        raise HTTPException(400, f"Invalid region. Must be one of: {valid_regions}")

    if not file.content_type.startswith("image/"):
        logger.warning(f"Invalid file type: {file.content_type}")
        raise HTTPException(400, "File must be an image")

    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    if image is None:
        logger.error("Could not decode image")
        raise HTTPException(400, "Could not decode image")

    skin_analyzer = get_analyzer()
    logger.info(f"Starting region analysis for: {region}")
    result = skin_analyzer.analyze_region(image, region)

    if not result["success"]:
        error_msg = result.get("error", "Analysis failed")
        logger.error(f"Region analysis failed: {error_msg}")
        raise HTTPException(422, error_msg)

    logger.info(f"Region analysis completed successfully for: {region}")

    return JSONResponse(content=skin_analyzer._make_serializable(result))


@app.get("/info")
async def get_info():
    """Get information about the analyzer capabilities."""
    logger.info("Info endpoint requested")
    return {
        "service": "Skin Analyzer",
        "version": "0.1.0",
        "capabilities": {
            "face_detection": {
                "model": "YuNet",
                "size_mb": 0.3
            },
            "skin_segmentation": {
                "method": "Color-space based (HSV + YCrCb)",
                "deep_model_available": False
            },
            "texture_analysis": {
                "methods": ["GLCM", "Blob detection", "Edge detection"],
                "metrics": ["smoothness", "roughness", "pore_density", "wrinkle_density"]
            },
            "pigmentation_analysis": {
                "detects": ["dark_spots", "light_spots", "redness", "freckles"],
                "metrics": ["evenness_score", "melanin_index"]
            },
            "tone_analysis": {
                "classifications": ["skin_tone", "undertone"],
                "metrics": ["dominant_color", "uniformity", "brightness"]
            }
        },
        "regions": ["forehead", "left_cheek", "right_cheek", "nose", "chin"],
        "supported_formats": ["JPEG", "PNG", "BMP", "TIFF"]
    }


# Run with: uvicorn src.api:app --reload
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
