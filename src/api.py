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
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, File, UploadFile, HTTPException, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import cv2
import numpy as np

from .skin_analyzer import SkinAnalyzer

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
        analyzer = SkinAnalyzer()
    return analyzer


@app.get("/health")
async def health_check():
    """Health check endpoint."""
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
    # Validate file type
    if not file.content_type.startswith("image/"):
        raise HTTPException(400, "File must be an image")

    # Read image
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    if image is None:
        raise HTTPException(400, "Could not decode image")

    # Analyze
    skin_analyzer = get_analyzer()
    result = skin_analyzer.analyze(image)

    if not result["success"]:
        raise HTTPException(422, result.get("error", "Analysis failed"))

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
    if not file.content_type.startswith("image/"):
        raise HTTPException(400, "File must be an image")

    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    if image is None:
        raise HTTPException(400, "Could not decode image")

    skin_analyzer = get_analyzer()
    result = skin_analyzer.quick_check(image)

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
    valid_regions = ["forehead", "left_cheek", "right_cheek", "nose", "chin"]
    if region not in valid_regions:
        raise HTTPException(400, f"Invalid region. Must be one of: {valid_regions}")

    if not file.content_type.startswith("image/"):
        raise HTTPException(400, "File must be an image")

    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    if image is None:
        raise HTTPException(400, "Could not decode image")

    skin_analyzer = get_analyzer()
    result = skin_analyzer.analyze_region(image, region)

    if not result["success"]:
        raise HTTPException(422, result.get("error", "Analysis failed"))

    return JSONResponse(content=skin_analyzer._make_serializable(result))


@app.get("/info")
async def get_info():
    """Get information about the analyzer capabilities."""
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
