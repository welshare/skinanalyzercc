# Skin Analyzer

A service for analyzing facial skin attributes including texture, pigmentation, and tone.

## Features

- **Face Detection**: YuNet-based lightweight face detection (~300KB)
- **Skin Segmentation**: Color-space based skin region extraction
- **Texture Analysis**: GLCM-based texture metrics, pore and wrinkle detection
- **Pigmentation Analysis**: Dark/light spot detection, evenness scoring
- **Tone Analysis**: Skin tone classification, undertone detection

## Model Sizes

| Component | Size |
|-----------|------|
| YuNet Face Detector | ~300KB |
| GLCM/Color Analysis | No weights (algorithm-based) |
| **Total** | **< 1MB** |

## Installation

### Local Development

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # or `venv\Scripts\activate` on Windows

# Install dependencies
pip install -r requirements.txt

# Download models
python scripts/download_models.py
```

### Docker

```bash
# Build image
docker build -t skin-analyzer .

# Run container
docker run -p 8000:8000 skin-analyzer
```

## Usage

### Python API

```python
from src.skin_analyzer import SkinAnalyzer

analyzer = SkinAnalyzer()
result = analyzer.analyze("path/to/image.jpg")

print(f"Skin Tone: {result['summary']['skin_tone']}")
print(f"Condition Score: {result['summary']['condition_score']}/100")
print(f"Key Findings: {result['summary']['key_findings']}")
```

### Command Line

```bash
# Full analysis
python -m src.cli analyze image.jpg --verbose

# Quick check
python -m src.cli check image.jpg

# Specific region
python -m src.cli region image.jpg --region left_cheek
```

### REST API

Start the server:
```bash
uvicorn src.api:app --host 0.0.0.0 --port 8000
```

Endpoints:

```bash
# Full analysis
curl -X POST "http://localhost:8000/analyze" \
  -F "file=@image.jpg" \
  -F "verbose=true"

# Quick check
curl -X POST "http://localhost:8000/check" \
  -F "file=@image.jpg"

# Region analysis
curl -X POST "http://localhost:8000/analyze/region" \
  -F "file=@image.jpg" \
  -F "region=left_cheek"

# Service info
curl "http://localhost:8000/info"
```

## Output Structure

```json
{
  "success": true,
  "has_face": true,
  "face_detection": {
    "confidence": 0.95
  },
  "summary": {
    "texture": "smooth",
    "pigmentation": "even_tone",
    "skin_tone": "medium_light",
    "undertone": "warm",
    "condition_score": 78.5,
    "key_findings": ["No significant concerns"]
  },
  "texture_analysis": {
    "smoothness_score": 0.72,
    "roughness_score": 0.28,
    "pore_analysis": {
      "pore_count": 45,
      "pore_density": 1.2
    },
    "glcm_features": {
      "homogeneity": 0.85,
      "contrast": 120.5,
      "energy": 0.02
    }
  },
  "pigmentation_analysis": {
    "dark_spot_count": 8,
    "evenness_score": 0.82,
    "melanin_distribution": {
      "melanin_index": 0.35,
      "distribution": "even"
    }
  },
  "tone_analysis": {
    "tone": "medium_light",
    "undertone": "warm",
    "dominant_color": {
      "hex": "#d4a574"
    },
    "color_uniformity": 0.78
  }
}
```

## Analysis Metrics

### Texture
- **Smoothness Score** (0-1): Higher = smoother skin
- **Pore Density**: Pores per 10,000 pixels
- **GLCM Homogeneity**: Texture uniformity

### Pigmentation
- **Evenness Score** (0-1): Higher = more even tone
- **Dark Spots**: Count of hyperpigmented areas
- **Melanin Index** (0-1): Estimated melanin content

### Tone
- **Skin Tone**: Very Light → Dark (6 levels)
- **Undertone**: Warm / Cool / Neutral
- **Color Uniformity** (0-1): Higher = more uniform

## Limitations

- **Moisture/Sebum**: Cannot be detected from images (requires sensors)
- **Specific Conditions**: General spot detection, not medical diagnosis
- **Lighting**: Results vary with image lighting conditions

## Project Structure

```
nilcc-demo/
├── src/
│   ├── skin_analyzer/
│   │   ├── __init__.py
│   │   ├── analyzer.py      # Main service
│   │   ├── detector.py      # Face detection
│   │   ├── parser.py        # Skin segmentation
│   │   ├── texture.py       # Texture analysis
│   │   ├── pigmentation.py  # Spot detection
│   │   └── tone.py          # Color analysis
│   ├── api.py               # REST API
│   └── cli.py               # CLI interface
├── scripts/
│   └── download_models.py
├── models/                   # Downloaded models
├── Dockerfile
├── requirements.txt
└── README.md
```

## License

MIT
