# Skin Analyzer Service
# Multi-stage build for smaller final image

# Build stage
FROM python:3.11-slim AS builder

WORKDIR /app

# Install build dependencies and clean in same layer to reduce size
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && pip install --no-cache-dir --upgrade pip \
    && rm -rf /var/lib/apt/lists/* \
    && rm -rf /tmp/* \
    && rm -rf /var/tmp/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir --user -r requirements.txt \
    && rm -rf /tmp/* \
    && rm -rf /var/tmp/*

# Runtime stage
FROM python:3.11-slim

WORKDIR /app

# Install runtime dependencies for OpenCV and clean in same layer
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    && rm -rf /var/lib/apt/lists/* \
    && rm -rf /tmp/* \
    && rm -rf /var/tmp/* \
    && apt-get clean

# Create non-root user for security
RUN useradd -m -u 1000 appuser

# Copy Python packages from builder to appuser's home
COPY --from=builder /root/.local /home/appuser/.local
RUN chown -R appuser:appuser /home/appuser/.local

# Copy application code
COPY src/ ./src/
COPY scripts/ ./scripts/
COPY models/ ./models/

# Download models during build and clean up
RUN python scripts/download_models.py \
    && rm -rf /tmp/* \
    && rm -rf /var/tmp/* \
    && find /home/appuser/.local -type d -name __pycache__ -exec rm -r {} + 2>/dev/null || true \
    && find /home/appuser/.local -name "*.pyc" -delete \
    && find /home/appuser/.local -name "*.pyo" -delete

# Set ownership and switch to appuser
RUN chown -R appuser:appuser /app
USER appuser
ENV PATH=/home/appuser/.local/bin:$PATH \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

# Expose port for API
EXPOSE 8000

# Default command - run API server
CMD ["python", "-m", "uvicorn", "src.api:app", "--host", "0.0.0.0", "--port", "8000"]

# Alternative: Run CLI
# CMD ["python", "-m", "src.cli"]
