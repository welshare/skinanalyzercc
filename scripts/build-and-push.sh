#!/bin/bash
set -e

# Configuration
REGISTRY="ghcr.io"
USERNAME="elmariachi111"
IMAGE_NAME="skin-analyzer"
VERSION=${1:-latest}

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${YELLOW}Building and pushing Docker image...${NC}"
echo "Registry: ${REGISTRY}"
echo "Image: ${REGISTRY}/${USERNAME}/${IMAGE_NAME}:${VERSION}"
echo ""

# Check if buildx is available
if ! docker buildx version &>/dev/null; then
    echo "⚠️  Docker buildx not available. Using standard build for current platform."
    echo ""
    
    # Standard build for current platform
    echo "Building image..."
    docker build -t ${REGISTRY}/${USERNAME}/${IMAGE_NAME}:${VERSION} .
    
    echo "Tagging image..."
    docker tag ${REGISTRY}/${USERNAME}/${IMAGE_NAME}:${VERSION} ${REGISTRY}/${USERNAME}/${IMAGE_NAME}:${VERSION}
    
    echo "Pushing image..."
    docker push ${REGISTRY}/${USERNAME}/${IMAGE_NAME}:${VERSION}
    
    echo -e "${GREEN}✅ Image pushed successfully!${NC}"
    exit 0
fi

# Check if user wants multi-arch or single arch
if [ "$2" == "--amd64-only" ]; then
    echo "Building for AMD64 only (Linux x86_64)..."
    docker build --platform linux/amd64 -t ${REGISTRY}/${USERNAME}/${IMAGE_NAME}:${VERSION} .
    docker push ${REGISTRY}/${USERNAME}/${IMAGE_NAME}:${VERSION}
    echo -e "${GREEN}✅ AMD64 image pushed successfully!${NC}"
    exit 0
fi

# Multi-architecture build
echo "Building multi-architecture image (AMD64 + ARM64)..."
echo ""

# Create or use buildx builder
if ! docker buildx ls | grep -q multiarch; then
    echo "Creating buildx builder..."
    docker buildx create --name multiarch --use
else
    echo "Using existing buildx builder..."
    docker buildx use multiarch
fi

# Build and push for multiple platforms
docker buildx build \
  --platform linux/amd64,linux/arm64 \
  -t ${REGISTRY}/${USERNAME}/${IMAGE_NAME}:${VERSION} \
  --push .

echo ""
echo -e "${GREEN}✅ Multi-architecture image pushed successfully!${NC}"
echo "   - AMD64 (Linux x86_64) for servers"
echo "   - ARM64 (Apple Silicon, ARM servers)"
echo ""
echo "Image available at: ${REGISTRY}/${USERNAME}/${IMAGE_NAME}:${VERSION}"

