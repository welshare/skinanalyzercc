# Troubleshooting Docker Issues

## Error: "exec format error" / "exec /usr/local/bin/python: exec format error"

This error occurs when there's an **architecture mismatch** between the Docker image and the host system.

### 🔍 Cause
- Image was built on **ARM64** (Apple Silicon Mac) but host is **AMD64/x86_64** (Linux servers)
- Or vice versa: Image built on **AMD64** but host is **ARM64**

### ✅ Solutions

#### Solution 1: Build for AMD64 (Most Common for Remote Servers)

**Option A: Build with platform flag**
```bash
# Build specifically for AMD64 (Linux x86_64)
docker build --platform linux/amd64 -t ghcr.io/elmariachi111/skin-analyzer:latest .

# Push
docker push ghcr.io/elmariachi111/skin-analyzer:latest
```

**Option B: Use buildx for multi-architecture**
```bash
# Create and use buildx builder
docker buildx create --name multiarch --use

# Build for both architectures
docker buildx build --platform linux/amd64,linux/arm64 \
  -t ghcr.io/elmariachi111/skin-analyzer:latest \
  --push .
```

#### Solution 2: Use docker-compose with platform

Update your `docker-compose.yml`:
```yaml
services:
  skin-analyzer:
    image: ghcr.io/elmariachi111/skin-analyzer:latest
    platform: linux/amd64  # Force AMD64 architecture
    expose:
      - 8000
```

#### Solution 3: Build on the Remote Server

Build directly on the target architecture:
```bash
# SSH into remote server
ssh user@remote-server

# Build there
docker build -t ghcr.io/elmariachi111/skin-analyzer:latest .
docker push ghcr.io/elmariachi111/skin-analyzer:latest
```

### 🔍 Verify Architecture

**Check your local machine:**
```bash
uname -m
# Apple Silicon: arm64
# Intel Mac: x86_64
```

**Check Docker image architecture:**
```bash
docker inspect ghcr.io/elmariachi111/skin-analyzer:latest | grep Architecture
```

**Check remote server architecture:**
```bash
# On remote server
uname -m
# Most Linux servers: x86_64
```

### 📝 Recommended: Multi-Architecture Build Script

Create a `build-multiarch.sh` script:
```bash
#!/bin/bash
set -e

REGISTRY="ghcr.io"
USERNAME="elmariachi111"
IMAGE_NAME="skin-analyzer"
VERSION=${1:-latest}

echo "Building multi-architecture image..."

docker buildx create --name multiarch --use 2>/dev/null || docker buildx use multiarch

docker buildx build \
  --platform linux/amd64,linux/arm64 \
  -t ${REGISTRY}/${USERNAME}/${IMAGE_NAME}:${VERSION} \
  --push .

echo "✅ Pushed ${REGISTRY}/${USERNAME}/${IMAGE_NAME}:${VERSION} for AMD64 and ARM64"
```

---

## Error: "manifest unknown" / "docker compose pull failed: manifest unknown"

This error means Docker cannot find the image at the specified registry location. Here are the most common causes and solutions:

### 🔍 Common Causes

1. **Image hasn't been pushed to the registry yet** (Most Common)
2. **Wrong image path or tag**
3. **Authentication failure** (for private images)
4. **Image is private but no credentials provided**
5. **Typo in registry URL or image name**
6. **Tag doesn't exist** (e.g., using `:latest` but only `:v1.0.0` exists)

---

## Solutions

### 1. Verify Image Exists in Registry

**For GHCR:**
```bash
# Check if image exists (replace with your details)
curl -s https://ghcr.io/v2/elmariachi111/skin-analyzer/tags/list

# Or visit in browser:
# https://github.com/users/elmariachi111/packages/container/skin-analyzer/versions
```

**For Docker Hub:**
```bash
# Check if image exists
curl -s https://hub.docker.com/v2/repositories/USERNAME/skin-analyzer/tags/
```

### 2. Build and Push the Image First

If you haven't pushed the image yet:

```bash
# Build the image
docker build -t ghcr.io/elmariachi111/skin-analyzer:latest .

# Login to registry
echo $GITHUB_TOKEN | docker login ghcr.io -u elmariachi111 --password-stdin

# Push the image
docker push ghcr.io/elmariachi111/skin-analyzer:latest
```

### 3. Check Your docker-compose.yml

Make sure the image reference matches exactly what you pushed:

```yaml
services:
  skin-analyzer:
    image: ghcr.io/elmariachi111/skin-analyzer:latest  # Must match exactly
    ports:
      - "8000:8000"
```

**Common mistakes:**
- ❌ `ghcr.io/elmariachi111/skin-analyzer` (missing tag)
- ❌ `ghcr.io/elmariachi111/skin-analyzer:v1` (tag doesn't exist)
- ❌ `ghcr.io/elmariachi111/skin-analyzer:latest` (image not pushed)
- ✅ `ghcr.io/elmariachi111/skin-analyzer:latest` (correct, if pushed)

### 4. Authentication for Private Images

If your image is private, you need to authenticate:

**Option A: Login before docker compose**
```bash
# Login to GHCR
echo $GITHUB_TOKEN | docker login ghcr.io -u elmariachi111 --password-stdin

# Then run docker compose
docker compose pull
```

**Option B: Use docker-compose.yml with auth**
```yaml
services:
  skin-analyzer:
    image: ghcr.io/elmariachi111/skin-analyzer:latest
    # Docker Compose will use your local Docker credentials
```

**Option C: Make image public**
1. Go to: https://github.com/users/elmariachi111/packages
2. Click on `skin-analyzer` package
3. Package settings → Change visibility → Make public

### 5. Test Image Pull Manually

Before using docker-compose, test pulling the image directly:

```bash
# Test pull
docker pull ghcr.io/elmariachi111/skin-analyzer:latest

# If this fails, the issue is with the image/registry
# If this succeeds, the issue is with docker-compose configuration
```

### 6. Check Available Tags

List all available tags for your image:

```bash
# For GHCR (requires authentication for private repos)
curl -H "Authorization: Bearer $GITHUB_TOKEN" \
  https://ghcr.io/v2/elmariachi111/skin-analyzer/tags/list

# Or use Docker manifest inspect
docker manifest inspect ghcr.io/elmariachi111/skin-analyzer:latest
```

### 7. Use Local Build Instead

If you don't want to use a registry, build locally:

```yaml
services:
  skin-analyzer:
    build:
      context: .
      dockerfile: Dockerfile
    ports:
      - "8000:8000"
```

Then run:
```bash
docker compose up --build
```

---

## Step-by-Step Debugging

### Step 1: Verify Image Path
```bash
# Check what image docker-compose is trying to pull
docker compose config | grep image
```

### Step 2: Check Authentication
```bash
# Verify you're logged in
docker login ghcr.io
# Should show: "Login Succeeded"
```

### Step 3: Test Direct Pull
```bash
# Try pulling the exact image from docker-compose.yml
docker pull ghcr.io/elmariachi111/skin-analyzer:latest
```

### Step 4: Check Image Visibility
- Public images: No auth needed
- Private images: Must authenticate

### Step 5: Verify Image Exists
```bash
# List tags (may require auth)
curl -H "Authorization: Bearer $GITHUB_TOKEN" \
  https://ghcr.io/v2/elmariachi111/skin-analyzer/tags/list
```

---

## Quick Fixes

### Fix 1: Build and Push Image
```bash
# Complete workflow
docker build -t ghcr.io/elmariachi111/skin-analyzer:latest .
docker login ghcr.io
docker push ghcr.io/elmariachi111/skin-analyzer:latest
docker compose pull
```

### Fix 2: Use Local Build
```yaml
# docker-compose.yml
services:
  skin-analyzer:
    build: .
    ports:
      - "8000:8000"
```

### Fix 3: Check Exact Image Name
```bash
# See what docker-compose is looking for
docker compose config

# Compare with what's in registry
docker images | grep skin-analyzer
```

---

## Example: Complete Working Setup

### 1. Build and Push
```bash
# Build
docker build -t ghcr.io/elmariachi111/skin-analyzer:latest .

# Login
echo $GITHUB_TOKEN | docker login ghcr.io -u elmariachi111 --password-stdin

# Push
docker push ghcr.io/elmariachi111/skin-analyzer:latest
```

### 2. docker-compose.yml
```yaml
services:
  skin-analyzer:
    image: ghcr.io/elmariachi111/skin-analyzer:latest
    ports:
      - "8000:8000"
    environment:
      - PYTHONUNBUFFERED=1
```

### 3. Pull and Run
```bash
# Pull (will use cached credentials)
docker compose pull

# Run
docker compose up
```

---

## Still Having Issues?

1. **Check Docker logs:**
   ```bash
   docker compose pull --verbose
   ```

2. **Verify network connectivity:**
   ```bash
   curl -I https://ghcr.io
   ```

3. **Check Docker daemon:**
   ```bash
   docker info
   ```

4. **Try different tag:**
   ```bash
   # If :latest doesn't work, try a specific version
   docker pull ghcr.io/elmariachi111/skin-analyzer:v1.0.0
   ```

