# Docker Registry Recommendations

## Quick Comparison

| Registry | Best For | Free Tier | Authentication | Private Repos |
|----------|----------|-----------|----------------|---------------|
| **Docker Hub** | General use, public images | 1 private repo | Username/password | Limited |
| **GitHub Container Registry (GHCR)** | GitHub projects | Unlimited public | GitHub token | Unlimited |
| **AWS ECR** | AWS deployments | Pay-as-you-go | AWS IAM | Unlimited |
| **Google Container Registry** | GCP deployments | Pay-as-you-go | GCP service account | Unlimited |
| **Azure Container Registry** | Azure deployments | Pay-as-you-go | Azure AD | Unlimited |

## Recommendations

### 🏆 **Best Overall: GitHub Container Registry (GHCR)**
- **Why**: Free for public repos, unlimited private repos, integrated with GitHub workflows
- **Best for**: Open source projects, GitHub-hosted codebases
- **Limitations**: Requires GitHub account

### 🥈 **Best for Cloud Deployments**
- **AWS ECR**: If deploying to AWS (ECS, EKS, Lambda)
- **GCR/Artifact Registry**: If deploying to GCP (Cloud Run, GKE)
- **ACR**: If deploying to Azure (Container Instances, AKS)

### 🥉 **Best for Simplicity: Docker Hub**
- **Why**: Most widely used, simple setup
- **Best for**: Public images, quick sharing
- **Limitations**: Limited free private repos (1), rate limits

---

## Setup Instructions

### 1. GitHub Container Registry (GHCR) - **Recommended**

#### Setup
```bash
# Login with GitHub Personal Access Token
# Create token at: https://github.com/settings/tokens
# Required scopes: read:packages, write:packages, delete:packages
echo $GITHUB_TOKEN | docker login ghcr.io -u USERNAME --password-stdin

# Or interactively:
docker login ghcr.io
```

#### Build and Push
```bash
# Build image
docker build -t ghcr.io/elmariachi111/skin-analyzer:latest .

# Tag with version
docker tag ghcr.io/elmariachi111/skin-analyzer:latest ghcr.io/elmariachi111/skin-analyzer:v1.0.0

# Push
docker push ghcr.io/elmariachi111/skin-analyzer:latest
docker push ghcr.io/elmariachi111/skin-analyzer:v1.0.0
```

#### Pull
```bash
docker pull ghcr.io/elmariachi111/skin-analyzer:latest
```

#### Make Public/Private
- Go to: https://github.com/users/elmariachi111/packages
- Click on package → Package settings → Change visibility

#### Container Runtime Service Configuration

When configuring a container runtime service (Kubernetes, Docker, cloud services), use:

**Registry Configuration:**
- **Registry URL**: `ghcr.io`
- **Image URL**: `ghcr.io/elmariachi111/skin-analyzer:latest`
- **Username**: `elmariachi111` (your GitHub username)
- **Password**: GitHub Personal Access Token (PAT) with `read:packages` scope

**Example Configurations:**

**Kubernetes Secret:**
```bash
kubectl create secret docker-registry ghcr-secret \
  --docker-server=ghcr.io \
  --docker-username=elmariachi111 \
  --docker-password=YOUR_GITHUB_TOKEN \
  --docker-email=your-email@example.com
```

**Docker Compose:**
```yaml
services:
  skin-analyzer:
    image: ghcr.io/elmariachi111/skin-analyzer:latest
    # For private images, add:
    # environment:
    #   - DOCKER_REGISTRY_USER=elmariachi111
    #   - DOCKER_REGISTRY_PASS=${GITHUB_TOKEN}
```

**AWS ECS Task Definition:**
```json
{
  "containerDefinitions": [{
    "name": "skin-analyzer",
    "image": "ghcr.io/elmariachi111/skin-analyzer:latest",
    "repositoryCredentials": {
      "credentialsParameter": "arn:aws:secretsmanager:region:account:secret:ghcr-credentials"
    }
  }]
}
```

**Google Cloud Run / GKE:**
```bash
# Create secret
kubectl create secret docker-registry ghcr-secret \
  --docker-server=ghcr.io \
  --docker-username=elmariachi111 \
  --docker-password=YOUR_GITHUB_TOKEN
```

**Azure Container Instances / AKS:**
```bash
# Create secret
kubectl create secret docker-registry ghcr-secret \
  --docker-server=ghcr.io \
  --docker-username=elmariachi111 \
  --docker-password=YOUR_GITHUB_TOKEN
```

**Note**: For public images, authentication may not be required. For private images, always use a GitHub PAT with appropriate scopes.

---

### 2. Docker Hub

#### Setup
```bash
docker login
# Enter Docker Hub username and password
```

#### Build and Push
```bash
# Build image (must include Docker Hub username)
docker build -t YOUR_USERNAME/skin-analyzer:latest .

# Tag with version
docker tag YOUR_USERNAME/skin-analyzer:latest YOUR_USERNAME/skin-analyzer:v1.0.0

# Push
docker push YOUR_USERNAME/skin-analyzer:latest
docker push YOUR_USERNAME/skin-analyzer:v1.0.0
```

#### Pull
```bash
docker pull YOUR_USERNAME/skin-analyzer:latest
```

---

### 3. AWS ECR

#### Prerequisites
```bash
# Install AWS CLI and configure
aws configure
```

#### Setup
```bash
# Get login token
aws ecr get-login-password --region REGION | docker login --username AWS --password-stdin ACCOUNT_ID.dkr.ecr.REGION.amazonaws.com

# Create repository (if doesn't exist)
aws ecr create-repository --repository-name skin-analyzer --region REGION
```

#### Build and Push
```bash
# Build
docker build -t skin-analyzer:latest .

# Tag
docker tag skin-analyzer:latest ACCOUNT_ID.dkr.ecr.REGION.amazonaws.com/skin-analyzer:latest
docker tag skin-analyzer:latest ACCOUNT_ID.dkr.ecr.REGION.amazonaws.com/skin-analyzer:v1.0.0

# Push
docker push ACCOUNT_ID.dkr.ecr.REGION.amazonaws.com/skin-analyzer:latest
docker push ACCOUNT_ID.dkr.ecr.REGION.amazonaws.com/skin-analyzer:v1.0.0
```

---

### 4. Google Container Registry / Artifact Registry

#### Setup (Artifact Registry - recommended)
```bash
# Install gcloud CLI
# Authenticate
gcloud auth configure-docker REGION-docker.pkg.dev

# Create repository
gcloud artifacts repositories create skin-analyzer \
  --repository-format=docker \
  --location=REGION \
  --description="Skin analyzer service"
```

#### Build and Push
```bash
# Build
docker build -t skin-analyzer:latest .

# Tag
docker tag skin-analyzer:latest REGION-docker.pkg.dev/PROJECT_ID/skin-analyzer/skin-analyzer:latest
docker tag skin-analyzer:latest REGION-docker.pkg.dev/PROJECT_ID/skin-analyzer/skin-analyzer:v1.0.0

# Push
docker push REGION-docker.pkg.dev/PROJECT_ID/skin-analyzer/skin-analyzer:latest
docker push REGION-docker.pkg.dev/PROJECT_ID/skin-analyzer/skin-analyzer:v1.0.0
```

---

### 5. Azure Container Registry

#### Setup
```bash
# Login to Azure
az login

# Create registry (if needed)
az acr create --resource-group RESOURCE_GROUP --name REGISTRY_NAME --sku Basic

# Login to ACR
az acr login --name REGISTRY_NAME
```

#### Build and Push
```bash
# Build
docker build -t skin-analyzer:latest .

# Tag
docker tag skin-analyzer:latest REGISTRY_NAME.azurecr.io/skin-analyzer:latest
docker tag skin-analyzer:latest REGISTRY_NAME.azurecr.io/skin-analyzer:v1.0.0

# Push
docker push REGISTRY_NAME.azurecr.io/skin-analyzer:latest
docker push REGISTRY_NAME.azurecr.io/skin-analyzer:v1.0.0
```

---

## Best Practices

### 1. **Version Tagging Strategy**
```bash
# Use semantic versioning
docker tag skin-analyzer:latest REGISTRY/skin-analyzer:v1.0.0
docker tag skin-analyzer:latest REGISTRY/skin-analyzer:v1.0
docker tag skin-analyzer:latest REGISTRY/skin-analyzer:v1
docker tag skin-analyzer:latest REGISTRY/skin-analyzer:latest

# Or use git commit SHA
docker tag skin-analyzer:latest REGISTRY/skin-analyzer:$(git rev-parse --short HEAD)
```

### 2. **Multi-Architecture Builds** (Recommended)

**Why?** If you build on Apple Silicon (ARM64) but deploy to Linux servers (AMD64), you'll get "exec format error". Build for both architectures:

```bash
# Build for multiple platforms
docker buildx create --name multiarch --use
docker buildx build --platform linux/amd64,linux/arm64 \
  -t REGISTRY/skin-analyzer:latest \
  --push .
```

**Or use the provided script:**
```bash
./scripts/build-and-push.sh latest
```

**For AMD64 only (if you know your servers are x86_64):**
```bash
docker build --platform linux/amd64 -t REGISTRY/skin-analyzer:latest .
docker push REGISTRY/skin-analyzer:latest
```

**In docker-compose.yml, specify platform:**
```yaml
services:
  skin-analyzer:
    image: ghcr.io/elmariachi111/skin-analyzer:latest
    platform: linux/amd64  # Force AMD64 for remote servers
```

### 3. **Security**
- ✅ Use non-root user (already in Dockerfile)
- ✅ Scan images for vulnerabilities
- ✅ Use specific version tags, not just `latest`
- ✅ Keep base images updated
- ✅ Use `.dockerignore` to exclude sensitive files

### 4. **CI/CD Integration**

#### GitHub Actions Example (GHCR)
```yaml
name: Build and Push

on:
  push:
    tags:
      - 'v*'

jobs:
  build:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      
      - name: Login to GHCR
        uses: docker/login-action@v2
        with:
          registry: ghcr.io
          username: ${{ github.actor }}
          password: ${{ secrets.GITHUB_TOKEN }}
      
      - name: Set up Docker Buildx
        uses: docker/setup-buildx-action@v2
      
      - name: Build and push
        uses: docker/build-push-action@v4
        with:
          context: .
          platforms: linux/amd64,linux/arm64  # Multi-architecture support
          push: true
          tags: |
            ghcr.io/${{ github.repository_owner }}/skin-analyzer:latest
            ghcr.io/${{ github.repository_owner }}/skin-analyzer:${{ github.ref_name }}
```

---

## Quick Start Script

Create a `push-image.sh` script:

```bash
#!/bin/bash
set -e

REGISTRY=${1:-ghcr.io}
USERNAME=${2:-YOUR_USERNAME}
IMAGE_NAME="skin-analyzer"
VERSION=${3:-latest}

# Build
echo "Building image..."
docker build -t ${IMAGE_NAME}:${VERSION} .

# Tag
echo "Tagging image..."
docker tag ${IMAGE_NAME}:${VERSION} ${REGISTRY}/${USERNAME}/${IMAGE_NAME}:${VERSION}

# Push
echo "Pushing to ${REGISTRY}..."
docker push ${REGISTRY}/${USERNAME}/${IMAGE_NAME}:${VERSION}

echo "Done! Image available at: ${REGISTRY}/${USERNAME}/${IMAGE_NAME}:${VERSION}"
```

Usage:
```bash
chmod +x push-image.sh
./push-image.sh ghcr.io your-username v1.0.0
```

---

## Cost Considerations

- **Docker Hub**: Free for public, $7/month for 1 private repo, $21/month for unlimited
- **GHCR**: Free for public, free for private (GitHub Pro/Team)
- **AWS ECR**: ~$0.10 per GB/month storage, $0.01 per GB transfer
- **GCR**: ~$0.026 per GB/month storage, $0.10 per GB transfer
- **ACR**: Basic tier $5/month, Standard $50/month

---

## Recommendation for This Project

Given this is a demo project (`nilcc-demo`), I recommend:

1. **Primary**: **GitHub Container Registry (GHCR)**
   - Free for public images
   - Integrated with GitHub
   - Easy to use
   - Good for demos and open source

2. **Alternative**: **Docker Hub** if you want maximum compatibility and simplicity

Would you like me to:
- Create a GitHub Actions workflow for automated builds?
- Create a push script for your chosen registry?
- Update the README with registry information?

