#!/bin/bash

# Load environment variables from .env file
if [ -f .env ]; then
    export $(cat .env | grep -v '^#' | xargs)
else
    echo "Error: .env file not found"
    exit 1
fi

# Check if DOCKER_REGISTRY is set
if [ -z "$DOCKER_REGISTRY" ]; then
    echo "Error: DOCKER_REGISTRY not set in .env file"
    exit 1
fi

# Check if DOCKER_NAMESPACE is set
if [ -z "$DOCKER_NAMESPACE" ]; then
    echo "Error: DOCKER_NAMESPACE not set in .env file"
    exit 1
fi

# Check if DOCKER_REPOSITORY is set
if [ -z "$DOCKER_REPOSITORY" ]; then
    echo "Error: DOCKER_REPOSITORY not set in .env file"
    exit 1
fi

# Check if DOCKER_USERNAME is set
if [ -z "$DOCKER_USERNAME" ]; then
    echo "Error: DOCKER_USERNAME not set in .env file"
    exit 1
fi

# Check if DOCKER_PASSWORD is set
if [ -z "$DOCKER_PASSWORD" ]; then
    echo "Error: DOCKER_PASSWORD not set in .env file"
    exit 1
fi

# Login to Aliyun Container Registry
echo "Logging in to Container Registry..."
if ! echo "$DOCKER_PASSWORD" | docker login --username "$DOCKER_USERNAME" "$DOCKER_REGISTRY" --password-stdin; then
    echo "Error: Failed to login to Container Registry"
    exit 1
fi
echo "Login successful"

# Define the base image tag
PUBLIC_BASE_IMAGE="andimajore/biocyper_base:python3.10"
PLATFORM="amd64"
# Detect platform or use specified platform
if [ -z "$PLATFORM" ]; then
    ARCH=$(uname -m)
    if [ "$ARCH" == "x86_64" ]; then
        PLATFORM="amd64"
    elif [ "$ARCH" == "arm64" ] || [ "$ARCH" == "aarch64" ]; then
        PLATFORM="arm64"
    else
        echo "Unsupported architecture: $ARCH"
        exit 1
    fi
fi

echo "Building for platform: $PLATFORM"
BASE_IMAGE="$DOCKER_REGISTRY/$DOCKER_NAMESPACE/biocypher:$PLATFORM"

# Check if base image exists in registry
echo "Checking if base image exists..."
if ! docker manifest inspect "$BASE_IMAGE" &> /dev/null; then
    echo "Base image not found, creating it..."

    if [ "$PLATFORM" == "amd64" ]; then
        # create AMD64 image
        echo "FROM $DOCKER_REGISTRY/$DOCKER_NAMESPACE/biocypher:base" > Dockerfile.multiarch

        # Remove the existing builder if it exists
        docker buildx rm multiarch 2>/dev/null || true

        # Setup buildx
        docker buildx create --name multiarch --use

        # Build and push for AMD64
        if ! docker buildx build --platform linux/amd64 -t "$BASE_IMAGE" -f Dockerfile.multiarch --push .; then
            echo "Error: Failed to build and push amd64 base image"
            exit 1
        else
            echo "Successfully built and pushed amd64 base image to registry"
            rm Dockerfile.multiarch
        fi

        
    elif [ "$PLATFORM" == "arm64" ]; then
        # create ARM64 image
        echo "Pulling public biocypher image for ARM64..."
        if ! docker pull --platform linux/arm64 "$PUBLIC_BASE_IMAGE"; then
            echo "Error: Failed to pull public biocypher ARM64 image"
            exit 1
        fi
        
        # Tag it for your registry
        docker tag "$PUBLIC_BASE_IMAGE" "$BASE_IMAGE"
        
        # Push the base image
        if ! docker push "$BASE_IMAGE"; then
            echo "Error: Failed to push ARM64 base image to registry"
            exit 1
        fi
    else
        echo "Invalid platform: $PLATFORM"
        exit 1
    fi
    
    echo "Successfully created and pushed base image to registry"
else
    echo "Base image already exists in the registry"
fi

# Build the knowledge graph image
echo "Building knowledge graph image..."
if ! docker buildx build --platform linux/$PLATFORM \
    --build-arg BASE_IMAGE="$BASE_IMAGE" \
    -f Dockerfile.kg \
    -t biocypher-kg \
    --load \
    --no-cache .; then
    echo "Error: Failed to build knowledge graph image"
    exit 1
fi

# Tag the image for pushing
FULL_TAG="$DOCKER_REGISTRY/$DOCKER_NAMESPACE/$DOCKER_REPOSITORY:$PLATFORM"
docker tag biocypher-kg "$FULL_TAG"

# Push the image
echo "Pushing knowledge graph image to registry..."
if ! docker push "$FULL_TAG"; then
    echo "Error: Failed to push knowledge graph image to registry"
    exit 1
fi

# Verify the image was pushed successfully
echo "Verifying image was pushed successfully..."
if ! docker manifest inspect "$FULL_TAG" &> /dev/null; then
    echo "Error: Failed to verify knowledge graph image in registry"
    exit 1
else
    echo "Successfully built and pushed knowledge graph image to Container Registry"
    echo "Image: $FULL_TAG"
fi

