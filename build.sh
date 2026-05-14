#!/bin/bash

# HippoView Docker Image Build Script with Multi-Platform Support
set -e

# Parse command line arguments
PLATFORM=${1:-"arm64"}  # Default to arm64 if no parameter provided

case $PLATFORM in
    both)
        MULTI_PLATFORM=true
        PLATFORMS="linux/amd64,linux/arm64"
        TAG_SUFFIX="latest"
        echo "* Building HippoView Docker Image (Multi-Platform: $PLATFORMS)..."
        ;;
    arm64)
        MULTI_PLATFORM=false
        PLATFORMS="linux/arm64"
        TAG_SUFFIX="arm64"
        echo "* Building HippoView Docker Image (ARM64 Platform)..."
        ;;
    amd64)
        MULTI_PLATFORM=false
        PLATFORMS="linux/amd64"
        TAG_SUFFIX="amd64"
        echo "* Building HippoView Docker Image (AMD64 Platform)..."
        ;;
    --help|-h|help)
        echo "Usage: $0 [PLATFORM]"
        echo ""
        echo "PLATFORM options:"
        echo "  both      Build for multiple platforms (linux/amd64,linux/arm64) - tag: latest"
        echo "  arm64     Build for ARM64 platform only - tag: arm64"
        echo "  amd64     Build for AMD64 platform only - tag: amd64"
        echo "  help      Show this help message"
        echo ""
        echo "Examples:"
        echo "  $0              # Build for ARM64 (default)"
        echo "  $0 arm64        # Build for ARM64"
        echo "  $0 amd64        # Build for AMD64"
        echo "  $0 both         # Build for both platforms"
        exit 0
        ;;
    *)
        echo "- ERROR: Unknown platform: $PLATFORM"
        echo "Valid options: both, arm64, amd64"
        echo "Use '$0 help' for usage information"
        exit 1
        ;;
esac

# Load environment variables from .env file
if [ -f .env ]; then
    export $(cat .env | grep -v '^#' | xargs)
else
    echo "- WARNING: .env file not found, building locally only"
fi

# Load additional variables from docker-variables.env if available
if [ -f docker-variables.env ]; then
    export $(cat docker-variables.env | grep -v '^#' | xargs)
fi

# Setup buildx for multi-platform builds or specific platform builds
if [ "$MULTI_PLATFORM" = "true" ] || [ "$PLATFORM" = "amd64" ]; then
    echo "* Setting up Docker buildx..."
    
    # Check if buildx is available
    if ! docker buildx version > /dev/null 2>&1; then
        echo "- ERROR: Docker buildx not available. Please update Docker to a newer version."
        exit 1
    fi
    
    # Create and use buildx builder if not exists
    BUILDER_NAME="hippoview-builder"
    if ! docker buildx inspect $BUILDER_NAME > /dev/null 2>&1; then
        echo "📦 Creating builder: $BUILDER_NAME"
        docker buildx create --name $BUILDER_NAME --use
    else
        echo "📦 Using existing builder: $BUILDER_NAME"
        docker buildx use $BUILDER_NAME
    fi
fi

# Check if registry variables are set for pushing
PUSH_TO_REGISTRY=false
if [ -n "$DOCKER_REGISTRY" ] && [ -n "$DOCKER_NAMESPACE" ] && [ -n "$DOCKER_REPOSITORY" ]; then
    PUSH_TO_REGISTRY=true
    echo "Registry configured: $DOCKER_REGISTRY/$DOCKER_NAMESPACE/$DOCKER_REPOSITORY"
    
    # Login to registry if credentials are provided
    if [ -n "$DOCKER_USERNAME" ] && [ -n "$DOCKER_PASSWORD" ]; then
        echo "Logging in to Container Registry..."
        if ! echo "$DOCKER_PASSWORD" | docker login --username "$DOCKER_USERNAME" "$DOCKER_REGISTRY" --password-stdin; then
            echo "Warning: Failed to login to Container Registry, will build locally only"
            PUSH_TO_REGISTRY=false
        else
            echo "Login successful"
        fi
    fi
fi

# Build the image - remote only
# LOCAL_TAG removed - building only for remote registry

if [ "$MULTI_PLATFORM" = "true" ]; then
    # Multi-platform build with buildx
    echo "🔨 Building multi-platform image for: $PLATFORMS"
    
    if [ "$PUSH_TO_REGISTRY" = "true" ]; then
        REMOTE_TAG="$DOCKER_REGISTRY/$DOCKER_NAMESPACE/$DOCKER_REPOSITORY:$TAG_SUFFIX"
        echo "* Building and pushing: $REMOTE_TAG"
        
        # Build and push directly with buildx
        docker buildx build \
            --platform "$PLATFORMS" \
            --tag "$REMOTE_TAG" \
            --push \
            -f dockerfile \
            .
        
        echo "* Successfully built and pushed multi-platform image: $REMOTE_TAG"
        echo "* Platforms: $PLATFORMS"
    else
        echo "- ERROR: Multi-platform build requires registry configuration."
        echo "   Please configure DOCKER_REGISTRY, DOCKER_NAMESPACE, DOCKER_REPOSITORY in .env file"
        exit 1
    fi
elif [ "$PLATFORM" = "amd64" ]; then
    # AMD64 build with buildx (for cross-platform compatibility)
    if [ "$PUSH_TO_REGISTRY" = "true" ]; then
        REMOTE_TAG="$DOCKER_REGISTRY/$DOCKER_NAMESPACE/$DOCKER_REPOSITORY:$TAG_SUFFIX"
        echo "🔨 Building and pushing: $REMOTE_TAG for platform: $PLATFORMS"
        
        # Build and push with buildx
        docker buildx build \
            --platform "$PLATFORMS" \
            --tag "$REMOTE_TAG" \
            --push \
            -f dockerfile \
            .
        
        echo "* Successfully built and pushed: $REMOTE_TAG"
        echo "* Platform: $PLATFORMS"
    else
        echo "- ERROR: AMD64 build requires registry configuration."
        echo "   Please configure DOCKER_REGISTRY, DOCKER_NAMESPACE, DOCKER_REPOSITORY in .env file"
        exit 1
    fi
else
    # ARM64 build with regular docker build
    if [ "$PUSH_TO_REGISTRY" = "true" ]; then
        REMOTE_TAG="$DOCKER_REGISTRY/$DOCKER_NAMESPACE/$DOCKER_REPOSITORY:$TAG_SUFFIX"
        echo "🔨 Building and pushing: $REMOTE_TAG for platform: $PLATFORMS"
        
        # Build and tag for registry
        docker build -f dockerfile -t "$REMOTE_TAG" .
        
        echo "* Pushing to registry..."
        if docker push "$REMOTE_TAG"; then
            echo "* Successfully pushed $REMOTE_TAG"
        echo "* Platform: $PLATFORMS"
        else
            echo "- ERROR: Failed to push to registry"
            exit 1
        fi
    else
        echo "- ERROR: ARM64 build requires registry configuration."
        echo "   Please configure DOCKER_REGISTRY, DOCKER_NAMESPACE, DOCKER_REPOSITORY in .env file"
        exit 1
    fi
fi

echo ""
echo "* Build complete!"
echo "* Image pushed to: $REMOTE_TAG"
echo ""
echo "* Usage Examples:"
echo "  # Pull and run services:"
echo "  docker run -p 8501:8501 $REMOTE_TAG                              # Streamlit"
echo "  docker run -p 8000:8000 $REMOTE_TAG /app/entrypoints/mcp.sh      # MCP Server"
echo "  docker run $REMOTE_TAG /app/entrypoints/agent.sh                 # Agent"
echo "  docker run $REMOTE_TAG /app/entrypoints/kg.sh --config config.yaml  # Knowledge Graph"
echo ""
echo "  # Build options:"
echo "  ./build.sh                                                       # ARM64 build (default)"
echo "  ./build.sh arm64                                                 # ARM64 build"
echo "  ./build.sh amd64                                                 # AMD64 build"
echo "  ./build.sh both                                                  # Multi-platform build"
echo "  ./build.sh help                                                  # Show help"
