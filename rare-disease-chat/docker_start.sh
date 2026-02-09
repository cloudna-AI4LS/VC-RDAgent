#!/bin/bash
# Quick start script for Docker all-in-one container
# Usage: ./docker_start.sh [build|start|stop|logs|restart]

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

IMAGE_NAME="rarellm-all-in-one:latest"
CONTAINER_NAME="rarellm-all"
MCP_PORT="${MCP_PORT:-3000}"
DASHBOARD_PORT="${DASHBOARD_PORT:-8080}"

build() {
    echo "=== Building Docker image ==="
    docker build -t "$IMAGE_NAME" .
    echo "✓ Build completed"
}

start() {
    echo "=== Starting container ==="
    
    # Check if container already exists
    if docker ps -a --format '{{.Names}}' | grep -q "^${CONTAINER_NAME}$"; then
        if docker ps --format '{{.Names}}' | grep -q "^${CONTAINER_NAME}$"; then
            echo "Container is already running"
            return
        else
            echo "Removing stopped container..."
            docker rm "$CONTAINER_NAME"
        fi
    fi
    
    docker run -d \
        --name "$CONTAINER_NAME" \
        -p "${MCP_PORT}:${MCP_PORT}" \
        -p "${DASHBOARD_PORT}:${DASHBOARD_PORT}" \
        -e MCP_PORT="${MCP_PORT}" \
        -e DASHBOARD_PORT="${DASHBOARD_PORT}" \
        "$IMAGE_NAME"
    
    echo "✓ Container started"
    echo ""
    echo "MCP Server: http://localhost:${MCP_PORT}"
    echo "Dashboard: http://localhost:${DASHBOARD_PORT}/rdagent/"
    echo ""
    echo "View logs: ./docker_start.sh logs"
}

stop() {
    echo "=== Stopping container ==="
    docker stop "$CONTAINER_NAME" 2>/dev/null || echo "Container not running"
    docker rm "$CONTAINER_NAME" 2>/dev/null || echo "Container not found"
    echo "✓ Container stopped"
}

logs() {
    echo "=== Container logs ==="
    docker logs -f "$CONTAINER_NAME" 2>/dev/null || echo "Container not found"
}

restart() {
    stop
    sleep 2
    start
}

case "${1:-start}" in
    build)
        build
        ;;
    start)
        start
        ;;
    stop)
        stop
        ;;
    logs)
        logs
        ;;
    restart)
        restart
        ;;
    *)
        echo "Usage: $0 [build|start|stop|logs|restart]"
        echo ""
        echo "Commands:"
        echo "  build   - Build Docker image"
        echo "  start   - Start container (default)"
        echo "  stop    - Stop container"
        echo "  logs    - View container logs"
        echo "  restart - Restart container"
        echo ""
        echo "Environment variables:"
        echo "  MCP_PORT       - MCP Server port (default: 3000)"
        echo "  DASHBOARD_PORT - Dashboard port (default: 8080)"
        exit 1
        ;;
esac
