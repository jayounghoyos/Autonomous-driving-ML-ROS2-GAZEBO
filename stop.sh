#!/bin/bash
# Stop the Kaiju development container

echo "🛑 Stopping Kaiju container..."
docker compose down

echo "✅ Container stopped."
echo ""
echo "To remove all build volumes (clean slate):"
echo "  docker compose down -v"
