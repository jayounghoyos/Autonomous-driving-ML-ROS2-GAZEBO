#!/bin/bash
# Configure Isaac Sim to use cloud assets
# This script updates the user.config.json file

CONFIG_FILE="$HOME/.local/share/ov/data/Kit/Isaac-Sim Full/5.0/user.config.json"

echo "Isaac Sim Cloud Assets Configuration"
echo "====================================="
echo ""
echo "Config file: $CONFIG_FILE"
echo ""

# Check if file exists
if [ ! -f "$CONFIG_FILE" ]; then
    echo "❌ Config file not found!"
    echo "Creating new config file..."
    mkdir -p "$(dirname "$CONFIG_FILE")"
fi

# Backup existing config
if [ -f "$CONFIG_FILE" ]; then
    cp "$CONFIG_FILE" "$CONFIG_FILE.backup"
    echo "✓ Backed up existing config to $CONFIG_FILE.backup"
fi

# Create new config with cloud assets
cat > "$CONFIG_FILE" << 'EOF'
{
    "persistent": {
        "isaac": {
            "asset_root": {
                "default": "http://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/Isaac/4.0"
            }
        }
    }
}
EOF

echo "✓ Config file updated!"
echo ""
echo "Cloud asset path configured:"
echo "  http://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/Isaac/4.0"
echo ""
echo "✅ Configuration complete!"
echo ""
echo "Next steps:"
echo "1. Test asset loading:"
echo "   source setup_isaac_env.sh"
echo "   \$ISAAC_PYTHON isaac_worlds/test_assets.py"
echo ""
