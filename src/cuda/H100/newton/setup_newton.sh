#!/bin/bash
# Setup Newton environment (similar to huggingface/setup_environment.sh)

NEWTON_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
NEWTON_SUBMODULE="$NEWTON_DIR/../external/newton"

# Create Python virtual environment if it doesn't exist
if [ ! -d "$NEWTON_DIR/newton_venv" ]; then
    echo "Creating Python virtual environment for Newton..."
    python3 -m venv "$NEWTON_DIR/newton_venv"
    source "$NEWTON_DIR/newton_venv/bin/activate"

    # Install Newton and dependencies
    pip install --upgrade pip
    if [ -d "$NEWTON_SUBMODULE" ]; then
        echo "Installing Newton from submodule..."
        pip install -e "$NEWTON_SUBMODULE"
        # Install additional dependencies for robot and USD examples
        pip install usd-core mujoco-warp
    else
        echo "WARNING: Newton submodule not found at $NEWTON_SUBMODULE"
        echo "Run: git submodule update --init --recursive"
    fi
else
    source "$NEWTON_DIR/newton_venv/bin/activate"
fi

export NEWTON_ENV="$NEWTON_DIR/newton_venv"
export PYTHONPATH="$NEWTON_SUBMODULE:$PYTHONPATH"
