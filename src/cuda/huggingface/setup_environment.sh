#!/bin/bash
# Setup environment for running huggingface examples

# Check if we are already in a virtual environment
if [ -n "$VIRTUAL_ENV" ]; then
    echo "Already in a virtual environment, skipping setup"
    exit 0
fi

# Get the location of this script when sourcing
if test -n "$BASH" ; then SCRIPT_LOC=$BASH_SOURCE
elif test -n "$ZSH_NAME" ; then SCRIPT_LOC=${(%):-%x}
else
    echo "WARNING this script only tested with bash and zsh, use with caution with your shell at $SHELL"
    if test -n "$TMOUT"; then SCRIPT_LOC=${.sh.file}
    elif test ${0##*/} = dash; then x=$(lsof -p $$ -Fn0 | tail -1); SCRIPT_LOC=${x#n}
    elif test -n "$FISH_VERSION" ; then SCRIPT_LOC=(status current-filename)
    else echo "ERROR unknown shell, cannot determine script location" && return 1
    fi
fi

# Get the directory of the script
SCRIPT_DIR=${SCRIPT_LOC%/*}

# Check if virtual environment exists
if [ ! -d "$SCRIPT_DIR/.venv" ]; then
    # Create virtual environment
    python3 -m venv $SCRIPT_DIR/.venv
    
    # Activate virtual environment
    source $SCRIPT_DIR/.venv/bin/activate
    
    # Install dependencies
    pip install -r $SCRIPT_DIR/requirements.txt
else
    # Activate virtual environment
    source $SCRIPT_DIR/.venv/bin/activate
fi

# Permission for python scripts
chmod u+x $SCRIPT_DIR/*.py

echo "Environment setup complete"
