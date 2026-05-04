#!/usr/bin/env bash
set -e

echo "=============================================="
echo " PyElastica environment setup"
echo "=============================================="

# Go to the folder where this launcher is located
cd "$(dirname "$0")"

echo ""
echo "Working directory:"
pwd

echo ""
echo "Checking Python..."
if command -v python3 >/dev/null 2>&1; then
    PYTHON_CMD="python3"
elif command -v python >/dev/null 2>&1; then
    PYTHON_CMD="python"
else
    echo "ERROR: Python is not installed or not in PATH."
    echo "Install Python 3.9+ first, then run this again."
    exec "${SHELL:-/bin/bash}"
fi

$PYTHON_CMD --version

echo ""
echo "Creating virtual environment: .venv"
$PYTHON_CMD -m venv .venv

echo ""
echo "Activating virtual environment..."
source .venv/bin/activate

echo ""
echo "Upgrading pip..."
python -m pip install --upgrade pip setuptools wheel

echo ""
echo "Installing requirements..."
python -m pip install -r requirements.txt

echo ""
echo "=============================================="
echo " Environment ready."
echo "=============================================="
echo ""
echo "Virtual environment is active."
echo ""
echo "You can now run your simulation, for example:"
echo ""
echo "python multi_wrap_unwinding.py \\"
echo "  --lambda-bend 2.0 \\"
echo "  --gamma-twist 1.0 \\"
echo "  --elongation 0.6 \\"
echo "  --output-dir outputs"
echo ""
echo "To leave the environment later, type:"
echo "deactivate"
echo ""

# Keep terminal open with the virtual environment active
exec "${SHELL:-/bin/bash}"
