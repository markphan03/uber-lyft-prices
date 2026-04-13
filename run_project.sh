#!/usr/bin/env bash

set -e  # stop on error

echo "Setting up virtual environment..."
python -m venv .venv

echo "Activating virtual environment..."
source .venv/bin/activate

echo "Installing dependencies..."
pip install -r requirements.txt

# =========================

# CONDITIONAL PIPELINE

# =========================

if [ ! -f "train_test_split.npz" ]; then
echo "train_test_split.npz not found → running preprocess_data.py"
python preprocess_data.py
fi

if [ ! -f "nn_checkpoint.pth" ]; then
echo "nn_checkpoint not found → running nn_training.py"
python nn_training.py
fi

echo "Running prediction..."
python prediction.py

echo "Done."
