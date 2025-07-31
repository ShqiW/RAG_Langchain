#!/bin/bash

# M3E Model Download Script
echo "Starting to download M3E-base model..."

# Activate virtual environment
source rag_env/bin/activate

# Check if necessary packages are installed
python -c "import sentence_transformers" 2>/dev/null || {
    echo "Installing sentence-transformers..."
    pip install sentence-transformers
}

# Create model directory
mkdir -p ./AI-ModelScope/m3e-base

# Download model
echo "Download M3E-base model to ./AI-ModelScope/m3e-base"
python -c "
import os
from sentence_transformers import SentenceTransformer

# Download model
print('Downloading model...')
model = SentenceTransformer('moka-ai/m3e-base')

# Save to local
print('Saving model to local...')
model.save('./AI-ModelScope/m3e-base')

# Verify
print('验证模型...')
test_model = SentenceTransformer('./AI-ModelScope/m3e-base')
test_embedding = test_model.encode('Test text')
print(f'Model verification successful! Embedding dimension: {test_embedding.shape}')
"

if [ $? -eq 0 ]; then
            echo "✅ Model download completed!"
    echo "Model saved in: ./AI-ModelScope/m3e-base"
    echo ""
    echo "Now you can run RAG system:"
    echo "  ./02_run_rag.sh"
else
    echo "❌ Model download failed"
    echo "Please check network connection or disk space"
fi

# Exit virtual environment
deactivate 