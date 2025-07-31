#!/bin/bash

# RAG Environment Setup
echo "Setting up LangChain RAG environment..."

# 检查Python版本
python_version=$(python3 --version 2>&1 | awk '{print $2}' | cut -d. -f1,2)
echo "Detected Python version: $python_version"

# Create virtual environment
echo "Creating virtual environment..."
python3 -m venv rag_env

# Activate virtual environment
echo "Activating virtual environment..."
source rag_env/bin/activate

# Upgrade pip   
echo "Upgrading pip..."
pip install --upgrade pip

# Install dependencies
echo "Installing dependencies..."
pip install -r requirements.txt

echo "✓ Environment setup completed!"
echo ""
echo "Usage:"
echo "1. Run RAG system: ./run_rag.sh"
echo "2. Manually activate environment: source rag_env/bin/activate"
echo "3. Manually run: python rag_implementation.py"
echo "4. Exit environment: deactivate" 