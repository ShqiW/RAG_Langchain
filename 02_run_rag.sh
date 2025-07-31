#!/bin/bash

# RAG System Startup Script
echo "Starting LangChain RAG system..."

# Check if virtual environment exists
if [ ! -d "rag_env" ]; then
    echo "Error: Virtual environment does not exist, please run setup_env.sh first"
    exit 1
fi

# Activate virtual environment
source rag_env/bin/activate

# Check Python and dependencies
echo "Checking environment..."
python -c "import langchain, langchain_openai, chromadb, sentence_transformers; print('✓ All dependencies installed')"

# Run RAG system
echo "Starting RAG system..."
python rag_implementation.py

# Exit virtual environment
deactivate
echo "RAG system exited" 