#!/bin/bash
echo ">>> TÜM SÜREÇLER SONLANDIRILIYOR..."
pkill -f "ollama"
pkill -f "uvicorn"
pkill -f "python.*main:app"

