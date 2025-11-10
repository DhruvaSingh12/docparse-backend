@echo off
cd /d F:\Projects\docparse\backend
call .venv\Scripts\activate.bat
echo Virtual environment activated
echo Starting DocParse API server...
echo.
echo NOTE: Using --reload for development
echo If you experience reload loops, use: python -m uvicorn app.main:app --port 8000
echo.
python -m uvicorn app.main:app --reload --port 8000 --reload-exclude="*.pt" --reload-exclude="*.pth" --reload-exclude="*.bin" --reload-exclude="*.safetensors" --reload-exclude="__pycache__" --reload-exclude=".venv" --reload-exclude="dataset" --reload-exclude="runs"
pause