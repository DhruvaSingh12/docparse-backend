# 1. Navigate to your project
cd F:\Projects\docparse\backend

# 2. Activate virtual environment
.\.venv\Scripts\Activate.ps1

# 3. Start the server
python -m uvicorn app.main:app --port 8000 --host 127.0.0.1