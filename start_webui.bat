@echo off
cd /d %~dp0
echo Starting 3D Scene Validator on http://localhost:5050
python -m uvicorn webui.validate_server:app --port 5050 --reload
pause
