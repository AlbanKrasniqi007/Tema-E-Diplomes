@echo off
setlocal
set PYTHONUNBUFFERED=1
cd /d "%~dp0"
python -u "%~dp0scripts\uci.py" --config "%~dp0configs\default.yaml"
