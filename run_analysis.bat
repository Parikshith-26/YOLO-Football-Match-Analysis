@echo off
echo Installing dependencies...
py -3.12 -m pip install -r requirements.txt
echo.
echo Starting Football Analysis...
py -3.12 main.py
pause
