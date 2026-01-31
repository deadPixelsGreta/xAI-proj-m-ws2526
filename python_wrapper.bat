@echo off
REM Wrapper script to set PYTHONPATH before running train_single_model.py
REM This ensures Python can find the experiments module when WandB runs it

setlocal enabledelayedexpansion

REM Set the project root
set "PROJECT_ROOT=C:\Users\ba081274\Documents\xAI\xAI-proj-m-ws2526"

REM Set PYTHONPATH to include the project root
set "PYTHONPATH=%PROJECT_ROOT%;%PYTHONPATH%"

REM Run Python with the train.py script, passing all arguments
python %*

endlocal
