@echo off
REM Windows batch file to launch the backend

REM Activate virtual environment and run
call venv\Scripts\activate.bat
python gendj.py

