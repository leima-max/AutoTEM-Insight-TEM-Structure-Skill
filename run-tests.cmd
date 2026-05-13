@echo off
setlocal
set "PYEXE=%~dp0.venv\Scripts\python.exe"
if not exist "%PYEXE%" (
  echo Project virtual environment not found at .venv
  echo Create it first: python -m venv .venv
  exit /b 1
)
"%PYEXE%" -m pytest %*
