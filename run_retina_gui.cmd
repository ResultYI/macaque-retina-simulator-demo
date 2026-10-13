@echo off
setlocal
cd /d "%~dp0"
if defined RETINA_PYTHON (
    set "RETINA_EXEC=%RETINA_PYTHON%"
) else if exist "%~dp0.venv\Scripts\python.exe" (
    set "RETINA_EXEC=%~dp0.venv\Scripts\python.exe"
) else (
    set "RETINA_EXEC=%USERPROFILE%\anaconda3\envs\retipath_v1\python.exe"
)
if not exist "%RETINA_EXEC%" (
    echo Python not found: "%RETINA_EXEC%"
    pause
    exit /b 1
)
"%RETINA_EXEC%" -B -m retina_platform --project-root "%CD%"
if errorlevel 1 pause
