@echo off
setlocal
cd /d "%~dp0"
if defined RETINA_PYTHON (
    set "RETINA_REPLAY_EXEC=%RETINA_PYTHON%"
) else if exist "%~dp0..\macaque_retina_simulator_envs\cu126_py311_v1\Scripts\python.exe" (
    set "RETINA_REPLAY_EXEC=%~dp0..\macaque_retina_simulator_envs\cu126_py311_v1\Scripts\python.exe"
) else if exist "%~dp0.venv\Scripts\python.exe" (
    set "RETINA_REPLAY_EXEC=%~dp0.venv\Scripts\python.exe"
) else (
    echo Set RETINA_PYTHON to the existing Python environment.
    pause
    exit /b 1
)
"%RETINA_REPLAY_EXEC%" -B -X utf8 -m retina_platform --project-root "%CD%" --cell6-replay
if errorlevel 1 pause
