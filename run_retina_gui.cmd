@echo off
setlocal
cd /d "%~dp0"
rem Refresh cache paths when the parent terminal predates the user settings.
for %%V in (PIP_CACHE_DIR TORCH_HOME) do (
    if not defined %%V for /f "tokens=2,*" %%A in ('reg query "HKCU\Environment" /v %%V 2^>nul') do set "%%V=%%B"
)
if defined RETINA_PYTHON (
    set "RETINA_EXEC=%RETINA_PYTHON%"
) else if exist "%~dp0..\macaque_retina_simulator_envs\cu126_py311_v1\Scripts\python.exe" (
    set "RETINA_EXEC=%~dp0..\macaque_retina_simulator_envs\cu126_py311_v1\Scripts\python.exe"
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
"%RETINA_EXEC%" -B -X utf8 -m retina_platform --project-root "%CD%" --response-replay
if errorlevel 1 pause
