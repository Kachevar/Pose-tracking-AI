@echo off
setlocal enabledelayedexpansion

REM
FOR /F "delims=" %%i IN ('python -c "import site; print(site.USER_BASE)"') DO set "SCRIPTS_PATH=%%i\Scripts"

REM
echo Check availability Scripts-paths в PATH...
echo %PATH% | findstr /C:"%SCRIPTS_PATH%" >nul
if errorlevel 1 (
    echoWe add it temporarily PATH: %SCRIPTS_PATH%
    set "PATH=%PATH%;%SCRIPTS_PATH%"
) else (
    echo PATH already contains Scripts
)

REM
echo Installing dependencies...
pip install --disable-pip-version-check --quiet numpy opencv-python ultralytics

REM
echo Run Pose_AI.py...
python Pose_AI.py

echo.
pause
