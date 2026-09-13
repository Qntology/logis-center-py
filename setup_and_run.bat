@echo off
chcp 65001 >nul
title Vision NMS Crop - Setup
setlocal enabledelayedexpansion

echo ============================================================
echo   Vision NMS Crop - Full Auto Setup
echo ============================================================
echo.

REM ── 1. Python 감지 (3.10~3.12) ──────────────────────────────
set PYTHON_CMD=

py -3.12 --version >nul 2>&1 && set PYTHON_CMD=py -3.12 && goto :found
py -3.11 --version >nul 2>&1 && set PYTHON_CMD=py -3.11 && goto :found
py -3.10 --version >nul 2>&1 && set PYTHON_CMD=py -3.10 && goto :found

echo [ERROR] Python 3.10~3.12 not found.
echo Install from: https://www.python.org/downloads/release/python-31210/
pause
exit /b 1

:found
for /f "tokens=2" %%v in ('%PYTHON_CMD% --version 2^>^&1') do set PYVER=%%v
echo [OK] Python %PYVER% detected

REM ── 2. venv 생성/활성화 ─────────────────────────────────────
if not exist "%~dp0venv" (
    echo [INFO] Creating venv...
    %PYTHON_CMD% -m venv "%~dp0venv"
)
call "%~dp0venv\Scripts\activate.bat"

echo.
echo ============================================================
echo   [STEP 1/5] Upgrade pip
echo ============================================================
python -m pip install --upgrade pip >nul 2>&1

echo.
echo ============================================================
echo   [STEP 2/5] Install PyTorch (GPU auto-detect)
echo ============================================================

set GPU_TYPE=cpu

nvidia-smi >nul 2>&1
if %errorlevel% equ 0 (
    set GPU_TYPE=cuda
    echo [GPU] NVIDIA GPU detected:
    nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>nul
    goto :install_torch
)

wmic path win32_videocontroller get name 2>nul | findstr /i "Radeon" >nul 2>&1
if %errorlevel% equ 0 (
    echo [GPU] AMD GPU detected (ROCm not supported on Windows, using CPU)
    goto :install_torch
)

echo [GPU] No GPU detected. Using CPU.

:install_torch
if "%GPU_TYPE%"=="cuda" (
    echo [INFO] Installing PyTorch with CUDA 12.1...
    pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121 --no-deps
    if !errorlevel! equ 0 (
        echo [OK] PyTorch CUDA 12.1 installed.
    ) else (
        echo [WARN] CUDA install failed. Falling back to CPU.
        pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu --no-deps
    )
) else (
    echo [INFO] Installing PyTorch CPU build...
    pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu --no-deps
)

REM ✅ 수정: total_mem → total_memory
echo.
python -c "import torch; print(f'  PyTorch: {torch.__version__}'); print(f'  CUDA available: {torch.cuda.is_available()}'); print(f'  GPU: {torch.cuda.get_device_name(0)}' if torch.cuda.is_available() else '  GPU: N/A (CPU mode)'); print(f'  VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB' if torch.cuda.is_available() else '')"

echo.
echo ============================================================
echo   [STEP 3/5] Install remaining packages
echo ============================================================
pip install -r "%~dp0requirements.txt"

if %errorlevel% neq 0 (
    echo [ERROR] Package installation failed.
    pause
    exit /b 1
)

echo.
echo ============================================================
echo   [STEP 4/5] Low-VRAM support (optional)
echo ============================================================

for /f "tokens=*" %%v in ('python -c "import torch;print(int(torch.cuda.get_device_properties(0).total_memory/1024**3)) if torch.cuda.is_available() else print(0)" 2^>nul') do set VRAM_GB=%%v
if "%VRAM_GB%"=="" set VRAM_GB=0

echo [INFO] Detected VRAM: %VRAM_GB% GB

if %VRAM_GB% GTR 0 if %VRAM_GB% LSS 6 (
    echo [INFO] Low VRAM detected. Installing bitsandbytes for 4-bit quantization...
    pip install bitsandbytes >nul 2>&1
    if !errorlevel! equ 0 (
        echo   [OK] bitsandbytes installed. 2B LLM will run in 4-bit.
    ) else (
        echo   [SKIP] bitsandbytes install failed. CPU offload will be used instead.
    )
) else (
    echo [INFO] Sufficient VRAM. Skipping quantization package.
)

echo.
echo ============================================================
echo   [STEP 5/5] Verify
echo ============================================================
python -c "import torch; print(f'  PyTorch: {torch.__version__} | CUDA: {torch.cuda.is_available()}')"
python -c "import transformers; print(f'  transformers: {transformers.__version__}')"
python -c "import numpy; print(f'  numpy: {numpy.__version__}')"
python -c "import PIL; print(f'  pillow: {PIL.__version__}')"

python -c "import webview; print(f'  pywebview: {webview.__version__}')" 2>nul
if !errorlevel! neq 0 echo   pywebview: NOT INSTALLED

python -c "import stanza; print(f'  stanza: {stanza.__version__}')" 2>nul
if !errorlevel! neq 0 echo   stanza: NOT INSTALLED (NLP gate disabled)

python -c "import bitsandbytes; print('  bitsandbytes: OK (4-bit available)')" 2>nul
if !errorlevel! neq 0 echo   bitsandbytes: not installed (offload fallback)

python -c "import fitz; print('  PyMuPDF: OK')" 2>nul
if !errorlevel! neq 0 echo   PyMuPDF: NOT INSTALLED (PDF input disabled)

echo.
echo ============================================================
echo   Starting app...
if "%GPU_TYPE%"=="cuda" (
    echo   [GPU] CUDA acceleration ENABLED
) else (
    echo   [GPU] CPU mode
)
echo ============================================================
echo.

echo.
echo ============================================================
echo   Debugging
echo     Remote DevTools : http://127.0.0.1:9222  (Chrome)
echo     In-app console  : Ctrl+Shift+D
echo     Disable remote  : setup_and_run.bat --no-devtools
echo ============================================================
echo.

python "%~dp0app.py" %*

if %errorlevel% neq 0 (
    echo.
    echo [ERROR] App crashed
    pause
)

endlocal