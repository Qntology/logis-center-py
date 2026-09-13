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
echo   [STEP 1/6] Upgrade pip
echo ============================================================
python -m pip install --upgrade pip >nul 2>&1

echo.
echo ============================================================
echo   [STEP 2/6] Install PyTorch (GPU auto-detect)
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
echo   [STEP 3/6] Install remaining packages
echo ============================================================
pip install -r "%~dp0requirements.txt"

if %errorlevel% neq 0 (
    echo [ERROR] Package installation failed.
    pause
    exit /b 1
)

echo.
echo ============================================================
echo   [STEP 4/6] Install PaddleOCR backend (PP-OCRv5)
echo ============================================================
echo [INFO] paddlepaddle is NOT on PyPI. Using the official index.

set PADDLE_PKG=paddlepaddle
set PADDLE_INDEX=https://www.paddlepaddle.org.cn/packages/stable/cpu/

if "%GPU_TYPE%"=="cuda" (
    for /f "tokens=*" %%c in ('python -c "import torch;v=(torch.version.cuda or '');p=v.split('.');print(p[0]+'.'+p[1] if len(p)^>1 else '')" 2^>nul') do set CUDA_VER=%%c
    if "!CUDA_VER!"=="12.9" (
        set PADDLE_PKG=paddlepaddle-gpu
        set PADDLE_INDEX=https://www.paddlepaddle.org.cn/packages/stable/cu129/
    ) else if "!CUDA_VER!"=="12.6" (
        set PADDLE_PKG=paddlepaddle-gpu
        set PADDLE_INDEX=https://www.paddlepaddle.org.cn/packages/stable/cu126/
    ) else if "!CUDA_VER!"=="12.1" (
        set PADDLE_PKG=paddlepaddle-gpu
        set PADDLE_INDEX=https://www.paddlepaddle.org.cn/packages/stable/cu118/
    ) else if "!CUDA_VER!"=="11.8" (
        set PADDLE_PKG=paddlepaddle-gpu
        set PADDLE_INDEX=https://www.paddlepaddle.org.cn/packages/stable/cu118/
    )
)

echo [INFO] Package : !PADDLE_PKG!
echo [INFO] Index   : !PADDLE_INDEX!

python -c "import paddle" >nul 2>&1
if !errorlevel! equ 0 (
    echo   [SKIP] paddle already installed.
) else (
    pip install !PADDLE_PKG! -i !PADDLE_INDEX! --trusted-host www.paddlepaddle.org.cn
    if !errorlevel! neq 0 (
        echo   [WARN] !PADDLE_PKG! install failed. Retrying with CPU index...
        pip install paddlepaddle -i https://www.paddlepaddle.org.cn/packages/stable/cpu/ --trusted-host www.paddlepaddle.org.cn
    )
)

python -c "import paddleocr" >nul 2>&1
if !errorlevel! equ 0 (
    echo   [SKIP] paddleocr already installed.
) else (
    pip install paddleocr
)

python -c "import paddle, paddleocr; print('  [OK] PaddleOCR backend ready')" 2>nul
if !errorlevel! neq 0 (
    echo   [WARN] PaddleOCR backend unavailable.
    echo          OCR drafts will be skipped; the VLM will read crops directly.
    echo          Manual install:
    echo            pip install paddlepaddle -i https://www.paddlepaddle.org.cn/packages/stable/cpu/
    echo            pip install paddleocr
)

echo.
echo ============================================================
echo   [STEP 5/6] Low-VRAM support (optional)
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
echo   [STEP 6/6] Verify
echo ============================================================
python -c "import torch; print(f'  PyTorch: {torch.__version__} | CUDA: {torch.cuda.is_available()}')"
python -c "import transformers; print(f'  transformers: {transformers.__version__}')"
python -c "import numpy; print(f'  numpy: {numpy.__version__}')"
python -c "import PIL; print(f'  pillow: {PIL.__version__}')"

python -c "import paddle; print(f'  paddlepaddle: {paddle.__version__}')" 2>nul
if !errorlevel! neq 0 echo   paddlepaddle: NOT INSTALLED (OCR draft disabled)

python -c "import paddleocr; print(f'  paddleocr: {paddleocr.__version__}')" 2>nul
if !errorlevel! neq 0 echo   paddleocr: NOT INSTALLED (OCR draft disabled)

python -c "import os;os.environ.setdefault('FLAGS_use_mkldnn','0');from paddleocr import TextDetection;import numpy as np;d=TextDetection(model_name='PP-OCRv5_mobile_det',device='cpu',enable_mkldnn=False);list(d.predict(input=[np.full((96,320,3),255,dtype='uint8')]));print('  PP-OCRv5 det: OK (oneDNN off)')" 2>nul
if !errorlevel! neq 0 echo   PP-OCRv5 det: selftest failed (image-processing fallback will be used)

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

if "%FLAGS_use_mkldnn%"=="" set FLAGS_use_mkldnn=0
if "%FLAGS_call_stack_level%"=="" set FLAGS_call_stack_level=1
if "%GLOG_minloglevel%"=="" set GLOG_minloglevel=2
if "%NMS_PADDLE_DEVICE%"=="" set NMS_PADDLE_DEVICE=cpu

echo.
echo ============================================================
echo   Debugging
echo     Remote DevTools : http://127.0.0.1:9222  (Chrome)
echo     In-app console  : Ctrl+Shift+D
echo     Disable remote  : setup_and_run.bat --no-devtools
echo   PaddleOCR
echo     Auto install    : ON  (disable: --no-paddle-install)
echo     Env override    : set NMS_PADDLE_AUTOINSTALL=0
echo     Device          : %NMS_PADDLE_DEVICE%
echo     oneDNN          : OFF (FLAGS_use_mkldnn=%FLAGS_use_mkldnn%)
echo                       PP-OCRv5 det has a double-array attribute
echo                       the oneDNN PIR executor cannot convert.
echo                       Force on with: --paddle-mkldnn
echo ============================================================
echo.

python "%~dp0app.py" %*

if %errorlevel% neq 0 (
    echo.
    echo [ERROR] App crashed
    pause
)

endlocal