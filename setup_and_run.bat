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
echo   [STEP 5/6] Memory check
echo ============================================================

for /f "tokens=*" %%m in ('python -c "import ctypes;class M(ctypes.Structure):_fields_=[('a',ctypes.c_ulong),('b',ctypes.c_ulong),('c',ctypes.c_ulonglong),('d',ctypes.c_ulonglong),('e',ctypes.c_ulonglong),('f',ctypes.c_ulonglong),('g',ctypes.c_ulonglong),('h',ctypes.c_ulonglong),('i',ctypes.c_ulonglong)]" 2^>nul') do set _DUMMY=%%m

python -c "from core.memory import ram_info,usable_ram_gb;i=ram_info();print(f'  Total RAM     : {i[\"total_gb\"]:.1f} GB');print(f'  Available     : {i[\"available_gb\"]:.1f} GB');print(f'  Commit free   : {i[\"commit_available_gb\"]:.1f} GB');print(f'  Usable        : {usable_ram_gb():.1f} GB')" 2>nul
if !errorlevel! neq 0 echo   [WARN] memory probe failed

for /f "tokens=*" %%r in ('python -c "from core.memory import usable_ram_gb;print(int(usable_ram_gb()))" 2^>nul') do set RAM_GB=%%r
if "%RAM_GB%"=="" set RAM_GB=0

if %RAM_GB% GTR 0 if %RAM_GB% LSS 11 (
    echo.
    echo   [WARN] Usable RAM is %RAM_GB% GB.
    echo          Qwen3.5-4B weights are staged in system RAM before
    echo          they reach the GPU, even with 4-bit quantization.
    echo          The app will refuse to load the refiner instead of
    echo          being killed by the OS.
    echo.
    echo          Options:
    echo            - Close other applications
    echo            - Increase the Windows page file size
    echo            - Force it with: setup_and_run.bat --allow-low-ram
)

echo.
echo ============================================================
echo   [STEP 6/6] Quantization backend (bitsandbytes)
echo ============================================================

for /f "tokens=*" %%v in ('python -c "import torch;print(round(torch.cuda.get_device_properties(0).total_memory/1024**3)) if torch.cuda.is_available() else print(0)" 2^>nul') do set VRAM_GB=%%v
if "%VRAM_GB%"=="" set VRAM_GB=0

for /f "tokens=*" %%x in ('python -c "import torch;print(f'{torch.cuda.get_device_properties(0).total_memory/1024**3:.1f}') if torch.cuda.is_available() else print('0.0')" 2^>nul') do set VRAM_EXACT=%%x
if "%VRAM_EXACT%"=="" set VRAM_EXACT=0.0

echo [INFO] Detected VRAM: %VRAM_EXACT% GB (rounded: %VRAM_GB% GB)
echo [INFO] Qwen3.5-4B needs ~8.6 GB in fp16, ~3.1 GB in 4-bit.

if %VRAM_GB% EQU 0 (
    echo [INFO] No CUDA device. Quantization is not applicable.
    goto :quant_done
)

if %VRAM_GB% GEQ 10 (
    echo [INFO] VRAM is sufficient for fp16. Installing bitsandbytes anyway
    echo        so the 4-bit path stays available for larger models.
)

python -c "import bitsandbytes,sys;v=bitsandbytes.__version__;p=[int(''.join(c for c in x if c.isdigit()) or 0) for x in v.split('.')[:3]];sys.exit(0 if p>=[0,43,0] else 1)" 2>nul
if !errorlevel! equ 0 (
    for /f "tokens=*" %%b in ('python -c "import bitsandbytes;print(bitsandbytes.__version__)" 2^>nul') do set BNB_VER=%%b
    echo   [SKIP] bitsandbytes !BNB_VER! already installed.
) else (
    echo [INFO] Installing bitsandbytes ^>=0.43.0 ...
    pip install "bitsandbytes>=0.43.0"
    if !errorlevel! neq 0 (
        echo   [WARN] bitsandbytes install failed.
        echo          The refiner will fall back to CPU offload, which is slow,
        echo          or be skipped entirely on low-RAM machines.
        goto :quant_done
    )
)

echo [INFO] Verifying the 4-bit kernel actually runs...
python -c "import torch,bitsandbytes as bnb;d=torch.bfloat16 if torch.cuda.get_device_capability(0)[0]>=8 else torch.float16;l=bnb.nn.Linear4bit(64,64,bias=False,compute_dtype=d,quant_type='nf4').to('cuda');x=torch.randn(2,64,device='cuda',dtype=d);y=l(x);assert torch.isfinite(y.float()).all();print(f'  [OK] 4-bit NF4 verified (compute dtype: {str(d).replace(chr(34),chr(39))})')" 2>nul
if !errorlevel! neq 0 (
    echo   [WARN] 4-bit self-test failed. Trying 8-bit...
    python -c "import torch,bitsandbytes as bnb;l=bnb.nn.Linear8bitLt(64,64,bias=False,has_fp16_weights=False).to('cuda');x=torch.randn(2,64,device='cuda',dtype=torch.float16);y=l(x);assert torch.isfinite(y.float()).all();print('  [OK] 8-bit LLM.int8 verified')" 2>nul
    if !errorlevel! neq 0 (
        echo   [WARN] Neither 4-bit nor 8-bit works.
        echo          The CUDA binary inside bitsandbytes may not match PyTorch.
        echo          Try: pip install --force-reinstall "bitsandbytes>=0.43.0"
    )
)

:quant_done

echo.
echo ============================================================
echo   Verify
echo ============================================================
python -c "import torch; print(f'  PyTorch: {torch.__version__} | CUDA: {torch.cuda.is_available()}')"
python -c "import transformers; print(f'  transformers: {transformers.__version__}')"
python -c "import numpy; print(f'  numpy: {numpy.__version__}')"
python -c "import PIL; print(f'  pillow: {PIL.__version__}')"

python -c "import psutil; print(f'  psutil: {psutil.__version__}')" 2>nul
if !errorlevel! neq 0 echo   psutil: NOT INSTALLED (Windows API fallback)

python -c "import paddle; print(f'  paddlepaddle: {paddle.__version__}')" 2>nul
if !errorlevel! neq 0 echo   paddlepaddle: NOT INSTALLED (OCR draft disabled)

python -c "import paddleocr; print(f'  paddleocr: {paddleocr.__version__}')" 2>nul
if !errorlevel! neq 0 echo   paddleocr: NOT INSTALLED (OCR draft disabled)

python -c "import os;os.environ.setdefault('FLAGS_use_mkldnn','0');from paddleocr import TextDetection;import numpy as np;d=TextDetection(model_name='PP-OCRv5_mobile_det',device='cpu',enable_mkldnn=False);list(d.predict(input=[np.full((96,320,3),255,dtype='uint8')]));print('  PP-OCRv5 det: OK (oneDNN off)')" 2>nul
if !errorlevel! neq 0 echo   PP-OCRv5 det: selftest failed (image-processing fallback will be used)

python -c "import webview; print(f'  pywebview: {webview.__version__}')" 2>nul
if !errorlevel! neq 0 (
    echo   pywebview: IMPORT FAILED — details below
    python -c "import webview" 2>&1 | findstr /v /c:\"\" 
    echo.
    echo          Windows needs pythonnet for the EdgeChromium backend:
    echo            pip install --upgrade "pywebview>=5.0" "pythonnet>=3.0.3" pywin32
    echo          The CLI still works: python app.py document.pdf --save
)

python -c "import stanza; print(f'  stanza: {stanza.__version__}')" 2>nul
if !errorlevel! neq 0 echo   stanza: NOT INSTALLED (NLP gate disabled)

python -c "import bitsandbytes; print(f'  bitsandbytes: {bitsandbytes.__version__}')" 2>nul
if !errorlevel! neq 0 echo   bitsandbytes: NOT INSTALLED (offload fallback, very slow)

python -c "import pypdfium2; print(f'  pypdfium2: {pypdfium2.V_PYPDFIUM2}')" 2>nul
if !errorlevel! neq 0 python -c "import pypdfium2; print('  pypdfium2: OK')" 2>nul
if !errorlevel! neq 0 echo   pypdfium2: NOT INSTALLED (PDF rendering disabled)

python -c "import pypdf; print(f'  pypdf: {pypdf.__version__}')" 2>nul
if !errorlevel! neq 0 echo   pypdf: NOT INSTALLED (PDF text fallback disabled)

echo.
echo ============================================================
echo   Preflight — importing app.py
echo ============================================================
python -c "import app; print('  [OK] app.py imported cleanly')"
if %errorlevel% neq 0 (
    echo.
    echo [ERROR] app.py failed to import. The traceback above shows the cause.
    echo         Common causes:
    echo           - a missing name in a type annotation
    echo           - a module removed but still imported
    echo         Fix it before the app can start.
    pause
    exit /b 1
)

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
echo   Memory
echo     Usable RAM      : %RAM_GB% GB
echo     Cap RAM         : --ram-limit 8
echo     Force load      : --allow-low-ram  (risk of process kill)
echo   PDF
echo     Engine          : PDFium via pypdfium2 (BSD-3-Clause)
echo                       No Poppler, no Ghostscript, no copyleft.
echo     Resolution      : --pdf-dpi 200
echo     Page cap        : --pdf-pages 32
echo     Text layer      : used automatically when present
echo                       override with --pdf-force-vision
echo   Quantization
echo     Backend         : bitsandbytes (MIT)
echo     Mode            : auto (4-bit NF4, falls back to 8-bit)
echo     Force           : --quant 4bit ^| --quant 8bit ^| --quant off
echo     Skip install    : --no-quant-install
echo   Diagnostics
echo     Component check : python app.py --preflight
echo     Model status    : python app.py --check-only
echo ============================================================
echo.

python "%~dp0app.py" %*

if %errorlevel% neq 0 (
    echo.
    echo [ERROR] App crashed
    pause
)

endlocal