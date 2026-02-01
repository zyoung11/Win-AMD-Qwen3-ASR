@echo off
chcp 65001 >nul
echo ==========================================
echo   Qwen3-ASR AMD GPU Windows 安装脚本
echo ==========================================
echo.

REM 配置 GitHub Release 地址
set "GITHUB_USER=zyoung11"
set "GITHUB_REPO=Win-AMD-Qwen3-ASR"
set "RELEASE_TAG=0.1.0"
set "RELEASE_URL=https://github.com/%GITHUB_USER%/%GITHUB_REPO%/releases/download/%RELEASE_TAG%"

REM 检查 uv
where uv >nul 2>nul
if errorlevel 1 (
    echo [错误] 需要安装 uv: winget install astral-sh.uv
    pause
    exit /b 1
)

echo [0/4] 从 GitHub Release 下载依赖文件...
echo.

REM 下载 4 个 whl 文件
call :download_file "torch-2.8.0a0+gitfc14c65-cp312-cp312-win_amd64.whl"
if errorlevel 1 exit /b 1

call :download_file "torchaudio-2.6.0a0+1a8f621-cp312-cp312-win_amd64.whl"
if errorlevel 1 exit /b 1

call :download_file "torchvision-0.24.0a0+c85f008-cp312-cp312-win_amd64.whl"
if errorlevel 1 exit /b 1

call :download_file "qwen_asr-0.0.6-py3-none-any.whl"
if errorlevel 1 exit /b 1

echo.
echo [1/4] 创建虚拟环境...
uv venv --python 3.12
if errorlevel 1 (
    echo [错误] 创建虚拟环境失败，确保已安装 Python 3.12
    pause
    exit /b 1
)

echo [2/4] 安装 AMD PyTorch...
uv pip install .\torch-2.8.0a0+gitfc14c65-cp312-cp312-win_amd64.whl .\torchaudio-2.6.0a0+1a8f621-cp312-cp312-win_amd64.whl .\torchvision-0.24.0a0+c85f008-cp312-cp312-win_amd64.whl
if errorlevel 1 (
    echo [错误] 安装 PyTorch 失败
    pause
    exit /b 1
)

echo [3/4] 安装 Qwen-ASR...
uv pip install .\qwen_asr-0.0.6-py3-none-any.whl
if errorlevel 1 (
    echo [错误] 安装 Qwen-ASR 失败
    pause
    exit /b 1
)

echo [4/4] 安装 ModelScope...
uv pip install -U modelscope
if errorlevel 1 (
    echo [警告] ModelScope 安装可能有问题，但继续...
)

echo.
echo ==========================================
echo   安装完成！
echo ==========================================
echo.
echo ==========================================
echo   Qwen3-ASR 模型下载
echo ==========================================
echo.
echo 可用模型：
echo   [1] Qwen3-ASR-1.7B      (~3.4GB, 高精度)
echo   [2] Qwen3-ASR-0.6B      (~1.2GB, 快速)
echo   [3] Qwen3-ForcedAligner-0.6B  (~1.2GB, 强制对齐)
echo   [a] 下载全部
echo   [q] 退出
echo.

set /p choice="请选择（可多选，如 123）: "

if /i "%choice%"=="q" exit /b 0

set "models="
if /i "%choice%"=="a" set "models=1.7B 0.6B forced"
echo %choice% | findstr "1" >nul && set "models=%models% 1.7B"
echo %choice% | findstr "2" >nul && set "models=%models% 0.6B"
echo %choice% | findstr "3" >nul && set "models=%models% forced"

if "%models%"=="" (
    echo [错误] 无效选择
    pause
    exit /b 1
)

echo.
echo 将下载以下模型: %models%
echo.

for %%m in (%models%) do (
    if "%%m"=="1.7B" (
        echo [下载] Qwen3-ASR-1.7B...
        uv run python -m modelscope download --model Qwen/Qwen3-ASR-1.7B --local_dir ./Qwen3-ASR-1.7B
        if errorlevel 1 echo [错误] 1.7B 下载失败
    )
    if "%%m"=="0.6B" (
        echo [下载] Qwen3-ASR-0.6B...
        uv run python -m modelscope download --model Qwen/Qwen3-ASR-0.6B --local_dir ./Qwen3-ASR-0.6B
        if errorlevel 1 echo [错误] 0.6B 下载失败
    )
    if "%%m"=="forced" (
        echo [下载] Qwen3-ForcedAligner-0.6B...
        uv run python -m modelscope download --model Qwen/Qwen3-ForcedAligner-0.6B --local_dir ./Qwen3-ForcedAligner-0.6B
        if errorlevel 1 echo [错误] ForcedAligner 下载失败
    )
)

echo.
echo ==========================================
echo   下载完成！
echo ==========================================
echo.
echo 运行 run.bat 启动演示
echo.
pause
exit /b 0

REM 下载文件函数
:download_file
set "filename=%~1"
set "filepath=.\%filename%"

if exist "%filepath%" (
    echo [跳过] %filename% 已存在
    exit /b 0
)

echo [下载] %filename%...
uv run python -c "import urllib.request; urllib.request.urlretrieve('%RELEASE_URL%/%filename%', '%filename%')" 2>nul
if errorlevel 1 (
    echo [错误] 下载 %filename% 失败
    echo 请检查网络连接或手动下载：
    echo   %RELEASE_URL%/%filename%
    pause
    exit /b 1
)
echo [完成] %filename%
exit /b 0