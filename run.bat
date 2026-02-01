@echo off
chcp 65001 >nul

echo ==========================================
echo   Qwen3-ASR 启动工具
echo ==========================================
echo.

REM 检查虚拟环境
if not exist ".venv\Scripts\uv.exe" (
    echo [错误] 未找到虚拟环境，请先运行 INSTALL.bat
    pause
    exit /b 1
)

REM 扫描可用模型
set "count=0"
set "model1="
set "model2="
set "model3="

for /d %%d in (Qwen3-ASR-1.7B Qwen3-ASR-0.6B Qwen3-ForcedAligner-0.6B) do (
    if exist "%%d" (
        set /a count+=1
        set "model!count!=%%d"
        echo   [!count!] %%d
    )
)

if %count%==0 (
    echo [错误] 未找到模型，请先运行 download_models.bat
    pause
    exit /b 1
)

echo   [q] 退出
echo.

set /p choice="请选择模型编号: "

if /i "%choice%"=="q" exit /b 0

set "selected="
if "%choice%"=="1" set "selected=%model1%"
if "%choice%"=="2" set "selected=%model2%"
if "%choice%"=="3" set "selected=%model3%"

if "%selected%"=="" (
    echo [错误] 无效选择
    pause
    exit /b 1
)

echo.
echo 启动 %selected%...
echo 访问 http://localhost:8000 使用 Web 界面
echo 按 Ctrl+C 停止服务
echo.

uv run qwen-asr-demo --asr-checkpoint ./%selected% --ip 0.0.0.0 --port 8000 --backend transformers --backend-kwargs "{\"device_map\":\"cuda\",\"dtype\":\"bfloat16\"}"

pause