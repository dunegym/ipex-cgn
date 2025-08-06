@echo off
chcp 65001 >nul
echo ==========================================
echo 启动 ipex-cgn 项目
echo ==========================================
echo.
echo [步骤1] 激活conda环境: openvino-cgn
call conda activate openvino-cgn
if %errorlevel% neq 0 (
    echo 错误: 环境激活失败
    pause
    exit /b 1
)
echo 环境激活成功
echo.
echo [步骤2] 切换到项目目录: %~dp0
cd /d %~dp0
echo 当前目录: %cd%
echo.
echo [步骤3] 启动主程序
python src/main.py
echo.
echo 程序已退出，按任意键关闭窗口...
pause
