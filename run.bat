@echo off
REM 启动脚本，用于快速运行项目
REM 该脚本会最小化启动一个命令行窗口，激活conda环境并运行主程序

REM 关闭命令回显
start "" /min cmd /c "
    REM 激活conda环境
    call conda activate openvino-cgn
    REM 切换到脚本所在目录
    cd /d %~dp0
    REM 运行主程序
    python main.py
"
