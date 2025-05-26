@echo off
start "" /min cmd /c "
    call conda activate openvino-cgn
    cd /d %~dp0
    python main.py
"