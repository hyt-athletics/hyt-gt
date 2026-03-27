@echo off
chcp 65001 >nul 2>&1
title 井中探测数据处理与反演子系统 v0.3

echo ========================================
echo   井中探测数据处理与反演子系统 v0.3
echo ========================================
echo.

:: 检查 Python 是否安装
python --version >nul 2>&1
if errorlevel 1 (
    echo [错误] 未检测到 Python，请先安装 Python 3.11+
    echo 下载地址: https://www.python.org/downloads/
    echo 安装时请勾选 "Add Python to PATH"
    pause
    exit /b 1
)

:: 首次运行检查依赖
if not exist ".venv" (
    echo 首次运行，正在安装依赖（约需 2-5 分钟）...
    echo.
    pip install uv >nul 2>&1
    uv sync
    echo.
    echo 依赖安装完成！
    echo.
)

:: 启动软件
echo 正在启动...
python main.py

if errorlevel 1 (
    echo.
    echo [错误] 软件启动失败，请检查上方错误信息。
    echo 常见解决方法:
    echo   1. 运行: uv sync
    echo   2. 运行: pip install PyQt6 numpy scipy matplotlib
    pause
)
