@echo off
chcp 65001 >nul 2>&1
echo ========================================
echo   算法包验证工具
echo ========================================
echo.
if "%~1"=="" (
    echo 用法: 将算法文件夹拖到此文件上即可验证
    echo 或者: verify.bat algorithms\你的算法名
    echo.
    echo 验证全部算法: verify.bat --all
    echo.
    pause
    exit /b
)
python "%~dp0validate_algo.py" %*
echo.
pause
