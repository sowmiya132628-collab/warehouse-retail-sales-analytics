@echo off
echo.
echo  =====================================================
echo   Warehouse ^& Retail Sales Intelligence Dashboard
echo  =====================================================
echo.
echo  Starting Streamlit dashboard...
echo.

cd /d "C:\Users\Sowmiya Subramaniyan\OneDrive\Desktop\UPSC\Screenshots\aicte data analyst"

:: Check if streamlit is installed
streamlit --version >nul 2>&1
if %errorlevel% neq 0 (
    echo  Installing required packages...
    pip install -r requirements.txt
    echo.
)

:: Open browser and run dashboard
start http://localhost:8501
streamlit run app.py

pause
