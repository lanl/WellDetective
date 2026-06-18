@echo off
setlocal

set "CONDA_BAT=%USERPROFILE%\anaconda3\Scripts\activate.bat"

if not exist "%CONDA_BAT%" (
    set "CONDA_BAT=%USERPROFILE%\miniconda3\Scripts\activate.bat"
)

if not exist "%CONDA_BAT" (
    set "CONDA_BAT=%USERPROFILE%\AppData\Local\miniconda3\Scripts\activate.bat
)

if not exist "%CONDA_BAT%" (
    echo Could not find Anaconda or Miniconda activate.bat.
    echo Expected one of:
    echo   %USERPROFILE%\anaconda3\Scripts\activate.bat
    echo   %USERPROFILE%\miniconda3\Scripts\activate.bat
    echo.
    echo Open Anaconda Prompt manually and run:
    echo   where conda
    pause
    exit /b 1
)

call "%CONDA_BAT%"

cd /d ".\src"

echo Creating conda environment: WellDetectiveEnv
call conda create -n WellDetectiveEnv python=3.13 pip -y

echo Activating conda environment
call conda activate WellDetectiveEnv

echo Installing dependencies
python -m pip install -r requirements.txt

echo.
echo Setup complete.
pause