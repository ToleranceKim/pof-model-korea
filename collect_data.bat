@echo off
echo === 데이터 수집 파이프라인 시작 ===
echo 시작 시간: %TIME%
echo.

REM Anaconda/Miniconda 환경 활성화 (경로는 실제 설치 위치로 수정해야 함)
call C:\Users\USER\anaconda3\Scripts\activate.bat ds_env
if %ERRORLEVEL% neq 0 (
    echo 오류: Anaconda 환경 활성화 실패
    goto :error
)

echo 1. 산불 데이터 수집 확인...
if not exist "data\raw\DL_FIRE_M-C61_613954\fire_archive_M-C61_613954.csv" (
    echo 경고: 산불 데이터 파일이 존재하지 않습니다.
    echo DL_FIRE_M-C61_613954\fire_archive_M-C61_613954.csv 파일을 data\raw 디렉토리에 위치시키세요.
    goto :error
)
echo 산불 데이터 확인 완료.

REM 날씨 데이터 수집 기간 설정 (아래 값을 수정하여 수집 기간 변경)
set START_YEAR=2024
set END_YEAR=2024
set START_MONTH=1
set END_MONTH=12

echo 2. 날씨 데이터 수집 실행 중...
echo    수집 기간: %START_YEAR%-%START_MONTH%월 ~ %END_YEAR%-%END_MONTH%월
cd src\data_collection
python collect_weather.py --start_year %START_YEAR% --end_year %END_YEAR% --start_month %START_MONTH% --end_month %END_MONTH% --output_dir "..\..\data\raw"
if %ERRORLEVEL% neq 0 (
    echo 오류: 날씨 데이터 수집 실패
    goto :error
)
cd ..\..

echo.
echo === 데이터 수집 완료 ===
echo 완료 시간: %TIME%
echo 다음 단계: process_data.bat 실행하여 데이터 전처리 진행
echo.
echo 수집된 데이터 검증 후 전처리를 진행하세요.
echo - 날씨 데이터: data\raw\era5_korea_*.nc
echo - 산불 데이터: data\raw\DL_FIRE_M-C61_613954\fire_archive_M-C61_613954.csv
goto :end

:error
echo.
echo === 데이터 수집 중 오류 발생 ===
echo 중단 시간: %TIME%

:end
echo.
echo 엔터 키를 눌러 종료하세요...
pause > nul 