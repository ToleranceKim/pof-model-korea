@echo off
echo === 날씨 데이터 및 산불 데이터 처리 파이프라인 시작 ===
echo 시작 시간: %TIME%
echo.

REM Anaconda/Miniconda 환경 활성화 (경로는 실제 설치 위치로 수정해야 함)
call C:\Users\USER\anaconda3\Scripts\activate.bat ds_env
if %ERRORLEVEL% neq 0 (
    echo 오류: Anaconda 환경 활성화 실패
    goto :error
)

echo 1. 이전 처리 파일 삭제 중...
if exist "processed_data\*.csv" del /Q "processed_data\*.csv"
if exist "outputs\data\weather_data.csv" del /Q "outputs\data\weather_data.csv"

echo 2. 산불 데이터 전처리 중...
cd src\preprocessing
python process_af_flag.py --input "..\..\DL_FIRE_M-C61_613954\fire_archive_M-C61_613954.csv" --output "..\..\data\reference\af_flag_korea.csv"
if %ERRORLEVEL% neq 0 (
    echo 오류: 산불 데이터 전처리 실패
    goto :error
)
cd ..\..

echo 3. 날씨 데이터 수집 실행 중...
cd src\data_collection
python collect_weather.py
if %ERRORLEVEL% neq 0 (
    echo 오류: 날씨 데이터 수집 실패
    goto :error
)
cd ..\..

echo 4. 날씨 데이터 전처리 및 풍속 계산, 데이터 병합 처리...
cd src\preprocessing
python process_weather.py --final_output "..\..\outputs\data\weather_data_with_wind.csv"
if %ERRORLEVEL% neq 0 (
    echo 오류: 날씨 데이터 전처리 실패
    goto :error
)
cd ..\..

echo.
echo === 파이프라인 완료 ===
echo 완료 시간: %TIME%
echo 결과 파일:
echo - 산불 데이터: data\reference\af_flag_korea.csv
echo - 최종 데이터: outputs\data\weather_data_with_wind.csv (풍속 계산 포함)
goto :end

:error
echo.
echo === 파이프라인 실행 중 오류 발생 ===
echo 중단 시간: %TIME%

:end
echo.
echo 엔터 키를 눌러 종료하세요...
pause > nul 