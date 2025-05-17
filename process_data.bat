@echo off
echo === 데이터 전처리 파이프라인 시작 ===
echo 시작 시간: %TIME%
echo.

REM Anaconda/Miniconda 환경 활성화 (경로는 실제 설치 위치로 수정해야 함)
call C:\Users\USER\anaconda3\Scripts\activate.bat ds_env
if %ERRORLEVEL% neq 0 (
    echo 오류: Anaconda 환경 활성화 실패
    goto :error
)

echo 1. 수집된 데이터 확인 중...
if not exist "data\raw\DL_FIRE_M-C61_613954\fire_archive_M-C61_613954.csv" (
    echo 오류: 산불 데이터 파일이 존재하지 않습니다.
    echo 먼저 collect_data.bat를 실행하여 데이터를 수집하세요.
    goto :error
)

set WEATHER_FILES_EXIST=0
for %%F in (data\raw\era5_korea_*.nc) do set WEATHER_FILES_EXIST=1
if %WEATHER_FILES_EXIST%==0 (
    echo 오류: 날씨 데이터 파일이 존재하지 않습니다.
    echo 먼저 collect_data.bat를 실행하여 데이터를 수집하세요.
    goto :error
)
echo 수집된 데이터 확인 완료.

echo 2. 이전 처리 파일 삭제 중...
if exist "processed_data\*.csv" del /Q "processed_data\*.csv"
if exist "outputs\data\weather_data.csv" del /Q "outputs\data\weather_data.csv"
if exist "outputs\data\weather_data_with_wind.csv" del /Q "outputs\data\weather_data_with_wind.csv"

echo 3. 산불 데이터 전처리 중...
cd src\preprocessing
python process_af_flag.py --input "..\..\data\raw\DL_FIRE_M-C61_613954\fire_archive_M-C61_613954.csv" --output "..\..\data\reference\af_flag_korea.csv"
if %ERRORLEVEL% neq 0 (
    echo 오류: 산불 데이터 전처리 실패
    goto :error
)
cd ..\..

echo 4. 날씨 데이터 전처리 및 풍속 계산, 데이터 병합 처리...
cd src\preprocessing
python process_weather.py --data_dir "..\..\data\raw" --output_dir "..\..\processed_data" --target_path "..\..\data\reference\af_flag_korea.csv" --final_output "..\..\outputs\data\weather_data_with_wind.csv"
if %ERRORLEVEL% neq 0 (
    echo 오류: 날씨 데이터 전처리 실패
    goto :error
)
cd ..\..

echo 5. 차원 일치 검증 중...
python check_dimensions.py --weather_data "outputs\data\weather_data_with_wind.csv" --af_flag "data\reference\af_flag_korea.csv"
if %ERRORLEVEL% neq 0 (
    echo 경고: 차원 일치 검증에 문제가 발생했습니다. 데이터를 확인하세요.
)

echo.
echo === 전처리 파이프라인 완료 ===
echo 완료 시간: %TIME%
echo 결과 파일:
echo - 산불 데이터: data\reference\af_flag_korea.csv
echo - 최종 데이터: outputs\data\weather_data_with_wind.csv (풍속 계산 포함)
goto :end

:error
echo.
echo === 전처리 파이프라인 실행 중 오류 발생 ===
echo 중단 시간: %TIME%

:end
echo.
echo 엔터 키를 눌러 종료하세요...
pause > nul 