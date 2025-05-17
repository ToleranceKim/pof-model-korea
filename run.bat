@echo off
echo ===== 전체 파이프라인 실행 시작 =====
echo 시작 시간: %TIME%
echo.

echo 1단계: 데이터 수집 실행
call collect_data.bat
if %ERRORLEVEL% neq 0 (
    echo 데이터 수집 단계에서 오류가 발생했습니다.
    echo 오류를 해결한 후 다시 시도하세요.
    goto :end
)

echo.
echo 2단계: 데이터 전처리 실행
call process_data.bat
if %ERRORLEVEL% neq 0 (
    echo 데이터 전처리 단계에서 오류가 발생했습니다.
    echo 오류를 해결한 후 다시 시도하세요.
    goto :end
)

echo.
echo ===== 전체 파이프라인 실행 완료 =====
echo 완료 시간: %TIME%
echo.
echo 최종 결과 파일:
echo - 산불 데이터: data\reference\af_flag_korea.csv
echo - 최종 데이터: outputs\data\weather_data_with_wind.csv

:end
echo.
echo 엔터 키를 눌러 종료하세요...
pause > nul 