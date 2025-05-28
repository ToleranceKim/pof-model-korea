# ERA5 데이터 처리 문서

## 변수 처리 이슈와 해결책

### 문제 상황

초기에는 ERA5 데이터 처리 과정에서 일부 변수(t2m, d2m, u10, v10 등)가 최종 결과에 포함되지 않고 강수량(tp) 변수만 처리되는 문제가 있었습니다.

### 원인 분석

1. ERA5 원본 파일은 확장자가 `.nc`이지만 실제로는 ZIP 파일 형식이었습니다.
2. 각 ZIP 파일 내부에는 두 개의 NetCDF 파일이 포함되어 있었습니다:
   - `data_stream-oper_stepType-accum.nc`: 누적 변수(강수량 등) 포함
   - `data_stream-oper_stepType-instant.nc`: 순간 변수(온도, 습도, 바람 등) 포함
3. 초기 코드는 첫 번째 파일(`accum.nc`)만 처리하고 있었기 때문에 강수량(tp) 변수만 최종 결과에 포함되었습니다.

### 해결 방법

1. `interpolate_era5.py` 스크립트를 수정하여 각 ZIP 파일에서 추출된 모든 NetCDF 파일을 처리하도록 변경했습니다.
2. 각 파일에서 보간된 변수들을 하나의 데이터셋으로 결합하여 모든 변수(t2m, d2m, u10, v10, tp 등)가 최종 결과에 포함되도록 했습니다.
3. 수정된 처리 흐름:
   - 원본 ERA5 ZIP 파일에서 모든 NetCDF 파일 추출
   - 각 NetCDF 파일의 모든 변수를 0.1도 그리드로 보간
   - 모든 변수를 하나의 데이터셋으로 결합
   - 결합된 데이터에서 일별 통계 계산

## 데이터 처리 파이프라인

### 디렉토리 구조

- `data/raw`: 원본 ERA5 데이터 (ZIP 형식의 NC 파일들)
- `processed_data/era5`: 0.1도 그리드로 보간된 중간 결과물 (모든 변수 포함)
- `processed_data/era5_daily`: 일별 통계가 계산된 최종 결과물 (CSV, Parquet 형식)

### 주요 스크립트

- `src/preprocessing/interpolate_era5.py`: ERA5 데이터를 0.1도 그리드로 보간
- `src/preprocessing/process_interpolated.py`: 보간된 데이터에서 일별 통계 계산
- `src/preprocessing/combine_era5_data.py`: 모든 월별 일일 통계 파일을 하나의 통합 데이터셋으로 결합

### 통합 데이터셋 생성 방법

전체 2019-2024년 기간의 데이터를 하나의 파일로 통합하려면:

```bash
# 기본 사용법 (processed_data/era5_daily 디렉토리의 모든 파일 통합)
python src/preprocessing/combine_era5_data.py

# 특정 기간 지정
python src/preprocessing/combine_era5_data.py processed_data/era5_daily processed_data 201901 202412
```

생성된 통합 파일은 `processed_data/era5_daily_combined_201901_202412.parquet` 및 `processed_data/era5_daily_combined_201901_202412.csv` 형식으로 저장됩니다.

### 주의사항

1. ERA5 데이터 파일의 실제 형식이 파일 확장자와 다를 수 있으므로 항상 내용을 확인해야 합니다.
2. NetCDF 파일을 읽기 위해서는 적절한 라이브러리(netCDF4, h5netcdf 등)가 설치되어 있어야 합니다.
3. 대용량 데이터 처리 시 메모리 관리에 주의해야 합니다.
