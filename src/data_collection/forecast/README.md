# 기상 예보 데이터 수집 모듈

## 예보 데이터 통합 처리 도구: process_forecast_data.py

`process_forecast_data.py`는 ECMWF 예보 데이터의 수집, 전처리, CSV 변환 기능을 하나의 스크립트로 통합한 도구입니다. 이 스크립트는 다음과 같은 기능을 제공합니다:

1. **데이터 수집**: ECMWF Open Data API를 통해 기상 예보 데이터(2t, 2d, 10u, 10v, tp)를 수집
2. **데이터 전처리**: GRIB2 파일을 읽고, 한국 주변 지역(위경도 범위: [39, 124, 33, 132])의 데이터 추출
3. **변수 변환**: 바람 성분(10u, 10v)을 이용한 풍속(wind10m) 계산, 단위 변환 등
4. **파일 출력**: 처리된 데이터를 CSV 및 Parquet 형식으로 저장

### 주요 변수 설명

| 최종 변수명     | 설명                    | 단위       | 출처                         |
| --------------- | ----------------------- | ---------- | ---------------------------- |
| t2m             | 지상 2m 기온            | K          | 2t 변수에서 변환             |
| td2m            | 지상 2m 이슬점 온도     | K          | 2d 변수에서 변환             |
| tp              | 총 강수량               | mm         | tp 변수에서 변환 (m → mm)    |
| 10u             | 지상 10m 동서 바람 성분 | m/s        | 10u 변수                     |
| 10v             | 지상 10m 남북 바람 성분 | m/s        | 10v 변수                     |
| wind10m         | 지상 10m 풍속           | m/s        | 10u, 10v 변수로 계산         |
| grid_id         | 격자 ID                 | -          | 위경도 기반 계산             |
| forecast_date   | 예보 발표일             | YYYY-MM-DD | 스크립트 실행 시 지정한 날짜 |
| prediction_date | 예측 날짜               | YYYY-MM-DD | forecast_date + lead 일수    |
| lead            | 예보 리드타임           | 일         | 예보 시간 / 24 (1~5)         |

### 사용법

#### 1. 데이터 수집 및 처리

```bash
# 오늘 날짜 기준 데이터 수집 및 처리
python src/data_collection/forecast/process_forecast_data.py --collect

# 특정 날짜 기준 데이터 수집 및 처리
python src/data_collection/forecast/process_forecast_data.py --collect --date 20240530

# 기존 수집 데이터만 처리
python src/data_collection/forecast/process_forecast_data.py --process_only
```

#### 2. 상세 옵션

```bash
python src/data_collection/forecast/process_forecast_data.py --help
```

| 옵션             | 설명                          | 기본값                  |
| ---------------- | ----------------------------- | ----------------------- |
| `--date`         | 예보 기준 날짜 (YYYYMMDD)     | 현재 날짜               |
| `--forecast_dir` | 예보 데이터 디렉토리          | data/forecast           |
| `--output_dir`   | 출력 디렉토리                 | data/processed/forecast |
| `--steps`        | 처리할 예보 시간 목록         | 24 48 72 96 120         |
| `--collect`      | 예보 데이터 수집 여부         | False                   |
| `--process_only` | 기존 다운로드된 데이터만 처리 | False                   |

#### 3. 출력 파일

처리된 데이터는 다음 형식의 파일로 저장됩니다:

```
forecast_{YYYYMMDD}_lead{N}.csv
forecast_{YYYYMMDD}_lead{N}.parquet
```

예: `forecast_20240530_lead1.csv` - 2024년 5월 30일 발표된 D+1(내일) 예보

## 파일 구조

- `process_forecast_data.py` - ECMWF 예보 데이터 수집, 전처리, CSV 변환 통합 스크립트
- `collect_fuel_data.py` - 연료 수분 함량 데이터 수집 스크립트

## ECMWF 예보 데이터

ECMWF(European Centre for Medium-Range Weather Forecasts)의 Open Data 서비스를 통해 제공되는 기상 예보 데이터를 수집합니다.

### 수집 가능한 변수 목록

현재 스크립트에서 직접 수집 가능한 ECMWF 변수 목록:

| 변수 코드 | 설명                    | 단위 | 비고                           |
| --------- | ----------------------- | ---- | ------------------------------ |
| 2t        | 지상 2m 기온            | K    | 켈빈 온도 (°C = K - 273.15)    |
| 2d        | 지상 2m 이슬점 온도     | K    | 켈빈 온도                      |
| 10u       | 지상 10m 동서 바람 성분 | m/s  | 양수: 서→동, 음수: 동→서       |
| 10v       | 지상 10m 남북 바람 성분 | m/s  | 양수: 남→북, 음수: 북→남       |
| tp        | 총 강수량               | m    | 누적값, 보통 mm로 변환(× 1000) |

### 완전한 Forecast 변수 목록 (보류류)

산불 예측 모델을 위해 필요한 전체 예보 변수 목록:

| Forecast 변수                 | 설명                                         | 수집 방법                                        | 제공 리드타임       | 수집 상태      |
| ----------------------------- | -------------------------------------------- | ------------------------------------------------ | ------------------- | -------------- |
| 2 m Temperature               | 지표면 2m 높이의 기온 (°C)                   | ECMWF IFS 중기예보 '2t' 변수로 수집              | D+1…D+10 (24–240 h) | 수집 가능      |
| 2 m Dewpoint                  | 지표면 2m 높이의 이슬점 온도 (°C)            | ECMWF IFS '2d' 변수로 수집                       | D+1…D+10            | 수집 가능      |
| 10 m Wind Speed               | 지표면 10m 높이의 풍속 (m/s), √(10u²+10v²)   | ECMWF IFS '10u', '10v' 변수로 수집 후 계산       | D+1…D+10            | 수집 가능      |
| Precipitation                 | 일간 총 강수량 (mm)                          | ECMWF IFS 'tp' 변수로 수집 후 전처리             | D+1…D+10            | 수집 가능      |
| Lightning density             | 일별 격자당 번개 발생 횟수 (flashes/km²/day) | ECMWF IFS 'lightning' 변수로 수집                | D+1…D+10            | 수집 가능      |
| Live Leaf Fuel Load           | 생엽(leaf) 연료 부하 (kg/m²)                 | McNorton & Di Giuseppe 모델 데이터 다운로드 필요 | D+1…D+10            | 추가 구현 필요 |
| Live Wood Fuel Load           | 생목(wood) 연료 부하 (kg/m²)                 | McNorton & Di Giuseppe 모델 데이터 다운로드 필요 | D+1…D+10            | 추가 구현 필요 |
| Dead Foliage Fuel Load        | 고엽(dead foliage) 연료 부하 (kg/m²)         | McNorton & Di Giuseppe 모델 데이터 다운로드 필요 | D+1…D+10            | 추가 구현 필요 |
| Dead Wood Fuel Load           | 고목(dead wood) 연료 부하 (kg/m²)            | McNorton & Di Giuseppe 모델 데이터 다운로드 필요 | D+1…D+10            | 추가 구현 필요 |
| Live Fuel Moisture Content    | 생연료 수분 함량 (%)                         | McNorton & Di Giuseppe 모델 데이터 다운로드 필요 | D+1…D+10            | 추가 구현 필요 |
| Dead Foliage Moisture Content | 고엽 수분 함량 (%)                           | McNorton & Di Giuseppe 모델 데이터 다운로드 필요 | D+1…D+10            | 추가 구현 필요 |
| Dead Wood Moisture Content    | 고목 수분 함량 (%)                           | McNorton & Di Giuseppe 모델 데이터 다운로드 필요 | D+1…D+10            | 추가 구현 필요 |

#### 추가 데이터 수집 관련

- 연료 부하 및 수분 함량 관련 데이터: McNorton & Di Giuseppe 전역 연료 특성 모델에서 생성된 데이터로, DOI 기록(https://doi.org/10.24381/378d1497)에서 API 또는 Zenodo를 통해 별도 다운로드가 필요합니다.
- 이 데이터를 자동으로 수집하는 추가 스크립트 구현이 필요합니다.

### 예보 시간

ECMWF 예보는 매일 00Z, 12Z(UTC 기준)에 10일(240시간) 예보를 제공합니다:

- `step001` ~ `step090`: 1시간 간격 (D+1 ~ D+3.75)
- `step093` ~ `step144`: 3시간 간격 (D+3.875 ~ D+6)
- `step150` ~ `step240`: 6시간 간격 (D+6.25 ~ D+10)

일반적으로 24시간 간격으로 다음 날(D+1)부터 10일 후(D+10)까지의 데이터를 수집합니다:

- `step024` (D+1), `step048` (D+2), `step072` (D+3), ... , `step240` (D+10)

### 파일 이름 규칙

수집된 파일은 다음 형식의 이름으로 저장됩니다:

```
{변수명}_{날짜}_{예보시각}z_step{예보시간}.grib2
```

예시:

- `2t_2023-04-01_00z_step024.grib2`: 2023년 4월 1일 00Z 예보의 24시간 후(D+1) 지상 2m 기온
- `tp_2023-04-01_12z_step048.grib2`: 2023년 4월 1일 12Z 예보의 48시간 후(D+2) 총 강수량

## 의존성

- `ecmwf-opendata`: ECMWF Open Data API에 접근하기 위한 패키지
- `cfgrib`: GRIB 파일 처리 라이브러리
- `xarray`, `pandas`, `numpy`: 데이터 처리 라이브러리
- `pyarrow`: Parquet 파일 저장을 위한 라이브러리
- 설치:
  ```bash
  pip install ecmwf-opendata cfgrib xarray pandas numpy pyarrow
  ```

## 참고 자료

- [ECMWF Open Data 문서](https://www.ecmwf.int/en/forecasts/datasets/open-data)
- [ecmwf-opendata 패키지](https://pypi.org/project/ecmwf-opendata/)
