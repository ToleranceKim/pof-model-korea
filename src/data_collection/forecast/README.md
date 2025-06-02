# 기상 예보 데이터 수집 모듈

## 예보 데이터 통합 처리 도구: process_forecast_data.py

`process_forecast_data.py`는 ECMWF 예보 데이터의 수집, 전처리, CSV 변환 기능을 하나의 스크립트로 통합한 도구입니다. 이 스크립트는 다음과 같은 기능을 제공합니다:

1. **데이터 수집**: ECMWF Open Data API를 통해 기상 예보 데이터(2t, 2d, 10u, 10v, tp)를 수집
2. **데이터 전처리**:
   - GRIB2 파일을 읽고 한국 지역(33-39°N, 124-132°E)만 추출
   - 추출된 한국 지역 데이터를 0.1° 격자로 선형 보간
   - 바람 성분 데이터(10u, 10v)로 풍속(wind10m) 계산 (원본 바람 성분도 유지)
3. **변수 변환**: 바람 성분(10u, 10v)을 이용한 풍속(wind10m) 계산
4. **grid_id 기반 필터링**: 유효한 격자 ID만 포함하도록 필터링
5. **파일 출력**: 처리된 데이터를 CSV 및 Parquet 형식으로 저장
6. **데이터 시각화**: 격자 분포 시각화 (선택 사항)

### 성능 최적화

- API 제한으로 위경도 지정 다운로드는 불가능하지만, 다운로드 후 xarray 라이브러리를 사용하여 한국 지역만 선택 후 보간 처리
- 전세계 데이터를 모두 보간하지 않고 필요한 지역만 처리하여 성능 대폭 향상
- 선형 보간 전 한국 지역만 선택하여 메모리 사용량 최소화

## 디렉토리 구조 및 데이터 흐름

### 디렉토리 구조

```
pof-model-korea/
│
├── data/                           # 데이터 저장 루트 디렉토리
│   ├── forecast/                   # 원본 예보 데이터 저장 디렉토리
│   │   ├── 2t/                     # 지상 2m 기온 데이터
│   │   ├── 2d/                     # 지상 2m 이슬점 온도 데이터
│   │   ├── 10u/                    # 지상 10m 동서 바람 성분 데이터
│   │   ├── 10v/                    # 지상 10m 남북 바람 성분 데이터
│   │   └── tp/                     # 총 강수량 데이터
│   │
│   └── processed/                  # 전처리된 데이터 저장 디렉토리
│       └── forecast/               # 전처리된 예보 데이터 저장 디렉토리
│           ├── forecast_*.csv      # 전처리된 CSV 파일
│           ├── forecast_*.parquet  # 전처리된 Parquet 파일
│           └── forecast_*_*.json   # 필터링 통계 및 메타데이터
│
└── src/
    └── data_collection/            # 데이터 수집 모듈
        └── forecast/               # 예보 데이터 수집 모듈
            ├── process_forecast_data.py  # 예보 데이터 처리 스크립트
            └── README.md                 # 문서
```

### 데이터 흐름

1. **수집 단계**:

   - ECMWF API를 통해 GRIB2 형식의 전세계 예보 데이터를 다운로드
   - 각 변수별로 `data/forecast/{변수명}/` 디렉토리에 원본 파일 저장
   - 파일명 형식: `{변수명}_{날짜}_{예보시각}z_step{예보시간}.grib2`

2. **전처리 단계**:

   - 각 GRIB2 파일에서 한국 지역 데이터만 추출
   - 추출된 데이터를 0.1° 격자로 선형 보간
   - 바람 성분 데이터(10u, 10v)로 풍속(wind10m) 계산 (원본 바람 성분도 유지)
   - 위경도 좌표를 grid_id로 변환
   - 유효한 grid_id 목록으로 필터링

3. **저장 단계**:
   - 처리된 데이터를 CSV 및 Parquet 형식으로 `data/processed/forecast/` 디렉토리에 저장
   - 예보 날짜와 리드타임으로 구분된 파일명 사용
   - 필터링 통계 정보를 JSON 파일로 저장
   - 선택적으로 격자 분포 시각화 이미지 저장

### 주요 변수 설명

| 최종 변수명     | 설명                    | 단위       | 출처                         | 용도                   |
| --------------- | ----------------------- | ---------- | ---------------------------- | ---------------------- |
| t2m             | 지상 2m 기온            | K          | 2t 변수에서 변환             | 기온 예측              |
| td2m            | 지상 2m 이슬점 온도     | K          | 2d 변수에서 변환             | 습도 관련 예측         |
| tp              | 총 강수량               | mm         | tp 변수에서 변환 (m → mm)    | 강수량 예측            |
| 10u             | 지상 10m 동서 바람 성분 | m/s        | 10u 변수                     | 풍향 계산 및 바람 분석 |
| 10v             | 지상 10m 남북 바람 성분 | m/s        | 10v 변수                     | 풍향 계산 및 바람 분석 |
| wind10m         | 지상 10m 풍속           | m/s        | 10u, 10v 변수로 계산         | 풍속 예측              |
| grid_id         | 격자 ID                 | -          | 위경도 기반 계산             | 다른 데이터셋과 결합   |
| forecast_date   | 예보 발표일             | YYYY-MM-DD | 스크립트 실행 시 지정한 날짜 | 발표 시점 식별         |
| prediction_date | 예측 날짜               | YYYY-MM-DD | forecast_date + lead 일수    | 예측 대상 날짜 식별    |
| lead            | 예보 리드타임           | 일         | 예보 시간 / 24 (1~5)         | 예측 선행 시간 식별    |

## 백엔드 개발자를 위한 핵심 함수 개요

스크립트에서 중요한 함수들과 그 역할을 설명합니다:

### 1. 데이터 수집 함수

```python
def collect_forecast_data(target_date=None, output_dir=None, steps=None, max_retries=3, retry_delay=5):
    """ECMWF에서 기본 기상 예보 변수 데이터를 수집합니다."""
    # 위치: 라인 34-147
    # 역할: ECMWF Open Data API를 통해 5가지 기본 변수의 GRIB2 파일을 다운로드
```

### 2. GRIB2 변환 함수

```python
def grib_to_dataframe(grib_file, interpolate=True):
    """GRIB2 파일을 Pandas DataFrame으로 변환합니다."""
    # 위치: 라인 148-437
    # 역할: GRIB2 파일을 읽고, 한국 지역만 추출한 후 0.1° 격자로 선형 보간
    # 핵심 로직:
    # 1. xarray로 GRIB2 파일 로드
    # 2. 한국 지역 좌표 추출 (위도 33-39°N, 경도 124-132°E)
    # 3. 선형 보간으로 0.1° 격자 생성
    # 4. DataFrame으로 변환하여 반환
    # 주의: 모든 원본 변수는 보존됩니다 (10u, 10v 변수 포함)
```

### 3. grid_id 변환 함수

```python
def calculate_grid_id(df):
    """위도/경도를 grid_id로 변환합니다."""
    # 위치: 라인 438-456
    # 역할: 위경도 좌표를 grid_id로 변환
    # 계산식: grid_id = (lat_bin + 900) * 3600 + (lon_bin + 1800)
    # lat_bin = latitude / 0.1 (정수형)
    # lon_bin = longitude / 0.1 (정수형)
```

### 4. 풍속 계산 함수

```python
def calculate_wind10m(df_u10, df_v10):
    """u10와 v10 성분을 사용하여 wind10m 값을 계산합니다."""
    # 위치: 라인 457-487
    # 역할: 바람 성분(10u, 10v)으로 풍속 계산
    # 계산식: wind10m = √(u10² + v10²)
    # 중요: 원본 바람 성분(10u, 10v)은 최종 출력에 함께 포함됩니다
```

### 5. grid_id 필터링 함수

```python
def filter_by_grid_ids(df, grid_ids_file=None):
    """지정된 grid_id 목록에 해당하는 데이터만 필터링합니다."""
    # 위치: 라인 520-568
    # 역할: 유효한 grid_id 목록을 기준으로 데이터 필터링
    # 로직: df[df['grid_id'].isin(valid_grid_ids)]
```

### 6. 예보 시간별 처리 함수

```python
def process_forecast_step(target_date, step, forecast_dir, output_dir, grid_ids_file=None, visualize=False, interpolate=True):
    """특정 예보 시간의 모든 변수를 처리하고 통합합니다."""
    # 위치: 라인 569-665
    # 역할: 특정 예보 시간(예: D+1)의 모든 변수 데이터를 처리하여 통합 파일 생성
```

### 7. 전체 예보 처리 함수

```python
def process_all_forecast_steps(target_date=None, forecast_dir=None, output_dir=None, steps=None, grid_ids_file=None, visualize=False, interpolate=True):
    """모든 예보 시간을 처리합니다."""
    # 위치: 라인 666-705
    # 역할: 여러 예보 시간(D+1~D+5)을 순차적으로 처리
```

## 사용법

### 1. 데이터 수집 및 처리

```bash
# 오늘 날짜 기준 데이터 수집 및 처리
python src/data_collection/forecast/process_forecast_data.py --collect

# 특정 날짜 기준 데이터 수집 및 처리
python src/data_collection/forecast/process_forecast_data.py --collect --date 20240530

# 기존 수집 데이터만 처리
python src/data_collection/forecast/process_forecast_data.py --process_only

# 시각화 포함 처리
python src/data_collection/forecast/process_forecast_data.py --process_only --visualize
```

### 2. 상세 옵션

```bash
python src/data_collection/forecast/process_forecast_data.py --help
```

| 옵션               | 설명                          | 기본값                   |
| ------------------ | ----------------------------- | ------------------------ |
| `--date`           | 예보 기준 날짜 (YYYYMMDD)     | 현재 날짜                |
| `--forecast_dir`   | 예보 데이터 디렉토리          | data/forecast            |
| `--output_dir`     | 출력 디렉토리                 | data/processed/forecast  |
| `--steps`          | 처리할 예보 시간 목록         | 24 48 72 96 120          |
| `--collect`        | 예보 데이터 수집 여부         | False                    |
| `--process_only`   | 기존 다운로드된 데이터만 처리 | False                    |
| `--grid_ids_file`  | 유효 grid_id 목록 파일 경로   | korea_land_grid_ids.json |
| `--visualize`      | 격자 분포 시각화 생성 여부    | False                    |
| `--no_interpolate` | 보간 비활성화 (원본 해상도)   | False                    |

### 3. 출력 파일

처리된 데이터는 다음 형식의 파일로 저장됩니다:

```
forecast_{YYYYMMDD}_lead{N}.csv             # 주요 데이터 (CSV)
forecast_{YYYYMMDD}_lead{N}.parquet         # 주요 데이터 (Parquet)
forecast_{YYYYMMDD}_lead{N}_filter_summary.json  # 필터링 통계 정보
forecast_{YYYYMMDD}_lead{N}_grid_distribution.png # 격자 분포 시각화 (--visualize 옵션 사용 시)
```

예: `forecast_20240530_lead1.csv` - 2024년 5월 30일 발표된 D+1(내일) 예보

---

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

### 완전한 Forecast 변수 목록 (보류)

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

## 한국 지역 데이터 필터링 방법

`process_forecast_data.py` 스크립트는 전체 지구 데이터를 수집한 후 한국 지역 데이터만 필터링하여 처리합니다:

### 위경도 좌표계와 grid_id 변환

필터링된 데이터는 최종적으로 위경도 좌표에서 grid_id로 변환되어 다른 데이터셋과 결합됩니다:

```python
# 위도/경도를 0.1° 단위로 변환하여 grid_id 계산
df['lat_bin'] = (df['latitude'] / 0.1).astype(int)
df['lon_bin'] = (df['longitude'] / 0.1).astype(int)
df['grid_id'] = (df['lat_bin'] + 900) * 3600 + (df['lon_bin'] + 1800)
```

이렇게 변환된 grid_id는 연료 데이터, 지형 데이터, 산불 발생 데이터 등 다른 데이터셋과의 결합에 사용됩니다.

### 특정 grid_id 목록 필터링

위경도 기반 필터링 후, 최종적으로 미리 정의된 특정 grid_id 목록에 해당하는 데이터만 추출하는 기능이 추가되었습니다. 이는 다음과 같은 이점이 있습니다:

1. **육지 지역만 포함**: 한국 경계 내 해양 지역은 제외하고 육지 지역만 포함
2. **관심 지역 집중**: 산불 모델에 필요한 특정 산림 지역만 포함
3. **데이터 크기 감소**: 처리 및 저장 효율성 향상

#### 사용 방법

특정 grid_id 목록은 `korea_land_grid_ids.json` 파일에 JSON 배열로 저장되어 있으며, 다음과 같이 사용할 수 있습니다:

```bash
# 기본 grid_id 파일 사용
python src/data_collection/forecast/process_forecast_data.py --collect

# 사용자 정의 grid_id 파일 지정
python src/data_collection/forecast/process_forecast_data.py --collect --grid_ids_file path/to/custom_grid_ids.json
```

#### 기술적 구현

필터링은 `filter_by_grid_ids` 함수를 통해 이루어지며, 다음과 같이 작동합니다:

```python
def filter_by_grid_ids(df, grid_ids_file=None):
    """지정된 grid_id 목록에 해당하는 데이터만 필터링"""
    # grid_id 목록 로드
    with open(grid_ids_file, 'r') as f:
        valid_grid_ids = set(json.load(f))

    # grid_id로 필터링
    df = df[df['grid_id'].isin(valid_grid_ids)]

    return df
```

이 함수는 데이터 처리 과정의 마지막 단계에서 호출되어, 최종 출력 전에 특정 grid_id 목록에 해당하는 데이터만 보존합니다.

#### 필터링 검증 기능

필터링 과정이 제대로 수행되었는지 확인하기 위한 검증 기능이 추가되었습니다:

1. **필터링 통계 수집**:

   - 필터링 전/후 레코드 수
   - 기대 grid_id 개수 vs 실제 포함된 grid_id 개수
   - 누락된 grid_id 목록
   - 데이터에 있지만 정의되지 않은 추가 grid_id 목록

2. **결과 저장**:

   - `forecast_{날짜}_lead{N}_filter_summary.json`: 필터링 결과 전체 통계
   - `forecast_{날짜}_lead{N}_missing_grid_ids.json`: 누락된 grid_id 목록 (있는 경우)

3. **로그 출력**:
   - 기대 grid_id 수와 실제 포함된 수 비교
   - 누락된 grid_id가 있을 경우 경고 메시지
   - 누락 비율 계산 (누락된 grid_id 수 / 전체 grid_id 수)

이 검증 기능을 통해 데이터 수집 및 처리 과정에서 발생할 수 있는 누락을 쉽게 파악하고, 필요시 대응할 수 있습니다.
