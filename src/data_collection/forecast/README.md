# 기상 예보 데이터 수집 모듈

이 디렉토리는 다양한 기상 예보 데이터를 수집하기 위한 스크립트와 도구를 포함하고 있습니다.

## 파일 구조

- `collect_ecmwf_basic.py` - ECMWF 기본 예보 변수 수집 스크립트 (최신 개선 버전)
- `check_forecast_data.py` - GRIB2 파일 메타데이터 검사 스크립트

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

### 완전한 Forecast 변수 목록

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

### 주요 변수 목록

| 변수 코드 | 설명                       | 단위   | 비고                                        |
| --------- | -------------------------- | ------ | ------------------------------------------- |
| 2t        | 지상 2m 기온               | K      | 켈빈 온도 (°C = K - 273.15)                 |
| 2d        | 지상 2m 이슬점 온도        | K      | 켈빈 온도                                   |
| 10u       | 지상 10m 동서 바람 성분    | m/s    | 양수: 서→동, 음수: 동→서                    |
| 10v       | 지상 10m 남북 바람 성분    | m/s    | 양수: 남→북, 음수: 북→남                    |
| tp        | 총 강수량                  | m      | 누적값, 보통 mm로 변환(× 1000)              |
| cape      | 대류가용잠재에너지         | J/kg   | 대기 불안정성 지수                          |
| lightning | 번개 발생 확률             | 0-1    | 격자 내 발생 확률                           |
| tcc       | 전운량 (Total Cloud Cover) | 0-1    | 구름 양 비율 (0: 맑음, 1: 흐림)             |
| msdrswrf  | 직달 단파 복사             | J/m²   | 누적값                                      |
| msdwswrf  | 하향 단파 복사             | J/m²   | 누적값                                      |
| msdwlwrf  | 하향 장파 복사             | J/m²   | 누적값                                      |
| skt       | 지표면 온도                | K      | 켈빈 온도                                   |
| rh        | 상대습도                   | %      | 일반적으로 850hPa, 700hPa 등 특정 기압 필요 |
| z         | 지위 고도                  | m² s⁻² | 일반적으로 500hPa 등 특정 기압 필요         |

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

## 사용 방법

### 기본 예보 변수 수집

ECMWF에서 제공하는 기본 예보 변수(2t, 2d, 10u, 10v, tp) 수집:

```bash
python src/data_collection/forecast/collect_ecmwf_basic.py --output_dir data/forecast
```

### 테스트 모드

데이터 가용성을 먼저 테스트하려면:

```bash
python src/data_collection/forecast/collect_ecmwf_basic.py --test
```

### GRIB2 파일 검사

다운로드한 GRIB2 파일의 메타데이터 검사:

```bash
python src/data_collection/forecast/check_forecast_data.py data/forecast/2t/2t_20240521_00z_step024.grib2
```

## 의존성

- `ecmwf-opendata`: ECMWF Open Data API에 접근하기 위한 패키지
- `eccodes`: GRIB 파일 처리 라이브러리
- 설치:
  ```bash
  pip install ecmwf-opendata eccodes
  ```

## 참고 자료

- [ECMWF Open Data 문서](https://www.ecmwf.int/en/forecasts/datasets/open-data)
- [ecmwf-opendata 패키지](https://pypi.org/project/ecmwf-opendata/)
- [eccodes 라이브러리](https://confluence.ecmwf.int/display/ECC)
- [McNorton & Di Giuseppe 전역 연료 특성 모델](https://doi.org/10.24381/378d1497)

## 디버깅 및 문제 해결

### ECMWF API 404 오류 해결

ECMWF 데이터 수집 시 다음과 같은 404 에러가 발생할 수 있습니다:

```
ERROR - 테스트 실패: 404 Client Error: Not Found for url: https://data.ecmwf.int/forecasts/YYYYMMDD/00z/ifs/0p25/oper/YYYYMMDD000000-24h-oper-fc.index
```

이 오류는 주로 두 가지 원인에서 발생합니다:

1. **예보 데이터 공개 시점 문제**

   - ECMWF Open Data의 00 UTC 예보는 01 UTC 이후에 공개됩니다.
   - 예: 00Z 예보는 01 UTC, 06Z 예보는 07 UTC, 12Z 예보는 13 UTC, 18Z 예보는 19 UTC 이후에 다운로드 가능
   - 공개 전에 요청하면 404 오류가 발생합니다.

2. **과거 예보 데이터 접근 제한**
   - ECMWF Open Data는 실시간 및 단기예보 데이터만 제공하며, 과거 날짜의 예보는 보관하지 않습니다.
   - 과거 데이터가 필요한 경우 ECMWF WebAPI(MARS)나 회원 전용 채널을 통해 별도 라이선스가 필요합니다.

### 바람 변수 수집 실패 해결

바람 성분 변수가 다음 오류로 수집 실패하는 경우가 있습니다:

```
No index entries for param=v10
Did you mean '10u' instead of 'v10'?
```

**원인:** ECMWF API에서 바람 변수는 `u10`/`v10` 대신 `10u`/`10v` 형식의 이름을 사용합니다.

**해결책:** 변수명을 `10u`, `10v`로 수정하고 해당 이름의 디렉토리 생성 후 수집합니다.

### 해결 방법

문제 해결을 위해 `collect_ecmwf_basic.py` 스크립트를 다음과 같이 수정했습니다:

1. **`date=-1` 옵션 사용**

   - 특정 날짜를 지정하는 대신 ECMWF API의 `date=-1` 옵션을 사용하여 항상 가장 최신 예보 데이터를 가져옵니다.
   - 이 방식으로 현재 공개된 가장 최신 예보를 자동으로 다운로드합니다.

2. **낙뢰 데이터 제외**

   - `lightning` 변수는 일부 환경에서 접근 제한이 있을 수 있어 기본 변수 목록에서 제외했습니다.
   - 현재 기본 변수 목록: "2t", "2d", "10u", "10v", "tp"

3. **시스템 시간 의존성 제거**
   - 시스템 시간이 잘못 설정되어 있어도 작동하도록 수정했습니다.
   - 파일명은 현재 시스템 시간을 기준으로 생성하되, 실제 데이터는 최신 공개 예보를 사용합니다.

### 수정된 사용법

수정 후 권장되는 스크립트 사용법:

```bash
# 테스트 모드 (단일 변수 테스트)
python src/data_collection/forecast/collect_ecmwf_basic.py --test

# 전체 데이터 수집 (최신 예보 데이터 사용)
python src/data_collection/forecast/collect_ecmwf_basic.py
```

### 주의사항

- 스크립트는 항상 가장 최신 예보 데이터를 다운로드합니다.
- 특정 과거 날짜의 예보 데이터가 필요한 경우, ECMWF WebAPI(MARS)를 사용해야 합니다.
- API 호출 시 적절한 시간 간격을 두고 요청하는 것이 좋습니다.
