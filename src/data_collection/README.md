# 데이터 수집 안내

PoF(Probability of Fire) 모델을 위한 데이터 수집 스크립트 디렉토리입니다.

## 1. 타겟(열점) 데이터 수집

### 개요다

타겟 데이터(`af_flag`)는 산불 발생 여부를 나타내는 지표입니다.

### 데이터 소스 옵션

#### 1. FIRMS REST API

- NASA 실시간 데이터 (위성 통과 후 10분 내 배포)
- URL 호출 만으로 날짜 및 BBOX 지정하여 내려받을 수 있어 자동화 가능
- 'I/n/h'의 confidence 필드 존재
- API 가이드: [FIRMS API](https://firms.modaps.eosdis.nasa.gov/api/)

#### 2. SFTP Archive

- 매달 한 번 품질 보정(QC) 및 위치 재조정 완료 데이터 - 결측 및 오검출이 적음
- confidence 필드가 0-100점수로 매핑
- 2000-02-24 이후 전 기간이 연.월 폴더 구조로 정리되어 있음
- 다운로드 링크: [FIRMS Download](https://firms.modaps.eosdis.nasa.gov/download/)
- SFTP User 가이드 참조하여 다운로드 혹은 사이트에서 신청 (1시간 내로 승인 되었음)

### 수집 후 처리

산불 데이터는 `src/preprocessing/process_af_flag.py` 스크립트를 통해 전처리됩니다.

## 2. 날씨 데이터 수집

### 개요

- 날씨 변수: `t2m`(2m 기온), `td2m`(2m 이슬점 온도), `10u`(10m 동서 풍속), `10v`(10m 남북 풍속), `tp`(강수량)
- 데이터 출처: ERA5-Land hourly data from 1950 to present
- API 링크: [ERA5-Land 데이터셋](https://cds.climate.copernicus.eu/datasets/reanalysis-era5-land?tab=download)

### API 키 설정

#### 보안 관련 중요 사항

Copernicus Climate Data Store API 키는 민감 정보이므로 `.env` 파일을 통해 안전하게 관리합니다.
이 방식을 사용하면 API 키가 코드에 직접 포함되거나 원격 저장소에 실수로 업로드되는 것을 방지할 수 있습니다.

#### .env 파일 설정 방법

1. 프로젝트 루트 디렉토리에 `.env` 파일을 생성합니다
2. 다음 내용을 입력합니다 (여러분의 API 키로 대체):

```
# Copernicus Climate Data Store API 설정
CDS_API_URL=https://cds.climate.copernicus.eu/api
CDS_API_KEY=your-api-key-here
```

3. `.env` 파일은 `.gitignore`에 포함되어 있으므로 원격 저장소에 업로드되지 않습니다.

#### API 키 발급 방법

1. [Copernicus Climate Data Store](https://cds.climate.copernicus.eu/) 계정 생성
2. 로그인 후 우측 상단의 프로필을 클릭하고 'API Key' 메뉴 선택
3. 'Show API key' 버튼을 클릭하여 본인의 API 키 확인

### 수집 방법

1. 위에서 설명한 대로 `.env` 파일에 API 키를 설정합니다
2. `collect_weather.py` 스크립트를 실행하여 데이터 수집

```bash
# 날씨 데이터 수집 예시 (2024년 1월부터 12월까지)
python collect_weather.py --start_year 2024 --end_year 2024 --start_month 1 --end_month 12
```

### 코드 예시

```python
# collect_weather.py에 구현된 코드와 유사합니다
import cdsapi
import calendar
from dotenv import load_dotenv
import os

# .env 파일에서 환경 변수 로드
load_dotenv()

# API 클라이언트 초기화 (인증은 자동으로 처리됨)
c = cdsapi.Client()

# 수집 기간 설정
years  = ["2025"]
months = [f"{m:02d}" for m in range(1,5)]
days   = [f"{d:02d}" for d in range(1,32)]
times  = [f"{h:02d}:00" for h in range(0,24)]
bbox   = [38, 124, 33, 132]  # [북위, 서경, 남위, 동경]

for year in years:
    for month in months:
        # 해당 월의 실제 일수만 추리기
        max_day = calendar.monthrange(int(year), int(month))[1]
        day_list = [f"{d:02d}" for d in range(1, max_day+1)]

        target_file = f"era5_korea_{year}{month}.nc"
        print(f"Retrieving {target_file} ...")

        c.retrieve(
            'reanalysis-era5-land',
            {
                'variable': [
                    '2m_temperature','2m_dewpoint_temperature',
                    '10m_u_component_of_wind','10m_v_component_of_wind',
                    'total_precipitation',
                ],
                'product_type': 'reanalysis',
                'year':   [year],
                'month':  [month],
                'day':    day_list,
                'time':   times,
                'area':   bbox,
                'format': 'netcdf'
            },
            target_file
        )
```

## 3. 낙뢰 데이터 수집 제약사항

### 1. 데이터 가용성 문제

- 낙뢰 플래시 밀도(파라미터 228050, shortName=litoti)는 공개 CDS API를 통해 접근할 수 없음
- 해당 변수는 ECMWF 백엔드 데이터 명세서에는 존재하나, 공개 API 파라미터 목록에는 포함되어 있지 않음
- 이는 Copernicus CDS의 공개 데이터셋(reanalysis-era5-complete)에 해당 변수가 포함되지 않기 때문

### 2. 접근 권한 요구사항

ECMWF 운영 예보 데이터에 접근하기 위해서는 다음 중 하나의 자격이 필요:

1. ECMWF 회원국 및 협력국 사용자

   - 국가기상청(NMHS) 소속자 또는 그 권한을 받은 사용자만 접근 가능
   - 별도 비용 없이 전체 아카이브 접근 가능
   - Computing Representative를 통해 접근 권한 획득 필요

2. WMO 회원국 연구자

   - 국가기상청(NMHS) 소속 연구자는 "non-commercial licence" 신청 가능
   - 연구 및 검증 목적으로 실시간(valid) 예보 제품 무료 사용 가능
   - 라이선스 신청 및 승인 절차 필요

3. 상업 사용자
   - Commercial Web Products Standard Licence 구매 필요
   - 연간 사용료 EUR 6,000 (소기업 50% 할인 가능)
   - 연구 목적 시 면제 가능성 있음 (별도 심사)

### 3. 대체 데이터 소스

1. AWS Open Data

   - "ecmwf-forecasts" S3 버킷을 통해 일부 실시간 예보 파라미터 무료 제공
   - 단, 낙뢰 파라미터 포함 여부 확인 필요

2. 대체 모델링 방법
   - Proxy 모델: CAPE, 대류강수량 등을 이용한 parameterization
   - 위성/지상 관측 데이터: WWLLN, LIS/OTD 등

### 4. 참고사항

- PoF 논문에서 사용된 낙뢰 데이터는 ECMWF 운영 예보 MARS 데이터를 통해 수집됨
- Copernicus CDS API만으로는 해당 데이터 수집이 불가능
- MarsNoDataError는 권한 문제가 아닌, 요청한 데이터셋에 해당 변수가 존재하지 않아서 발생

## 기타 데이터 수집
