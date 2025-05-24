# 데이터 전처리 가이드

이 디렉토리는 POF(Probability of Fire) 모델에 사용되는 데이터 전처리 스크립트를 포함하고 있습니다.

## 타겟 데이터(af_flag) 전처리

### 개요

`process_af_flag.py` 스크립트는 MODIS 활성 화재(Active Fire) 데이터를 전처리하여 머신러닝 모델의 타겟 변수로 사용할 af_flag 데이터를 생성합니다. 이 데이터는 특정 날짜와 지역(격자)에 화재가 발생했는지 여부를 0 또는 1로 표시합니다.

### 입력 데이터

- **형식**: CSV 파일 (MODIS Active Fire 데이터)
- **필수 컬럼**:
  - `latitude`: 위도 (도 단위)
  - `longitude`: 경도 (도 단위)
  - `acq_date`: 데이터 수집 날짜 (YYYY-MM-DD 형식)
  - `confidence`: 화재 감지 신뢰도 (0-100 사이 값)
- **위치**: `data/raw/DL_FIRE_M-C61_613954/fire_archive_M-C61_615377.csv` (기본 위치)

### 전처리 과정

전처리 과정은 다음 단계로 이루어집니다:

1. **데이터 로드 및 필터링**:

   - 원본 MODIS 화재 데이터를 로드합니다.
   - 신뢰도(confidence)가 30 이상인 레코드만 유지합니다.

2. **날짜 변환**:

   - 문자열 형태의 날짜를 datetime 객체로 변환합니다.

3. **격자 ID 생성**:

   - 위/경도 좌표를 0.1도 격자로 변환합니다.
   - 각 격자에 고유한 ID를 부여합니다.
   - 계산식: `grid_id = (lat_bin + 900) * 3600 + (lon_bin + 1800)`
   - 이는 위도(-90 ~ 90)와 경도(-180 ~ 180)를 고려한 전 지구 고유 ID입니다.

4. **데이터 통합**:

   - 같은 날짜와 같은 격자 내의 여러 화재 감지를 하나의 af_flag=1로 통합합니다.
   - 즉, '특정 날짜에 특정 격자에 화재가 있었는가'라는 이진 플래그로 변환합니다.

5. **전체 날짜-격자 조합 생성**:

   - 데이터셋의 최소 날짜부터 최대 날짜까지의 모든 날짜를 생성합니다.
   - 관측된 모든 고유 격자 ID 목록을 생성합니다.
   - 모든 날짜와 모든 격자의 모든 조합을 생성합니다(cartesian product).

6. **최종 데이터셋 생성**:
   - 모든 날짜-격자 조합에 대해, 화재가 관측된 조합은 af_flag=1, 나머지는 af_flag=0으로 설정합니다.
   - 결과를 CSV 파일로 저장합니다.

### 특징 및 중요 포인트

- **데이터 통합**: 같은 날짜와 격자 내의 여러 화재 감지는 하나의 af_flag=1로 통합됩니다. 예를 들어, 같은 날 같은 격자에서 3개의 화재가 감지되더라도 최종 데이터셋에는 하나의 af_flag=1만 기록됩니다.

- **전체 조합 생성**: 전체 기간의 모든 날짜와 모든 격자의 조합을 생성하기 때문에, 최종 데이터셋은 매우 클 수 있습니다. 예: 2000일 × 1000격자 = 2백만 행

- **클래스 불균형**: 화재는 드문 현상이므로, 최종 데이터셋에서 af_flag=1의 비율은 매우 낮습니다(약 0.1% 내외).

### 출력 데이터

- **형식**: CSV 파일
- **출력 컬럼**:
  - `acq_date`: 날짜 (YYYY-MM-DD 형식)
  - `grid_id`: 격자 고유 ID (정수)
  - `af_flag`: 화재 발생 여부 (0 또는 1)
- **기본 출력 경로**: `data/reference/af_flag_korea.csv`

### 사용 방법

기본 사용법:

```bash
python src/preprocessing/process_af_flag.py --input <입력파일> --output <출력파일>
```

예시:

```bash
python src/preprocessing/process_af_flag.py --input data/raw/DL_FIRE_M-C61_613954/fire_archive_M-C61_615377.csv --output data/reference/af_flag_full_combine.csv
```

### 디버깅 모드

디버깅 모드를 사용하면 전처리 과정의 각 단계에서 상세한 정보를 출력하고, 추가적인 진단 파일을 생성합니다.

```bash
python src/preprocessing/process_af_flag.py --input <입력파일> --output <출력파일> --debug
```

특정 날짜의 처리 과정을 추적하려면:

```bash
python src/preprocessing/process_af_flag.py --input <입력파일> --output <출력파일> --debug --debug-date 2024-01-15
```

디버깅 모드에서 생성되는 추가 파일:

- `af_flag_1_only.csv`: af_flag=1인 데이터만 포함
- `af_flag_sample.csv`: 모든 af_flag=1 데이터와 일부 af_flag=0 데이터 샘플
- `af_flag_YYYYMMDD_debug.csv`: 특정 날짜(YYYYMMDD)의 모든 데이터

### 데이터 검증

전처리 결과의 정확성을 확인하기 위한 주요 체크포인트:

1. **원본 데이터 확인**:

   ```bash
   python -c "import pandas as pd; df = pd.read_csv('데이터경로'); print(df[df['acq_date'] == '2024-01-15'][['latitude', 'longitude', 'confidence']])"
   ```

2. **최종 데이터 확인**:
   ```bash
   python -c "import pandas as pd; df = pd.read_csv('데이터경로'); print(f'af_flag=1 개수: {df.af_flag.sum()}'); print(f'2024-01-15 날짜의 af_flag=1 개수: {df[df.acq_date == \"2024-01-15\"].af_flag.sum()}')"
   ```
