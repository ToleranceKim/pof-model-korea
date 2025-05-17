# 프로젝트 구조 개선 계획

## 현재 문제점

- 여러 디렉토리와 스크립트가 혼재되어 있어 전체 흐름 파악이 어려움
- 임시 파일과 주요 파일의 구분이 불명확함
- 파이프라인 실행 과정과 단계별 흐름이 직관적이지 않음

## 개선된 디렉토리 구조 제안

```
pof-model-test/
│
├── README.md                  # 프로젝트 개요 및 사용법
├── requirements.txt           # 필요 패키지 목록 [신규]
├── run_pipeline.bat           # 전체 파이프라인 실행 스크립트
│
├── data/                      # 원시 데이터 저장
│   ├── raw/                   # 수집된 원본 데이터 [신규]
│   │   └── era5_korea_*.nc    # ERA5 날씨 데이터
│   └── reference/             # 참조 데이터 [신규]
│       └── af_flag_korea.csv  # 산불 발생 데이터
│
├── src/                       # 소스 코드 [신규]
│   ├── data_collection/       # 데이터 수집 관련 코드
│   │   └── collect_weather.py # 날씨 데이터 수집
│   │
│   ├── preprocessing/         # 데이터 전처리 관련 코드
│   │   └── process_weather.py # 날씨 데이터 전처리(simpler_processing.py 리네임)
│   │
│   └── modeling/              # 모델링 관련 코드
│       └── train_model.py     # 모델 훈련(start_modeling.py 리네임)
│
├── processed_data/            # 전처리된 중간 데이터
│   └── era5_korea_*.csv       # 전처리된 날씨 데이터
│
└── outputs/                   # 최종 결과물 [신규]
    ├── data/                  # 최종 데이터셋
    │   └── weather_data.csv   # 모델링용 최종 데이터
    │
    └── models/                # 학습된 모델 및 결과
        ├── xgboost_model.json # 저장된 모델
        └── plots/             # 결과 시각화
            └── feature_importance.png  # 특성 중요도 시각화
```

## 구현 단계

1. **디렉토리 구조 개선**

   - 새로운 디렉토리 생성 (src, outputs 등)
   - 파일 적절한 위치로 이동

2. **코드 정리**

   - 파일 이름 명확하게 변경 (예: simpler_processing.py → process_weather.py)
   - 주석 및 문서화 개선

3. **실행 스크립트 개선**

   - 실행 경로 업데이트
   - 오류 처리 강화

4. **문서화**

   - README.md 업데이트로 새 구조 설명
   - requirements.txt 작성

5. **중복/불필요 파일 제거**
   - 임시 파일, 중복 스크립트, 불필요한 노트북 정리
