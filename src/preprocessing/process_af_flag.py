import pandas as pd
import numpy as np
import argparse
import os
from datetime import datetime

def process_af_flag(input_file, output_file):
    """
    MODIS 활성 화재 데이터를 처리하여 af_flag 데이터셋 생성
    
    Parameters:
    -----------
    input_file : str
        입력 파일 경로 (fire_archive_M-C61_613954.csv)
    output_file : str
        출력 파일 경로 (af_flag_korea.csv)
    """
    print(f"Processing MODIS Active Fire data from {input_file}")
    
    # 데이터 로드
    df = pd.read_csv(input_file)
    print(f"Loaded {len(df)} records from {input_file}")
    
    # confidence 30 이하 모두 제거
    df = df[df.confidence >= 30]
    print(f"Filtered to {len(df)} records with confidence >= 30")
    
    # 날짜 형식 변환
    df['acq_date'] = pd.to_datetime(df['acq_date'])
    
    # 0.1도 격자 ID 생성
    df['lon_bin'] = np.floor(df.longitude / 0.1).astype(int)
    df['lat_bin'] = np.floor(df.latitude / 0.1).astype(int)
    
    # 전 지구 고유 ID (위/경 0.1도 셀을 행번호처럼 사용)
    # 행 개수 = 1,800개 (위도)
    # 열 개수 = 3,600개 (경도)
    df['grid_id'] = (df.lat_bin + 900) * 3600 + (df.lon_bin + 1800)
    
    # 하루 및 격자별 열접 존재 여부 -> af_flag
    af = (df.groupby(['acq_date', 'grid_id'])
          .size()
          .reset_index(name='cnt')
          .assign(af_flag=1)[['acq_date', 'grid_id', 'af_flag']])
    
    # 음성(0) 샘플 채우기 - 0.1도 격자 목록 & 날짜 달력과 outer-join
    grid_df = pd.DataFrame({'grid_id': np.unique(df.grid_id)})
    start_date = df['acq_date'].min().strftime('%Y-%m-%d')
    end_date = df['acq_date'].max().strftime('%Y-%m-%d')
    
    print(f"Creating calendar from {start_date} to {end_date}")
    dates_df = pd.DataFrame({'acq_date': pd.date_range(start_date, end_date)})
    
    # 모든 날짜-격자 조합 생성
    full = (dates_df.assign(key=1)
            .merge(grid_df.assign(key=1), on='key')
            .drop('key', axis=1))
    
    # af_flag=1인 데이터와 병합하여 0인 경우 채우기
    target = (full.merge(af, on=['acq_date', 'grid_id'], how='left')
              .fillna({'af_flag': 0})
              .astype({'af_flag': 'uint8'}))
    
    # 결과 저장
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    target.to_csv(output_file, index=False, encoding="utf-8")
    
    # 요약 정보 출력
    print(f"\nProcessed Active Fire data:")
    print(f"Total samples: {len(target)}")
    print(f"Positive samples (af_flag=1): {target.af_flag.sum()} ({target.af_flag.mean()*100:.4f}%)")
    print(f"Date range: {target.acq_date.min()} to {target.acq_date.max()}")
    print(f"Grid ID range: {target.grid_id.min()} to {target.grid_id.max()}")
    print(f"\nOutput saved to: {output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='MODIS Active Fire 데이터 전처리')
    parser.add_argument('--input', type=str, required=True,
                        help='입력 파일 경로 (fire_archive_M-C61_613954.csv)')
    parser.add_argument('--output', type=str, default='../../data/reference/af_flag_korea.csv',
                        help='출력 파일 경로 (af_flag_korea.csv)')
    
    args = parser.parse_args()
    process_af_flag(args.input, args.output) 