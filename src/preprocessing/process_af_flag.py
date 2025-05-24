import pandas as pd
import numpy as np
import argparse
import os
from datetime import datetime

def process_af_flag(input_file, output_file, debug=True, debug_date='2024-01-15'):
    """
    MODIS 활성 화재 데이터를 처리하여 af_flag 데이터셋 생성
    
    Parameters:
    -----------
    input_file : str
        입력 파일 경로 (fire_archive_M-C61_613954.csv)
    output_file : str
        출력 파일 경로 (af_flag_korea.csv)
    debug : bool
        디버깅 정보 출력 여부
    debug_date : str
        특정 날짜에 대한 추적을 위한 날짜 문자열 (YYYY-MM-DD)
    """
    print(f"Processing MODIS Active Fire data from {input_file}")
    
    # 데이터 로드
    df = pd.read_csv(input_file)
    print(f"Loaded {len(df)} records from {input_file}")
    
    # 특정 날짜 디버깅
    debug_rows = df[df['acq_date'] == debug_date]
    print(f"\n>>> {debug_date} 날짜 데이터: {len(debug_rows)} 행")
    if len(debug_rows) > 0:
        print(debug_rows[['latitude', 'longitude', 'confidence']].to_string())
    
    if debug:
        print("\n=== 원본 데이터 샘플 ===")
        print(df.head())
        print("\n컬럼 목록:", df.columns.tolist())
        print(f"\nconfidence 값 범위: {df.confidence.min()} - {df.confidence.max()}")
        print(f"confidence 값 분포:\n{df.confidence.value_counts().sort_index().head(10)}")
    
    # confidence 30 이하 모두 제거
    print(f"\n필터링 전 레코드 수: {len(df)}")
    confidence_ge_30 = df.confidence >= 30
    print(f"confidence >= 30인 레코드 수: {confidence_ge_30.sum()} ({confidence_ge_30.mean()*100:.2f}%)")
    
    df = df[confidence_ge_30]
    print(f"필터링 후 레코드 수: {len(df)}")
    
    # 특정 날짜 디버깅 - 필터링 후
    debug_rows_after_filter = df[df['acq_date'] == debug_date]
    print(f"\n>>> {debug_date} 날짜 데이터 (필터링 후): {len(debug_rows_after_filter)} 행")
    if len(debug_rows_after_filter) > 0:
        print(debug_rows_after_filter[['latitude', 'longitude', 'confidence']].to_string())
    
    # 날짜 형식 변환
    df['acq_date'] = pd.to_datetime(df['acq_date'])
    
    # 특정 날짜 디버깅 - 날짜 변환 후
    debug_date_dt = pd.to_datetime(debug_date)
    debug_rows_after_date_conv = df[df['acq_date'] == debug_date_dt]
    print(f"\n>>> {debug_date} 날짜 데이터 (날짜 변환 후): {len(debug_rows_after_date_conv)} 행")
    if len(debug_rows_after_date_conv) > 0:
        print(debug_rows_after_date_conv[['latitude', 'longitude', 'confidence']].head().to_string())
    else:
        # 날짜 변환 후 데이터가 없다면 전체 날짜 목록에서 해당 날짜가 있는지 확인
        unique_dates = df['acq_date'].dt.strftime('%Y-%m-%d').unique()
        print(f"날짜 변환 후 고유 날짜 개수: {len(unique_dates)}")
        print(f"날짜 범위: {df['acq_date'].min()} ~ {df['acq_date'].max()}")
        print(f"{debug_date}가 변환된 날짜 목록에 있는지: {debug_date in unique_dates}")
        # 비슷한 날짜 찾기
        if len(unique_dates) > 0:
            similar_dates = [d for d in unique_dates if debug_date in d or d in debug_date]
            print(f"유사한 날짜: {similar_dates}")
    
    # 위/경도 범위 확인
    if debug:
        print(f"\n=== 위/경도 범위 ===")
        print(f"위도 범위: {df.latitude.min():.4f} - {df.latitude.max():.4f}")
        print(f"경도 범위: {df.longitude.min():.4f} - {df.longitude.max():.4f}")
    
    # 0.1도 격자 ID 생성
    df['lon_bin'] = np.floor(df.longitude / 0.1).astype(int)
    df['lat_bin'] = np.floor(df.latitude / 0.1).astype(int)
    
    # 전 지구 고유 ID (위/경 0.1도 셀을 행번호처럼 사용)
    # 행 개수 = 1,800개 (위도)
    # 열 개수 = 3,600개 (경도)
    df['grid_id'] = (df.lat_bin + 900) * 3600 + (df.lon_bin + 1800)
    
    # 특정 날짜 디버깅 - 격자 ID 생성 후
    debug_rows_after_grid = df[df['acq_date'] == debug_date_dt]
    print(f"\n>>> {debug_date} 날짜 데이터 (격자 ID 생성 후): {len(debug_rows_after_grid)} 행")
    if len(debug_rows_after_grid) > 0:
        print(debug_rows_after_grid[['latitude', 'longitude', 'lat_bin', 'lon_bin', 'grid_id']].to_string())
    
    if debug:
        print("\n=== 격자 ID 생성 결과 ===")
        print(df[['latitude', 'longitude', 'lat_bin', 'lon_bin', 'grid_id']].head(10))
        print(f"\n고유 격자 ID 개수: {df.grid_id.nunique()}")
    
    # 하루 및 격자별 열접 존재 여부 -> af_flag
    af = (df.groupby(['acq_date', 'grid_id'])
          .size()
          .reset_index(name='cnt')
          .assign(af_flag=1)[['acq_date', 'grid_id', 'af_flag']])
    
    # 특정 날짜 디버깅 - 그룹화 후
    debug_rows_after_group = af[af['acq_date'] == debug_date_dt]
    print(f"\n>>> {debug_date} 날짜 데이터 (그룹화 후 af_flag=1): {len(debug_rows_after_group)} 행")
    if len(debug_rows_after_group) > 0:
        print(debug_rows_after_group.to_string())
    
    if debug:
        print("\n=== af_flag=1 데이터 (그룹화 결과) ===")
        print(f"전체 화재 발생 레코드 수: {len(af)}")
        print(af.head(10))
    
    # 음성(0) 샘플 채우기 - 0.1도 격자 목록 & 날짜 달력과 outer-join
    grid_df = pd.DataFrame({'grid_id': np.unique(df.grid_id)})
    start_date = df['acq_date'].min().strftime('%Y-%m-%d')
    end_date = df['acq_date'].max().strftime('%Y-%m-%d')
    
    print(f"Creating calendar from {start_date} to {end_date}")
    dates_df = pd.DataFrame({'acq_date': pd.date_range(start_date, end_date)})
    
    # 특정 날짜 확인 - 달력 생성 후
    debug_date_in_calendar = debug_date_dt in dates_df['acq_date'].values
    print(f"\n>>> {debug_date}가 생성된 달력에 포함되어 있는지: {debug_date_in_calendar}")
    
    if debug:
        print(f"\n총 날짜 수: {len(dates_df)}")
        print(f"총 격자 수: {len(grid_df)}")
        print(f"총 조합 가능한 샘플 수: {len(dates_df) * len(grid_df):,}")
    
    # 모든 날짜-격자 조합 생성
    print(f"\n모든 날짜-격자 조합 생성 중...")
    full = (dates_df.assign(key=1)
            .merge(grid_df.assign(key=1), on='key')
            .drop('key', axis=1))
    
    print(f"생성된 전체 조합 수: {len(full):,}")
    
    # af_flag=1인 데이터와 병합하여 0인 경우 채우기
    print(f"\naf_flag=1 데이터 병합 중...")
    target = (full.merge(af, on=['acq_date', 'grid_id'], how='left')
              .fillna({'af_flag': 0})
              .astype({'af_flag': 'uint8'}))
    
    # 특정 날짜 디버깅 - 최종 타겟 데이터에서
    debug_rows_in_target = target[target['acq_date'] == debug_date_dt]
    print(f"\n>>> {debug_date} 날짜 데이터 (최종 타겟): {len(debug_rows_in_target)} 행")
    print(f">>> {debug_date} 날짜에 af_flag=1인 행 수: {debug_rows_in_target['af_flag'].sum()}")
    
    # 특정 날짜의 af_flag=1 데이터가 있어야 하지만 없는 경우 추가 디버깅
    if len(debug_rows_after_group) > 0 and debug_rows_in_target['af_flag'].sum() == 0:
        print("\n!!! 심각한 오류: 원본에는 있지만 최종 결과에서는 없는 af_flag=1 데이터 !!!")
        
        # 중간 조합 데이터에서 해당 날짜/격자가 있는지 확인
        debug_grid_ids = debug_rows_after_group['grid_id'].tolist()
        print(f"찾아야 할 격자 ID: {debug_grid_ids}")
        
        # full 데이터프레임에서 날짜와 격자 ID 조합이 존재하는지 확인
        debug_full_rows = full[
            (full['acq_date'] == debug_date_dt) & 
            (full['grid_id'].isin(debug_grid_ids))
        ]
        print(f"full 데이터프레임에서 해당 날짜/격자 조합 찾음: {len(debug_full_rows)} 행")
        
        # 전체 병합 과정을 작은 샘플로 재현하여 확인
        print("\n병합 과정 문제 확인을 위한 작은 샘플 병합 테스트:")
        small_full = full[(full['acq_date'] == debug_date_dt)].head(10)
        small_af = af[(af['acq_date'] == debug_date_dt)]
        print(f"작은 full 샘플: {len(small_full)} 행")
        print(f"작은 af 샘플: {len(small_af)} 행")
        small_merge = small_full.merge(small_af, on=['acq_date', 'grid_id'], how='left').fillna({'af_flag': 0})
        print(f"작은 병합 결과: {len(small_merge)} 행, af_flag=1 개수: {small_merge['af_flag'].sum()}")
        
        # 날짜 형식 문제 확인
        print("\n날짜 형식 비교:")
        full_date_sample = full['acq_date'].iloc[0]
        af_date_sample = af['acq_date'].iloc[0]
        print(f"full의 날짜 타입: {type(full_date_sample)}, 값: {full_date_sample}")
        print(f"af의 날짜 타입: {type(af_date_sample)}, 값: {af_date_sample}")
        
        # 정확한 타입 비교를 위해 각 샘플의 타임스탬프 출력
        try:
            print(f"full 날짜 타임스탬프: {pd.Timestamp(full_date_sample).value}")
            print(f"af 날짜 타임스탬프: {pd.Timestamp(af_date_sample).value}")
        except:
            print("타임스탬프 비교 실패")
    
    if debug:
        print("\n=== 최종 병합 결과 ===")
        print(f"af_flag=1 비율: {target.af_flag.mean()*100:.4f}%")
        # 데이터 검증: 화재가 있는 날짜와 격자가 최종 결과에 제대로 반영되었는지 확인
        merged_1_count = target[target.af_flag == 1].shape[0]
        original_1_count = af.shape[0]
        print(f"원본 af_flag=1 개수: {original_1_count}")
        print(f"최종 af_flag=1 개수: {merged_1_count}")
        if merged_1_count != original_1_count:
            print(f"!!! 경고: af_flag=1 개수가 병합 전후 일치하지 않습니다 !!!")
            # 불일치하는 데이터 샘플링
            print("\n--- af에 있지만 target에 없는 데이터 샘플 ---")
            missing = pd.merge(
                af, target[target.af_flag == 1][['acq_date', 'grid_id']], 
                on=['acq_date', 'grid_id'], 
                how='left', 
                indicator=True
            )
            missing = missing[missing._merge == 'left_only']
            if len(missing) > 0:
                print(missing.head(5))
    
    # 결과 저장
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    print(f"\n결과 저장 중: {output_file}")
    target.to_csv(output_file, index=False, encoding="utf-8")
    
    # 임시 진단용 파일 저장
    if debug:
        diag_dir = os.path.dirname(output_file)
        # 원본 af_flag=1 데이터 저장
        af_file = os.path.join(diag_dir, "af_flag_1_only.csv")
        af.to_csv(af_file, index=False)
        print(f"af_flag=1 데이터만 저장: {af_file}")
        
        # af_flag=1 데이터 샘플 저장
        sample_file = os.path.join(diag_dir, "af_flag_sample.csv")
        target_sample = pd.concat([
            target[target.af_flag == 1],  # 모든 양성 샘플
            target[target.af_flag == 0].sample(min(1000, len(target[target.af_flag == 0])))  # 최대 1000개 음성 샘플
        ])
        target_sample.to_csv(sample_file, index=False)
        print(f"샘플 데이터 저장: {sample_file}")
        
        # 특정 날짜 데이터 저장
        debug_file = os.path.join(diag_dir, f"af_flag_{debug_date.replace('-', '')}_debug.csv")
        debug_rows_in_target.to_csv(debug_file, index=False)
        print(f"{debug_date} 날짜 데이터 저장: {debug_file}")
    
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
    parser.add_argument('--debug', action='store_true',
                        help='디버깅 정보 출력')
    parser.add_argument('--debug-date', type=str, default='2024-01-15',
                        help='디버깅할 특정 날짜 (YYYY-MM-DD)')
    
    args = parser.parse_args()
    process_af_flag(args.input, args.output, args.debug, args.debug_date) 