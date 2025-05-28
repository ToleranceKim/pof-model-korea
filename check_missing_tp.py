import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def check_tp_missing_values():
    """ERA5 데이터에서 tp(강수량) 변수의 누락값을 확인하는 함수"""
    
    # 데이터 경로 설정
    data_dir = 'processed_data/era5_daily'
    
    # 결과 저장 디렉토리
    result_dir = 'analysis_results'
    os.makedirs(result_dir, exist_ok=True)
    
    print("ERA5 데이터에서 tp 변수의 누락값 확인\n")
    
    # 분석 결과 저장용 데이터프레임 초기화
    results = []
    
    # 데이터 파일 목록 가져오기
    data_files = [f for f in os.listdir(data_dir) if f.endswith('.csv')]
    data_files.sort()  # 시간순 정렬
    
    print(f"총 {len(data_files)}개 파일 분석 시작...\n")
    
    # 샘플 파일 수 (처리 시간 단축을 위해)
    sample_size = min(5, len(data_files))
    
    for i, file in enumerate(data_files[:sample_size]):
        file_path = os.path.join(data_dir, file)
        print(f"파일 분석 중 ({i+1}/{sample_size}): {file}")
        
        try:
            # 파일 로드 (메모리 효율성을 위해 필요한 열만 로드)
            df = pd.read_csv(file_path, usecols=['acq_date', 'grid_id', 'tp'])
            
            # 파일 크기 확인
            total_rows = len(df)
            missing_tp = df['tp'].isna().sum()
            missing_percentage = (missing_tp / total_rows) * 100 if total_rows > 0 else 0
            
            # 결과 저장
            file_result = {
                'file': file,
                'total_rows': total_rows,
                'missing_tp': missing_tp,
                'missing_percentage': missing_percentage
            }
            results.append(file_result)
            
            # 월별 데이터 결과 출력
            print(f"  - 총 행: {total_rows:,}")
            print(f"  - tp 누락값: {missing_tp:,} ({missing_percentage:.2f}%)")
            
            # 누락값이 있는 경우, 패턴 분석
            if missing_tp > 0:
                # 샘플 데이터 출력
                missing_sample = df[df['tp'].isna()].head(5)
                print("\n  tp 누락값 샘플:")
                print(missing_sample)
                
                # 날짜별 누락값 분포
                date_missing = df[df['tp'].isna()].groupby('acq_date').size()
                print(f"\n  날짜별 tp 누락값 분포 (상위 3개):")
                print(date_missing.sort_values(ascending=False).head(3))
                
                # grid_id별 누락값 분포
                grid_missing = df[df['tp'].isna()].groupby('grid_id').size()
                print(f"\n  grid_id별 tp 누락값 분포 (상위 3개):")
                print(grid_missing.sort_values(ascending=False).head(3))
            
            print("-" * 50)
            
        except Exception as e:
            print(f"  오류 발생: {str(e)}")
    
    # 결과 데이터프레임 생성
    results_df = pd.DataFrame(results)
    
    if not results_df.empty:
        # 결과 요약
        print("\n=== tp 누락값 분석 요약 ===")
        print(f"분석된 파일 수: {len(results_df)}")
        print(f"평균 tp 누락값 비율: {results_df['missing_percentage'].mean():.2f}%")
        print(f"최대 tp 누락값 비율: {results_df['missing_percentage'].max():.2f}%")
        print(f"최소 tp 누락값 비율: {results_df['missing_percentage'].min():.2f}%")
        
        # 누락값이 많은 파일 확인
        if results_df['missing_percentage'].max() > 0:
            max_missing_file = results_df.loc[results_df['missing_percentage'].idxmax()]
            print(f"\n누락값이 가장 많은 파일: {max_missing_file['file']} ({max_missing_file['missing_percentage']:.2f}%)")
        
        # 가능한 원인 분석
        print("\n=== 누락 원인 분석 ===")
        
        # 전체 데이터의 누락값 패턴 확인
        if results_df['missing_percentage'].mean() > 0:
            print("1. 데이터 원본 문제:")
            print("   - ERA5 원본 데이터에서 일부 시간대나 지역의 강수량 데이터가 누락되었을 수 있습니다.")
            print("   - 위성 관측이나 측정 기기의 한계로 일부 값들이 기록되지 않았을 가능성이 있습니다.")
            
            print("\n2. 처리 과정 문제:")
            print("   - 데이터 처리 과정에서 특정 조건에 해당하는 값들이 누락으로 처리되었을 수 있습니다.")
            print("   - 보간 과정에서 경계 지역이나 특정 조건의 데이터가 누락되었을 수 있습니다.")
            
            print("\n3. 0값 처리 문제:")
            print("   - 강수량이 0인 경우 누락값(NaN)으로 처리되었을 가능성이 있습니다.")
            print("   - 매우 작은 값(threshold 이하)이 0 또는 누락값으로 처리되었을 수 있습니다.")
        
        # 추가 분석 및 해결 방안 제안
        print("\n=== 해결 방안 ===")
        print("1. 누락값 대체:")
        print("   - 시공간적 보간법을 사용하여 누락된 강수량 값을 예측하여 채울 수 있습니다.")
        print("   - 인접 격자점의 값을 사용한 보간이 가능합니다.")
        
        print("\n2. 원본 데이터 확인:")
        print("   - ERA5 원본 데이터에서 해당 지점/시간의 값을 확인하여 실제 누락인지 처리 과정의 문제인지 파악합니다.")
        
        print("\n3. 0값 처리 방식 확인:")
        print("   - 처리 과정에서 0값 또는 매우 작은 값의 처리 방식을 확인하고 필요시 조정합니다.")
        
        # 최종 제안
        print("\n=== 최종 권장사항 ===")
        if results_df['missing_percentage'].mean() < 1:
            print("누락값 비율이 1% 미만으로 낮으므로, 분석에 큰 영향을 미치지 않을 것으로 예상됩니다.")
            print("필요시 0 또는 인접 값을 사용한 간단한 대체 방법을 사용할 수 있습니다.")
        else:
            print("누락값 비율이 상당하므로, 시공간적 보간법이나 머신러닝 기법을 활용한 더 정교한 대체 방법을 검토하는 것이 좋습니다.")
            print("또한 원본 데이터와 처리 과정을 재검토하여 누락 원인을 정확히 파악하는 것을 권장합니다.")

if __name__ == "__main__":
    check_tp_missing_values() 