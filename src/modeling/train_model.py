import os 
import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 
import seaborn as sns 
from sklearn.model_selection import train_test_split 
import xgboost as xgb 
from sklearn.metrics import roc_auc_score, confusion_matrix, classification_report 
import time 
 
def main(): 
    """?�속 변?��? 추�????�이?��? ?�용?�여 기본 XGBoost 모델 ?�련""" 
    start_time = time.time() 
    print("=== ?�불 ?�측 모델 ?�습 ?�작 ===") 
 
    # 결과 ?�렉?�리 ?�성 
    model_dir = "../../outputs/models" 
    os.makedirs(model_dir, exist_ok=True) 
    os.makedirs(os.path.join(model_dir, "plots"), exist_ok=True) 
 
    # ?�이??로드 
    input_file = "../../outputs/data/weather_data.csv" 
    print(f"?�이??로드 �? {input_file}") 
    df = pd.read_csv(input_file, parse_dates=['acq_date']) 
 
 
    # 모델 ?�일 경로 ?�정 
    model_file = f"{model_dir}/xgboost_weather_model.json" 
    model.save_model(model_file) 
    print(f"모델 ?�???�료: {model_file}") 
 
    # ?�성 중요???�각???�??경로 ?�정 
    plt.savefig(f"{model_dir}/plots/feature_importance.png") 
    print(f"?�성 중요???�각???�?? {model_dir}/plots/feature_importance.png") 
 
    # ?�요 ?�간 출력 
    processing_time = time.time() - start_time 
    print("=== 모델 ?�습 ?�료 ===") 
 
if __name__ == "__main__": 
    main() 
