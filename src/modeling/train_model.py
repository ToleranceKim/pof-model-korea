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
    """?ì† ë³€?˜ê? ì¶”ê????°ì´?°ë? ?¬ìš©?˜ì—¬ ê¸°ë³¸ XGBoost ëª¨ë¸ ?ˆë ¨""" 
    start_time = time.time() 
    print("=== ?°ë¶ˆ ?ˆì¸¡ ëª¨ë¸ ?™ìŠµ ?œì‘ ===") 
 
    # ê²°ê³¼ ?”ë ‰? ë¦¬ ?ì„± 
    model_dir = "../../outputs/models" 
    os.makedirs(model_dir, exist_ok=True) 
    os.makedirs(os.path.join(model_dir, "plots"), exist_ok=True) 
 
    # ?°ì´??ë¡œë“œ 
    input_file = "../../outputs/data/weather_data.csv" 
    print(f"?°ì´??ë¡œë“œ ì¤? {input_file}") 
    df = pd.read_csv(input_file, parse_dates=['acq_date']) 
 
 
    # ëª¨ë¸ ?Œì¼ ê²½ë¡œ ?˜ì • 
    model_file = f"{model_dir}/xgboost_weather_model.json" 
    model.save_model(model_file) 
    print(f"ëª¨ë¸ ?€???„ë£Œ: {model_file}") 
 
    # ?¹ì„± ì¤‘ìš”???œê°???€??ê²½ë¡œ ?˜ì • 
    plt.savefig(f"{model_dir}/plots/feature_importance.png") 
    print(f"?¹ì„± ì¤‘ìš”???œê°???€?? {model_dir}/plots/feature_importance.png") 
 
    # ?Œìš” ?œê°„ ì¶œë ¥ 
    processing_time = time.time() - start_time 
    print("=== ëª¨ë¸ ?™ìŠµ ?„ë£Œ ===") 
 
if __name__ == "__main__": 
    main() 
