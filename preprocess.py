# import numpy as np

# def batch_norm()



import pandas as pd
import numpy as np



import numpy as np
import pandas as pd

def transform_custom_df(df, clip_label_outlier=True, fillna_label=True, 
                        clip_feature_outlier=True, shrink_feature_outlier=True, 
                        fillna_feature=True):
    # 建立副本防止改動原始數據
    df = df.copy()

    # ==========================================
    # 內部核心轉換函數 (去除 groupby，直接對 Series 處理)
    # ==========================================
    def _label_norm(x):
        x = x - x.mean()
        x /= x.std() if x.std() != 0 else 1
        if clip_label_outlier:
            x = x.clip(-3, 3)
        if fillna_label:
            x = x.fillna(0)
        return x

    def _feature_norm(x):
        median = x.median()
        mad = (x - median).abs().median() * 1.4826
        x = (x - median) / mad if mad != 0 else (x - median)
        
        if clip_feature_outlier:
            x = x.clip(-3, 3)
            
        if shrink_feature_outlier:
            x_max = x.max()
            x_min = x.min()
            if x_max > 3:
                x = x.where(x <= 3, 3 + (x - 3).div(x_max - 3) * 0.5)
            if x_min < -3:
                x = x.where(x >= -3, -3 - (x + 3).div(x_min + 3) * 0.5)
                
        if fillna_feature:
            x = x.fillna(0)
        return x

    # ==========================================
    # 根據欄位名稱關鍵字直接進行特徵工程與歸一化
    # ==========================================
    
    # 1. Label 處理
    cols = df.columns[df.columns.str.contains("^Return")]
    for col in cols:
        df[col] = _label_norm(df[col])

    # 2. KLEN, KLOW, KUP 處理 (0.25 次方 + MAD 歸一化)
    cols = df.columns[df.columns.str.contains("^KLEN|^KLOW|^KUP")]
    for col in cols:
        df[col] = _feature_norm(df[col].pow(0.25))

    # 3. KLOW2, KUP2 處理 (0.5 次方 + MAD 歸一化)
    cols = df.columns[df.columns.str.contains("^KLOW2|^KUP2")]
    for col in cols:
        df[col] = _feature_norm(df[col].pow(0.5))

    # 4. 常規技術指標欄位處理
    _cols = [
        "KMID", "KSFT", "OPEN", "HIGH", "LOW", "CLOSE", "VWAP", "ROC", "MA", 
        "BETA", "RESI", "QTLU", "QTLD", "RSV", "SUMP", "SUMN", "SUMD", "VSUMP", 
        "VSUMN", "VSUMD"
    ]
    pat = "|".join(["^" + x for x in _cols])
    cols = df.columns[df.columns.str.contains(pat) & (~df.columns.isin(["HIGH0", "LOW0"]))]
    for col in cols:
        df[col] = _feature_norm(df[col])

    # 5. 波動度與成交量處理 (Log 自然對數 + MAD 歸一化)
    cols = df.columns[df.columns.str.contains("^STD|^VOLUME|^VMA|^VSTD")]
    for col in cols:
        df[col] = _feature_norm(np.log(df[col].replace(0, np.nan))) # 防止 log(0) 報錯

    # 6. RSQR 處理
    cols = df.columns[df.columns.str.contains("^RSQR")]
    for col in cols:
        df[col] = _feature_norm(df[col].fillna(0))

    # 7. MAX, HIGH0 處理
    cols = df.columns[df.columns.str.contains("^MAX|^HIGH0")]
    for col in cols:
        df[col] = _feature_norm((df[col] - 1).pow(0.5))

    # 8. MIN, LOW0 處理
    cols = df.columns[df.columns.str.contains("^MIN|^LOW0")]
    for col in cols:
        df[col] = _feature_norm((1 - df[col]).pow(0.5))

    # 9. CORR, CORD 處理 (Exp 指數化 + MAD 歸一化)
    cols = df.columns[df.columns.str.contains("^CORR|^CORD")]
    for col in cols:
        df[col] = _feature_norm(np.exp(df[col]))

    # 10. WVMA 處理 (Log1p 變換 + MAD 歸一化)
    cols = df.columns[df.columns.str.contains("^WVMA")]
    for col in cols:
        df[col] = _feature_norm(np.log1p(df[col]))

    return df
