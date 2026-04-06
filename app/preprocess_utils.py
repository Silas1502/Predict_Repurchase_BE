"""
=========================================================
🚀 BACKEND MODULE - OnlineRetailPreprocessor & Data Access
=========================================================
Mô tả: Tiền xử lý dữ liệu Online Retail thành 35 features cho model prediction.
=========================================================
"""

import pandas as pd
import numpy as np
import joblib
from dateutil.relativedelta import relativedelta
import os
from typing import List, Dict

# SHAP availability check
try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False


class OnlineRetailPreprocessor:
    def __init__(self, final_features: List[str] = None, scaler_params: Dict = None):
        # 22 features sau feature selection
        self.final_features = final_features or [
            'active_months_L5M', 'avg_L1M_value', 'avg_L3M_value', 'avg_L5M_items_log',
            'avg_L5M_skus', 'avg_L5M_value', 'avg_gap_L5M', 'avg_items_per_cat_L3M',
            'avg_items_per_cat_L5M', 'cancel_rate_L5M', 'cnt_L1M_orders', 'cnt_L3M_orders',
            'cv_L5M_value', 'global_cancel_val_ratio', 'recency_days', 'spend_velocity',
            'std_L1M_value', 'std_L3M_value', 'sum_L1M_items_log', 'sum_L1M_value',
            'sum_L3M_items_log', 'sum_L3M_value', 'tenure_days'
        ]
        self.scaler_params = scaler_params or SCALER_PARAMS

    def robust_scale(self, value, median, scale):
        return (value - median) / scale if scale != 0 else 0

    def transform(self, transaction, snapshot_base):
        df = transaction.copy()
        df['Order_date'] = pd.to_datetime(df['Order_date'])

        result = []

        for s_date in snapshot_base['snapshot_date'].unique():
            s_date = pd.to_datetime(s_date)
            snap = snapshot_base[snapshot_base['snapshot_date'] == s_date].copy()
            hist = df[df['Order_date'] <= s_date].copy()

            if hist.empty:
                continue

            # TIME WINDOWS
            L1 = s_date - relativedelta(months=1)
            L3 = s_date - relativedelta(months=3)
            L5 = s_date - relativedelta(months=5)

            # GEOGRAPHY

            latest_geo = (
                hist.sort_values('Order_date')
                .groupby('Customer_id')['Country']
                .last()
                .reset_index(name='main_country')
            )

            # --- 1. AGGREGATION CHO L1M, L3M, L5M ---
            def agg_features(sub_df, name):
                if sub_df.empty:
                    return pd.DataFrame()
                
                sub_df = sub_df.copy()
                sub_df['items_per_cat'] = sub_df['Order_n_lines'] / (sub_df['Order_n_categories'] + 0.001)
                
                agg_dict = {
                    f'sum_{name}_value': ('Order_value', 'sum'),
                    f'avg_{name}_value': ('Order_value', 'mean'),
                    f'std_{name}_value': ('Order_value', 'std'),
                    f'cnt_{name}_orders': ('Order_id', 'count'),
                    f'avg_{name}_skus': ('Order_n_lines', 'mean'),
                    f'sum_{name}_items_log': ('Log_items', 'sum'),
                    f'avg_{name}_items_log': ('Log_items', 'mean'),
                    f'sum_{name}_canceled': ('Is_canceled', 'sum'),
                    f'avg_items_per_cat_{name}': ('items_per_cat', 'mean'),
                }
                
                # Chỉ tính n_categories cho L3M và L5M (L1M không cần cho feature selection)
                if name in ['L3M', 'L5M']:
                    agg_dict[f'avg_n_categories_{name}'] = ('Order_n_categories', 'mean')
                
                f = sub_df.groupby('Customer_id').agg(**agg_dict).reset_index()
                f[f'std_{name}_value'] = f[f'std_{name}_value'].fillna(0)
                return f

            hist['items_per_cat'] = hist['Order_n_lines'] / (hist['Order_n_categories'] + 0.001)
            f1 = agg_features(hist[hist['Order_date'] > L1], 'L1M')
            f3 = agg_features(hist[hist['Order_date'] > L3], 'L3M')
            f5 = agg_features(hist[hist['Order_date'] > L5], 'L5M')

            # --- 2. RHYTHM & ACTIVITY ---
            past_5m = hist[hist['Order_date'] > L5].copy()
            rhythm = pd.DataFrame(columns=['Customer_id', 'avg_gap_L5M'])
            activity = pd.DataFrame(columns=['Customer_id', 'active_months_L5M'])
            
            if not past_5m.empty:
                past_5m = past_5m.sort_values(['Customer_id', 'Order_date'])
                past_5m['gap'] = past_5m.groupby('Customer_id')['Order_date'].diff().dt.days
                
                rhythm = past_5m.groupby('Customer_id')['gap'].agg(avg_gap_L5M='mean').reset_index()
                activity = past_5m.groupby('Customer_id').agg(
                    active_months_L5M=('Order_date', lambda x: x.dt.to_period('M').nunique())
                ).reset_index()

            # --- 3. RECENCY, TENURE & RISK ---
            history_summary = hist.sort_values('Order_date').groupby('Customer_id').agg(
                min_date=('Order_date', 'min'),
                max_date=('Order_date', 'max'),
                global_aov=('Order_value', 'mean'),
                total_canceled_val=('Canceled_value', 'sum'),
                total_gross_val=('Order_value', lambda x: x[x > 0].sum())
            ).reset_index()
            
            history_summary['tenure_days'] = (s_date - history_summary['min_date']).dt.days
            history_summary['recency_days'] = (s_date - history_summary['max_date']).dt.days
            history_summary['global_cancel_val_ratio'] = history_summary['total_canceled_val'] / (history_summary['total_gross_val'] + 0.001)

            # --- 4. MERGE ---
            feat = snap[['Customer_id', 'snapshot_date']].copy()
            dfs_to_merge = [f1, f3, f5, rhythm, activity,
                            history_summary[['Customer_id', 'tenure_days', 'recency_days', 'global_cancel_val_ratio']]]

            for df_merge in dfs_to_merge:
                if not df_merge.empty:
                    feat = feat.merge(df_merge, on='Customer_id', how='left')

            result.append(feat)

        modeling_df = pd.concat(result, ignore_index=True)
        num_cols = modeling_df.select_dtypes(include=[np.number]).columns
        modeling_df[num_cols] = modeling_df[num_cols].fillna(0)

        # Đảm bảo các cột cần thiết tồn tại trước khi tính derived features
        required_cols = ['sum_L1M_value', 'sum_L3M_value', 'sum_L5M_value', 
                        'avg_L5M_value', 'std_L5M_value', 'cnt_L5M_orders', 
                        'sum_L5M_canceled', 'sum_L1M_items_log', 'avg_L5M_items_log',
                        'cnt_L1M_orders', 'cnt_L3M_orders']
        for col in required_cols:
            if col not in modeling_df.columns:
                modeling_df[col] = 0

        # --- 5. DERIVED FEATURES ---
        # cancel_rate_L5M
        modeling_df['cancel_rate_L5M'] = modeling_df['sum_L5M_canceled'] / (modeling_df['cnt_L5M_orders'] + 0.001)
        
        # cv_L5M_value
        modeling_df['cv_L5M_value'] = modeling_df['std_L5M_value'] / (modeling_df['avg_L5M_value'] + 0.001)
        
        # spend_velocity
        modeling_df['spend_velocity'] = np.where(
            modeling_df['sum_L3M_value'] > 0,
            modeling_df['sum_L1M_value'] / (modeling_df['sum_L3M_value'] / 3),
            0
        )

        # --- 6. CHỈ GIỮ 22 FEATURES ---
        # Đảm bảo tất cả cột tồn tại
        for col in self.final_features:
            if col not in modeling_df.columns:
                modeling_df[col] = 0

        return modeling_df[self.final_features].reset_index(drop=True)

    def transform_api_input(self, transactions, customer_id, snapshot_date):
        # Map API field names (snake_case) to DataFrame column names (PascalCase)
        column_mapping = {
            'order_id': 'Order_id',
            'total_items': 'Total_items',
            'log_items': 'Log_items',
            'order_date': 'Order_date',
            'order_value': 'Order_value',
            'canceled_value': 'Canceled_value',
            'order_n_categories': 'Order_n_categories',
            'order_n_lines': 'Order_n_lines',
            'is_canceled': 'Is_canceled',
            'country': 'Country'
        }
        
        df_transactions = pd.DataFrame(transactions)
        df_transactions = df_transactions.rename(columns=column_mapping)
        df_transactions['Customer_id'] = customer_id

        snapshot_base = pd.DataFrame({
            'Customer_id': [customer_id],
            'snapshot_date': [pd.to_datetime(snapshot_date)]
        })

        return self.transform(df_transactions, snapshot_base)

# ==========================================
# 🛠️ HELPER FUNCTIONS
# ==========================================

# Nếu shap không chạy được thì dùng global feature importance từ model

def get_top_reasons(feature_importance_df: pd.DataFrame, feature_values: pd.Series, n: int = 3) -> List[Dict]:

    """
    Lấy top n lý do ảnh hưởng đến kết quả dự đoán.

    Args:
        feature_importance_df: DataFrame chứa feature importance với cột 'Feature' và 'Importance'
        feature_values: Series chứa giá trị các feature
        n: Số lý do cần lấy

    Returns:
        List các dict chứa feature, importance_percent, value, impact

    """

    if feature_importance_df is None or feature_importance_df.empty:
        return []

    # Đảm bảo có cột 'Feature' và 'Importance' (chữ hoa như trong CSV)
    if 'Feature' not in feature_importance_df.columns or 'Importance' not in feature_importance_df.columns:
        # Thử kiểm tra tên cột chữ thường
        if 'feature' in feature_importance_df.columns and 'importance' in feature_importance_df.columns:
            feature_col = 'feature'
            importance_col = 'importance'
        else:
            return []

    else:
        feature_col = 'Feature'
        importance_col = 'Importance'

    # Sort by importance
    sorted_fi = feature_importance_df.sort_values(importance_col, ascending=False)

    # Calculate total importance
    total_importance = sorted_fi[importance_col].sum()

    # Danh sách features có tác động tích cực (càng cao càng tốt)
    positive_features = [
        'active_months_L5M', 'sum_L1M_value', 'sum_L3M_value', 'avg_L5M_value',
        'cnt_L1M_orders', 'cnt_L3M_orders', 'tenure_days', 'avg_gap_L5M',
        'spend_velocity', 'sum_L1M_items_log', 'avg_L5M_items_log'
    ]
        
    # Danh sách features có tác động tiêu cực (càng cao càng xấu)
    negative_features = [
        'recency_days', 'cancel_rate_L5M', 'global_cancel_val_ratio', 'cv_L5M_value'
    ]

    reasons = []
    for _, row in sorted_fi.head(n).iterrows():
        feature_name = row[feature_col]
        importance = row[importance_col]
        # Get feature value
        if feature_name in feature_values.index:
            value = feature_values[feature_name]
        else:
            value = 0.0

        # Xác định impact dựa trên loại feature và giá trị
        if feature_name in positive_features:
            impact = 'positive' if value > 0 else 'negative'
        elif feature_name in negative_features:
            impact = 'negative' if value > 0 else 'positive'
        else:
            # Mặc định: nếu giá trị cao thì tích cực
            impact = 'positive' if value > 0 else 'neutral'

        reasons.append({
            'feature': feature_name,
            'importance_percent': round((importance / total_importance) * 100, 2) if total_importance > 0 else 0.0,
            'value': round(float(value), 4),
            'impact': impact
        })     

    return reasons

# Lấy SHAP values cho từng prediction cá nhân

def get_shap_reasons(model, features_df: pd.DataFrame, feature_names: List[str], n: int = 3) -> List[Dict]:

    """
    Tính SHAP values cho từng prediction cá nhân.

    Args:
        model: Model XGBoost đã train
        features_df: DataFrame chứa features (1 row cho 1 khách hàng)
        feature_names: List tên các feature
        n: Số lý do cần lấy

    Returns:
        List các dict chứa feature, importance_percent, value, impact (positive/negative)

    """

    if not SHAP_AVAILABLE:
        print("SHAP not available, returning empty list")
        return []

    try:
        # Tạo SHAP explainer
        explainer = shap.TreeExplainer(model)

        # Tính SHAP values
        shap_values = explainer.shap_values(features_df.values)

        # Với XGBoost classifier, shap_values có thể là list (binary classification)
        if isinstance(shap_values, list):
            shap_values = shap_values[1]  # Lấy SHAP cho class 1 (mua lại)

        # Lấy SHAP values cho sample đầu tiên (vì chỉ có 1 khách hàng)
        shap_for_sample = shap_values[0] if len(shap_values.shape) > 1 else shap_values

        # Tạo list các (feature_name, shap_value, feature_value)
        feature_shap = []
        for i, (fname, sval) in enumerate(zip(feature_names, shap_for_sample)):
            fval = features_df.iloc[0, i] if i < len(features_df.columns) else 0
            feature_shap.append({
                'feature': fname,
                'shap_value': sval,
                'value': round(float(fval), 4),
                'impact': 'positive' if sval > 0 else 'negative'
            })

        # Sắp xếp theo absolute SHAP value (mức độ ảnh hưởng)
        feature_shap.sort(key=lambda x: abs(x['shap_value']), reverse=True)

        # Tính tổng absolute SHAP để normalize
        total_abs_shap = sum(abs(x['shap_value']) for x in feature_shap)

        # Format kết quả
        reasons = []
        for item in feature_shap[:n]:
            importance_pct = round((abs(item['shap_value']) / total_abs_shap) * 100, 2) if total_abs_shap > 0 else 0.0
            reasons.append({
                'feature': item['feature'],
                'importance_percent': importance_pct,
                'value': item['value'],
                'impact': item['impact'],
                'shap_value': round(float(item['shap_value']), 4)
            })

        return reasons

    except Exception as e:
        print(f"SHAP calculation error: {e}")
        # Fallback về global feature importance nếu SHAP fail
        return []

def load_preprocessor(path: str = './backend/models/preprocessor.pkl'):
    """
    Load preprocessor từ file pickle.  
    Args:
        path: Đường dẫn đến file preprocessor.pkl
    Returns:
        OnlineRetailPreprocessor instance
    """
    import __main__
    __main__.OnlineRetailPreprocessor = OnlineRetailPreprocessor
    return joblib.load(path)

def load_retail_preprocessor(path: str = './backend/models/preprocessor.pkl'):
    """Alias cho load_preprocessor"""
    return load_preprocessor(path)

# ==========================================
# 🎯 FINAL FEATURES CONFIG
# ==========================================

# 22 features sau feature selection
FINAL_FEATURES_22 = [
    'active_months_L5M', 'avg_L1M_value', 'avg_L3M_value', 'avg_L5M_items_log',
    'avg_L5M_skus', 'avg_L5M_value', 'avg_gap_L5M', 'avg_items_per_cat_L3M',
    'avg_items_per_cat_L5M', 'cancel_rate_L5M', 'cnt_L1M_orders', 'cnt_L3M_orders',
    'cv_L5M_value', 'global_cancel_val_ratio', 'recency_days', 'spend_velocity',
    'std_L1M_value', 'std_L3M_value', 'sum_L1M_items_log', 'sum_L1M_value',
    'sum_L3M_items_log', 'sum_L3M_value', 'tenure_days'
]

# Scaler params từ training (output từ build_online_retail_features)
SCALER_PARAMS = {
    'r_median': 29.0,
    'r_scale': 54.0,
    'f_median': 3.0,
    'f_scale': 3.0,
    'm_median': 783.67,
    'm_scale': 1134.1575
}

# Khởi tạo và save preprocessor
preprocessor = OnlineRetailPreprocessor(final_features=FINAL_FEATURES_22, scaler_params=SCALER_PARAMS)
os.makedirs('./backend/models', exist_ok=True)
joblib.dump(preprocessor, './backend/models/preprocessor.pkl')

print(f"🚀 Preprocessor sẵn sàng với {len(FINAL_FEATURES_22)} features!")
