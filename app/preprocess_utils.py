"""
=========================================================
🚀 BACKEND MODULE - OnlineRetailPreprocessor & Data Access
=========================================================
Mô tả: Tiền xử lý dữ liệu Online Retail thành 23 features.
=========================================================
"""

import pandas as pd
import numpy as np
import joblib
import os
try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    shap = None
    SHAP_AVAILABLE = False
from dateutil.relativedelta import relativedelta
from typing import List, Dict, Union, Optional


class OnlineRetailPreprocessor:
    """
    Preprocessor cho Online Retail data.
    Chuyển đổi dữ liệu thô thành 23 features cho model prediction.
    """
    
    def __init__(self, final_features):
        self.final_features = final_features
        self.features_to_remove = [
            'global_cancel_val_ratio'
        ]

    def transform(self, transaction: pd.DataFrame, snapshot_base: pd.DataFrame) -> pd.DataFrame:
        df = transaction.copy()
        df['Order_date'] = pd.to_datetime(df['Order_date'])
        result = []

        for s_date in snapshot_base['snapshot_date'].unique():
            s_date = pd.to_datetime(s_date)
            snap = snapshot_base[snapshot_base['snapshot_date'] == s_date].copy()
            
            # hist: Lấy toàn bộ lịch sử có trong DataFrame truyền vào cho đến ngày snapshot
            hist = df[df['Order_date'] <= s_date].copy()
            if hist.empty:
                continue

            # TIME WINDOWS
            L1, L3, L5 = [s_date - relativedelta(months=i) for i in [1, 3, 5]]

            # --- AGGREGATION FUNCTION ---
            def agg_rfm_features(sub_df, name):
                if sub_df.empty:
                    return pd.DataFrame()
                sub_df = sub_df.copy()
                sub_df['items_per_cat'] = sub_df['Order_n_lines'] / (sub_df['Order_n_categories'] + 0.001)
                
                f = sub_df.groupby('Customer_id').agg(**{
                    f'sum_{name}_value': ('Order_value', 'sum'),
                    f'avg_{name}_value': ('Order_value', 'mean'),
                    f'std_{name}_value': ('Order_value', 'std'),
                    f'cnt_{name}_orders': ('Order_id', 'count'),
                    f'avg_{name}_skus': ('Order_n_lines', 'mean'),
                    f'sum_{name}_items_log': ('Log_items', 'sum'),
                    f'avg_{name}_items_log': ('Log_items', 'mean'),
                    f'sum_{name}_canceled': ('Is_canceled', 'sum'),
                    f'avg_n_categories_{name}': ('Order_n_categories', 'mean'),
                    f'sum_n_categories_{name}': ('Order_n_categories', 'sum'),
                    f'avg_items_per_cat_{name}': ('items_per_cat', 'mean')
                }).reset_index()
                f[f'category_diversity_{name}'] = f[f'sum_n_categories_{name}'] / (f[f'cnt_{name}_orders'] + 0.001)
                f[f'cancel_rate_{name}'] = f[f'sum_{name}_canceled'] / (f[f'cnt_{name}_orders'] + 0.001)
                return f

            hist['items_per_cat'] = hist['Order_n_lines'] / (hist['Order_n_categories'] + 0.001)

            f1 = agg_rfm_features(hist[hist['Order_date'] > L1], 'L1M')
            f3 = agg_rfm_features(hist[hist['Order_date'] > L3], 'L3M')
            f5 = agg_rfm_features(hist[hist['Order_date'] > L5], 'L5M')

            # --- RHYTHM & ACTIVITY ---
            past_5m = hist[hist['Order_date'] > L5].copy()
            rhythm, activity = pd.DataFrame(), pd.DataFrame()
            if not past_5m.empty:
                past_5m = past_5m.sort_values(['Customer_id', 'Order_date'])
                past_5m['gap'] = past_5m.groupby('Customer_id')['Order_date'].diff().dt.days
                rhythm = past_5m.groupby('Customer_id')['gap'].agg(avg_gap_L5M='mean').reset_index()
                activity = past_5m.groupby('Customer_id').agg(
                    active_months_L5M=('Order_date', lambda x: x.dt.to_period('M').nunique())
                ).reset_index()

            # --- RECENCY, TENURE & RISK ---
            history_summary = hist.sort_values('Order_date').groupby('Customer_id').agg(
                min_date=('Order_date', 'min'),
                max_date=('Order_date', 'max'),
                last_order_value=('Order_value', 'last'),
                last_order_canceled=('Is_canceled', 'last'),
                global_aov=('Order_value', 'mean'),
                total_canceled_val=('Canceled_value', 'sum'),
                total_gross_val=('Order_value', lambda x: x[x > 0].sum())
            ).reset_index()
            
            history_summary['tenure_days'] = (s_date - history_summary['min_date']).dt.days
            history_summary['recency_days'] = (s_date - history_summary['max_date']).dt.days
            history_summary['global_cancel_val_ratio'] = history_summary['total_canceled_val'] / (history_summary['total_gross_val'] + 0.001)
            history_summary['last_order_intensity'] = history_summary['last_order_value'] / (history_summary['global_aov'] + 0.001)

            # MERGE DATA
            feat = snap[['Customer_id', 'snapshot_date']].copy()
            dfs_to_merge = [f1, f3, f5, rhythm, activity, 
                            history_summary[['Customer_id', 'tenure_days', 'recency_days', 
                                              'last_order_intensity', 'last_order_canceled', 
                                              'global_cancel_val_ratio']]]
            
            for df_merge in dfs_to_merge:
                if not df_merge.empty:
                    feat = feat.merge(df_merge, on='Customer_id', how='left')
            
            result.append(feat)

        modeling_df = pd.concat(result, ignore_index=True)
        num_cols = modeling_df.select_dtypes(include=[np.number]).columns
        modeling_df[num_cols] = modeling_df[num_cols].fillna(0)

        # Đảm bảo các cột cần thiết tồn tại trước khi tính biến tương tác
        required_base = ['cnt_L5M_orders', 'tenure_days', 'global_cancel_val_ratio',
                        'sum_L1M_value', 'sum_L3M_value']
        for col in required_base:
            if col not in modeling_df.columns:
                modeling_df[col] = 0

        # BIẾN TƯƠNG TÁC (chỉ tính 3 biến tương tác trong FINAL_FEATURES_23)
        modeling_df['order_velocity'] = modeling_df['cnt_L5M_orders'] / 5
        modeling_df['success_order_rate'] = 1 - modeling_df['global_cancel_val_ratio']
        modeling_df['spend_velocity'] = np.where(
            modeling_df['sum_L3M_value'] > 0,
            modeling_df['sum_L1M_value'] / (modeling_df['sum_L3M_value'] / 3),
            0
        )

        # LOẠI BỎ BIẾN TRUNG GIAN (theo feature selection)
        modeling_df = modeling_df.drop(columns=[f for f in self.features_to_remove if f in modeling_df.columns])
        
        # Fill các cột còn thiếu = 0 và trả về đúng thứ tự
        for f in self.final_features:
            if f not in modeling_df.columns:
                modeling_df[f] = 0
        
        return modeling_df[self.final_features].reset_index(drop=True)

    def transform_api_input(self, transactions: List[Dict], customer_id: str, snapshot_date: str) -> pd.DataFrame:
        """
        Transform API input thành features.
        
        Args:
            transactions: List các dict chứa thông tin giao dịch từ API
            customer_id: ID khách hàng
            snapshot_date: Ngày snapshot (ISO format)
        
        Returns:
            DataFrame với 23 features
        """
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
        
        # Transform transactions to DataFrame format
        df_transactions = pd.DataFrame(transactions)
        
        # Rename columns to match preprocessor expected format
        df_transactions = df_transactions.rename(columns=column_mapping)
        
        # Add Customer_id
        df_transactions['Customer_id'] = customer_id
        
        # Create snapshot_base
        snapshot_base = pd.DataFrame({
            'Customer_id': [customer_id],
            'snapshot_date': [pd.to_datetime(snapshot_date)]
        })
        
        return self.transform(df_transactions, snapshot_base)


# ==========================================
# 🛠️ HELPER FUNCTIONS
# ==========================================
def get_top_reasons(feature_importance_df: pd.DataFrame, feature_values: pd.Series, n: int = 3) -> List[Dict]:
    """
    Lấy top n lý do ảnh hưởng đến kết quả dự đoán.
    
    Args:
        feature_importance_df: DataFrame chứa feature importance với cột 'Feature' và 'Importance'
        feature_values: Series chứa giá trị các feature
        n: Số lý do cần lấy
    
    Returns:
        List các dict chứa feature, importance_percent, value
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
    
    reasons = []
    for _, row in sorted_fi.head(n).iterrows():
        feature_name = row[feature_col]
        importance = row[importance_col]
        
        # Get feature value
        if feature_name in feature_values.index:
            value = feature_values[feature_name]
        else:
            value = 0.0
        
        reasons.append({
            'feature': feature_name,
            'importance_percent': round((importance / total_importance) * 100, 2) if total_importance > 0 else 0.0,
            'value': round(float(value), 4)
        })
    
    return reasons


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