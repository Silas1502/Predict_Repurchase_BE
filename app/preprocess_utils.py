"""
==========================================
PREPROCESSOR MODULE - OlistPreprocessor Class
==========================================
Mô tả: Module này chứa class OlistPreprocessor để biến đổi dữ liệu thô thành 
25-30 features được chọn lọc cho model dự báo.

Vị trí: backend/app/preprocess_utils.py
==========================================
"""

import pandas as pd
import numpy as np
import joblib
import os
from typing import List, Dict, Union


class OlistPreprocessor:
    """
    Class tiền xử lý dữ liệu giao dịch Olist.
    
    Chức năng:
    1. Nhận dữ liệu giao dịch thô (transaction level)
    2. Tính toán các đặc trưng RFM, Velocity, Geography
    3. Trả về DataFrame với đúng 25 features cần thiết cho model
    
    Attributes:
        final_features (list): Danh sách các features cần giữ lại
        top_states (list): Danh sách các bang quan trọng nhất
    """
    
    def __init__(self, final_features: List[str] = None):
        # 1. Liệt kê 25 cột bạn đã chọn từ Feature Selection
        self.model_features = [
            'order_acceleration', 'tenure_days', 'avg_L6M_review', 'avg_gap_L6M', 
            'cnt_L6M_valid', 'cnt_L3M_total', 'is_state_SP', 'active_months_L6M', 
            'is_state_PR', 'is_state_RS', 'avg_L1M_items', 'cnt_L6M_total', 
            'is_state_SC', 'avg_L6M_items', 'is_state_BA', 'flag_high_churn_risk', 
            'recency_days', 'is_state_MG', 'unique_cat_L6M', 'avg_L3M_items', 
            'avg_L3M_review', 'is_state_RJ', 'spend_velocity', 'flag_vip', 'is_state_Others'
        ]
        
        # 2. Các cột bổ trợ cần thiết để tính toán (tránh bị xóa nhầm)
        self.helper_features = ['sum_L1M_value', 'sum_L3M_value', 'cnt_L1M_total']
        
        # 3. Gộp lại làm danh sách mặc định
        self.default_features = list(set(self.model_features + self.helper_features))
        
        self.final_features = final_features if final_features else self.model_features
        self.top_states = ['SP', 'RJ', 'MG', 'RS', 'PR', 'SC', 'BA']
        
    def transform(self, transaction: pd.DataFrame, snapshot_base: pd.DataFrame) -> pd.DataFrame:
        """
        Biến đổi dữ liệu giao dịch thô thành features.
        """
        # Copy để tránh modify data gốc
        transaction = transaction.copy()
        transaction['order_purchase_timestamp'] = pd.to_datetime(transaction['order_purchase_timestamp'])
        
        result = []
        
        # Xử lý từng snapshot_date
        for s_date in snapshot_base['snapshot_date'].unique():
            s_date = pd.to_datetime(s_date)
            
            # Lọc snapshot hiện tại
            snap = snapshot_base[snapshot_base['snapshot_date'] == s_date].copy()
            
            # Lọc lịch sử giao dịch <= snapshot_date
            hist = transaction[transaction['order_purchase_timestamp'] <= s_date].copy()
            
            if hist.empty:
                # Nếu không có lịch sử, tạo row với giá trị 0
                empty_row = {col: 0 for col in self.default_features}
                for _, row in snap.iterrows():
                    empty_row['customer_unique_id'] = row['customer_unique_id']
                    result.append(empty_row.copy())
                continue
            
            # ==========================================
            # 1. GEOGRAPHY - Xác định bang của khách hàng
            # ==========================================
            latest_geo = hist.sort_values('order_purchase_timestamp') \
                .groupby('customer_unique_id')['customer_state'] \
                .last() \
                .reset_index(name='main_state')
            
            # ==========================================
            # 2. AGGREGATION RFM - Tính toán các chỉ số RFM
            # ==========================================
            def get_agg(df: pd.DataFrame, name: str) -> tuple:
                if df.empty:
                    return pd.DataFrame(), pd.DataFrame()
                
                orders = df.groupby(['customer_unique_id', 'order_id']).agg(
                    order_value=('valid_spend_capped', 'sum'),
                    items=('product_id', 'count'),
                    review=('review_score', 'mean'),
                    is_total=('is_total_order', 'max'),
                    is_valid=('is_valid_order', 'max'),
                    order_time=('order_purchase_timestamp', 'max')
                ).reset_index()
                
                agg_dict = {
                    f'sum_{name}_value': ('order_value', 'sum'),
                    f'avg_{name}_value': ('order_value', 'mean'),
                    f'std_{name}_value': ('order_value', 'std'),
                    f'cnt_{name}_total': ('is_total', 'sum'),
                    f'cnt_{name}_valid': ('is_valid', 'sum'),
                    f'avg_{name}_items': ('items', 'mean'),
                    f'avg_{name}_review': ('review', 'mean')
                }
                
                f = orders.groupby('customer_unique_id').agg(**agg_dict).reset_index()
                return f, orders
            
            L1 = s_date - pd.DateOffset(months=1)
            L3 = s_date - pd.DateOffset(months=3)
            L6 = s_date - pd.DateOffset(months=6)
            
            f1, _ = get_agg(hist[hist['order_purchase_timestamp'] > L1], 'L1M')
            f3, _ = get_agg(hist[hist['order_purchase_timestamp'] > L3], 'L3M')
            f6, o6 = get_agg(hist[hist['order_purchase_timestamp'] > L6], 'L6M')
            
            # ==========================================
            # 3. RHYTHM, RECENCY, TENURE
            # ==========================================
            h_dates = hist.groupby('customer_unique_id')['order_purchase_timestamp'].agg(
                min_d='min',
                max_d='max'
            ).reset_index()
            
            h_dates['tenure_days'] = (s_date - h_dates['min_d']).dt.days
            h_dates['recency_days'] = (s_date - h_dates['max_d']).dt.days
            
            if not o6.empty:
                o6['order_month'] = o6['order_time'].dt.to_period('M')
                active_months = o6.groupby('customer_unique_id')['order_month'] \
                    .nunique() \
                    .reset_index(name='active_months_L6M')
                
                order_times = o6.sort_values(['customer_unique_id', 'order_time']) \
                    .groupby('customer_unique_id')['order_time'] \
                    .apply(lambda x: x.diff().dt.days.mean()) \
                    .reset_index(name='avg_gap_L6M')
                order_times['avg_gap_L6M'] = order_times['avg_gap_L6M'].fillna(0)
            else:
                active_months = pd.DataFrame(columns=['customer_unique_id', 'active_months_L6M'])
                order_times = pd.DataFrame(columns=['customer_unique_id', 'avg_gap_L6M'])
            
            if not hist.empty and 'category_en' in hist.columns:
                cat_L6 = hist[hist['order_purchase_timestamp'] > L6] \
                    .groupby('customer_unique_id')['category_en'] \
                    .nunique() \
                    .reset_index(name='unique_cat_L6M')
            else:
                cat_L6 = pd.DataFrame(columns=['customer_unique_id', 'unique_cat_L6M'])
            
            # ==========================================
            # 4. MERGE ALL FEATURES
            # ==========================================
            feat = snap[['customer_unique_id', 'snapshot_date']].copy()
            merge_list = [(f1, 'L1M'), (f3, 'L3M'), (f6, 'L6M'), (h_dates, 'dates'), 
                          (active_months, 'active'), (order_times, 'gap'), (cat_L6, 'cat'), (latest_geo, 'geo')]
            
            for df_merge, _ in merge_list:
                if not df_merge.empty:
                    feat = feat.merge(df_merge, on='customer_unique_id', how='left')
            
            feat = feat.fillna(0)

            # --- LƯỚI AN TOÀN (MỚI THÊM) ---
            # Đảm bảo các cột cần để tính Velocity ở bước tiếp theo phải tồn tại
            for col in ['sum_L1M_value', 'sum_L3M_value', 'cnt_L1M_total', 'cnt_L3M_total']:
                if col not in feat.columns:
                    feat[col] = 0
            
            # ==========================================
            # 5. VELOCITY & ACCELERATION
            # ==========================================
            feat['spend_velocity'] = feat['sum_L1M_value'] / (feat['sum_L3M_value'] / 3 + 1)
            feat['order_acceleration'] = feat['cnt_L1M_total'] / (feat['cnt_L3M_total'] / 3 + 1)
            
            # ==========================================
            # 6. GEOGRAPHY ONE-HOT ENCODING
            # ==========================================
            feat['main_state'] = feat['main_state'].apply(lambda x: x if x in self.top_states else 'Others')
            for s in self.top_states + ['Others']:
                feat[f'is_state_{s}'] = (feat['main_state'] == s).astype(int)
            
            result.append(feat)
        
        # Concatenate results
        if result:
            final_df = pd.concat(result, ignore_index=True)
        else:
            final_df = pd.DataFrame(columns=self.default_features)
        
        # ==========================================
        # 7. BƯỚC MÀNG LỌC & LOGIC BỔ SUNG
        # ==========================================
        
        # Đảm bảo flag_vip và flag_high_churn_risk được tính toán
        final_df['flag_vip'] = (final_df.get('sum_L6M_value', 0) > 500).astype(int)
        final_df['flag_high_churn_risk'] = (final_df.get('recency_days', 0) > 180).astype(int)

        # Sử dụng reindex để ép buộc ra đúng danh sách features mà Model yêu cầu
        # Nếu thiếu bất kỳ cột nào, nó sẽ tự điền 0
        final_df = final_df.reindex(columns=self.final_features, fill_value=0)
        
        # Đảm bảo kết quả trả về là 2D (DataFrame) và không rỗng
        if final_df.empty:
            # Tạo một dòng toàn số 0 nếu không tìm thấy dữ liệu
            final_df = pd.DataFrame([0] * len(self.final_features), index=self.final_features).T
            
        return final_df[self.final_features].reset_index(drop=True)
    
    def transform_api_input(self, transactions: List[Dict], customer_id: str, snapshot_date: str) -> pd.DataFrame:
        """
        Phiên bản đặc biệt cho API - nhận input từ JSON.
        
        Args:
            transactions: List các dict chứa thông tin giao dịch
            customer_id: ID khách hàng
            snapshot_date: Ngày chốt dữ liệu (format: YYYY-MM-DD)
        
        Returns:
            pd.DataFrame: DataFrame với features đã được tính toán
        """
        # Convert transactions thành DataFrame
        df_transactions = pd.DataFrame(transactions)
        
        # 2. QUAN TRỌNG: Gán ID khách hàng vào bảng này để máy tính nhận diện được
        df_transactions['customer_unique_id'] = customer_id
        
        # Tạo snapshot_base
        snapshot_base = pd.DataFrame({
            'customer_unique_id': [customer_id],
            'snapshot_date': [snapshot_date]
        })
        
        # Gọi hàm transform chính
        return self.transform(df_transactions, snapshot_base)


def get_top_reasons(feature_importance_df: pd.DataFrame, features: pd.Series, n: int = 3) -> List[Dict]:
    """
    Lấy top n lý do ảnh hưởng đến kết quả dự báo.
    
    Args:
        feature_importance_df: DataFrame chứa feature và importance
        features: Series chứa giá trị các features của 1 khách hàng
        n: Số lý do cần trả về (mặc định 3)
    
    Returns:
        List các dict chứa feature, importance, value
    """
    # Sort theo importance giảm dần
    fi_sorted = feature_importance_df.sort_values('Importance_Percent', ascending=False)
    
    reasons = []
    for _, row in fi_sorted.head(n).iterrows():
        feature_name = row['Feature']
        if feature_name in features.index:
            reasons.append({
                'feature': feature_name,
                'importance_percent': round(row['Importance_Percent'], 2),
                'value': round(features[feature_name], 4) if feature_name in features.index else 0
            })
    
    return reasons


def load_preprocessor(model_path: str = './models/preprocessor.pkl') -> OlistPreprocessor:
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Không tìm thấy file {model_path}")
    
    # Thêm dòng này để "đánh lừa" bộ giải mã pickle
    import __main__
    __main__.OlistPreprocessor = OlistPreprocessor
    
    preprocessor = joblib.load(model_path)
    return preprocessor


# ==========================================
# TEST
# ==========================================
if __name__ == "__main__":
    print("Testing OlistPreprocessor...")
    
    # Test data
    test_transactions = [
        {
            'customer_unique_id': 'test_customer_001',
            'order_id': 'order_001',
            'order_purchase_timestamp': '2018-08-01 10:00:00',
            'product_id': 'prod_001',
            'category_en': 'electronics',
            'valid_spend_capped': 150.50,
            'review_score': 5.0,
            'customer_state': 'SP',
            'is_total_order': 1,
            'is_valid_order': 1
        },
        {
            'customer_unique_id': 'test_customer_001',
            'order_id': 'order_002',
            'order_purchase_timestamp': '2018-08-15 14:30:00',
            'product_id': 'prod_002',
            'category_en': 'furniture',
            'valid_spend_capped': 299.99,
            'review_score': 4.0,
            'customer_state': 'SP',
            'is_total_order': 1,
            'is_valid_order': 1
        }
    ]
    
    # Khởi tạo preprocessor
    preprocessor = OlistPreprocessor()
    
    # Transform
    result = preprocessor.transform_api_input(
        transactions=test_transactions,
        customer_id='test_customer_001',
        snapshot_date='2018-09-01'
    )
    
    print(f"✅ Transform thành công!")
    print(f"   - Số features: {len(result.columns)}")
    print(f"   - Các features: {list(result.columns)}")
    print(f"   - Shape: {result.shape}")
    print(f"\nSample output:\n{result.head()}")
