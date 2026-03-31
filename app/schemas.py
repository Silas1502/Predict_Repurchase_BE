"""
==========================================
SCHEMAS MODULE - Pydantic Models
==========================================
Định nghĩa các Pydantic models cho request/response validation
==========================================
"""

from pydantic import BaseModel, Field, validator
from typing import List, Optional, Dict, Any
from datetime import datetime
from decimal import Decimal


# ==========================================
# 1. REQUEST MODELS - Input từ Frontend
# ==========================================

class TransactionItem(BaseModel):
    """Một dòng giao dịch (order) cho Online Retail data"""
    order_id: str = Field(..., description="Mã đơn hàng")
    total_items: int = Field(..., ge=0, description="Tổng số lượng sản phẩm trong đơn")
    log_items: float = Field(..., description="Logarithm của tổng số lượng sản phẩm")
    order_date: str = Field(..., description="Ngày mua (ISO format)")
    order_value: float = Field(..., description="Số tiền thanh toán (bao gồm giá trị âm đối với đơn bị hủy)")
    canceled_value: float = Field(default=0.0, description="Số tiền bị hủy (0 nếu đơn hàng không bị hủy)")
    order_n_categories: int = Field(..., ge=0, description="Số danh mục sản phẩm trong đơn")
    order_n_lines: int = Field(..., ge=0, description="Số dòng sản phẩm trong đơn")
    is_canceled: int = Field(..., ge=0, le=1, description="Cờ hủy đơn hàng (0 hoặc 1)")
    country: str = Field(..., description="Tên quốc gia")


class PredictRequest(BaseModel):
    """Request cho endpoint POST /predict"""
    customer_info: Dict[str, Any] = Field(
        ...,
        description="Thông tin khách hàng",
        example={
            "customer_id": "12345",
            "snapshot_date": "2010-09-30"
        }
    )
    transactions: List[TransactionItem] = Field(
        ...,
        description="Danh sách giao dịch của khách hàng",
        min_items=1
    )
    
    @validator('customer_info')
    def validate_customer_info(cls, v):
        """Validate customer_info có đủ các trường cần thiết"""
        required_fields = ['customer_id', 'snapshot_date']
        for field in required_fields:
            if field not in v:
                raise ValueError(f"customer_info thiếu trường: {field}")
        return v


# ==========================================
# 2. RESPONSE MODELS - Output trả về
# ==========================================

class TopReason(BaseModel):
    """Lý do ảnh hưởng đến kết quả"""
    feature: str = Field(..., description="Tên feature")
    importance_percent: float = Field(..., description="Tầm quan trọng %")
    value: float = Field(..., description="Giá trị của feature")
    impact: Optional[str] = Field(None, description="Tác động: positive (tích cực) hoặc negative (tiêu cực)")


class PredictResponse(BaseModel):
    """Response cho endpoint POST /predict"""
    success: bool = Field(..., description="Trạng thái thành công")
    customer_id: str = Field(..., description="ID khách hàng")
    snapshot_date: str = Field(..., description="Ngày chốt dữ liệu")
    probability: float = Field(
        ..., 
        ge=0, 
        le=1, 
        description="Xác suất mua lại (0-1)"
    )
    probability_percent: float = Field(
        ...,
        ge=0,
        le=100,
        description="Xác suất mua lại %"
    )
    is_repurchase: bool = Field(..., description="Có khả năng mua lại không")
    potential_level: str = Field(
        ...,
        description="Mức độ tiềm năng: Royal Diamond (Elite) / Diamond (High Potential) / Gold (Mainstream) / Silver (Nurturing) / Standard (Discard)"
    )
    threshold_used: float = Field(
        ...,
        description="Ngưỡng threshold sử dụng"
    )
    top_reasons: List[TopReason] = Field(
        ...,
        description="Top 3 lý do ảnh hưởng đến kết quả"
    )
    prediction_id: Optional[str] = Field(
        None,
        description="ID của lượt dự báo (nếu lưu vào DB)"
    )
    created_at: Optional[str] = Field(
        None,
        description="Thời gian tạo"
    )


class CustomerHistoryResponse(BaseModel):
    """Response cho endpoint GET /customers/{id}/history"""
    success: bool = Field(..., description="Trạng thái thành công")
    customer_id: str = Field(..., description="ID khách hàng")
    count: int = Field(..., description="Số lượng giao dịch")
    transactions: List[Dict[str, Any]] = Field(
        ...,
        description="Danh sách giao dịch"
    )


class ApplicationLog(BaseModel):
    """Schema cho một record trong lịch sử"""
    id: str = Field(..., description="ID dự báo")
    customer_id: str = Field(..., description="ID khách hàng")
    probability: float = Field(..., description="Xác suất")
    is_repurchase: bool = Field(..., description="Kết quả dự báo")
    potential_level: str = Field(..., description="Mức độ tiềm năng")
    created_at: str = Field(..., description="Thời gian tạo")


class ApplicationsListResponse(BaseModel):
    """Response cho endpoint GET /applications"""
    success: bool = Field(..., description="Trạng thái thành công")
    count: int = Field(..., description="Tổng số records")
    page: int = Field(..., description="Trang hiện tại")
    page_size: int = Field(..., description="Số records mỗi trang")
    data: List[ApplicationLog] = Field(..., description="Danh sách dự báo")


class HealthCheckResponse(BaseModel):
    """Response cho endpoint GET /health"""
    status: str = Field(..., description="Trạng thái tổng quan")
    model_loaded: bool = Field(..., description="Model đã load thành công")
    preprocessor_loaded: bool = Field(..., description="Preprocessor đã load thành công")
    database_connected: bool = Field(..., description="Database đã kết nối")
    model_version: Optional[str] = Field(None, description="Phiên bản model")
    threshold: Optional[float] = Field(None, description="Ngưỡng threshold")


class ModelInfoResponse(BaseModel):
    """Response cho endpoint GET /model-info (Bonus)"""
    model_type: str = Field(..., description="Loại model")
    model_version: str = Field(..., description="Phiên bản")
    training_date: Optional[str] = Field(None, description="Ngày huấn luyện")
    threshold: float = Field(..., description="Ngưỡng tối ưu")
    total_features: int = Field(..., description="Tổng số features")
    feature_list: List[str] = Field(..., description="Danh sách features")


class ErrorResponse(BaseModel):
    """Response khi có lỗi"""
    success: bool = Field(default=False, description="Trạng thái thất bại")
    error_code: str = Field(..., description="Mã lỗi")
    message: str = Field(..., description="Thông báo lỗi")
    details: Optional[Dict[str, Any]] = Field(None, description="Chi tiết lỗi")


# ==========================================
# 3. DATABASE MODELS - Lưu trữ
# ==========================================

class RepurchaseLogCreate(BaseModel):
    """Schema để tạo record trong repurchase_logs"""
    customer_id: str
    input_data: Dict[str, Any]  # Chứa mảng transactions
    probability: float
    is_repurchase: bool
    potential_level: str
    top_reasons: Optional[List[Dict]] = None
