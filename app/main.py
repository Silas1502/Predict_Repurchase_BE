"""
==========================================
MAIN API - FastAPI Application
==========================================
FastAPI application với các endpoints:
- POST /predict: Dự báo khả năng mua lại
- GET /customers/{id}/history: Lấy lịch sử giao dịch
- GET /applications: Lấy lịch sử dự báo
- GET /health: Health check
- GET /model-info: Thông tin model (Bonus)
==========================================
"""

import os
import joblib
import pandas as pd
import numpy as np
from typing import List, Dict, Any, Optional
from datetime import datetime
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Query, Depends, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

# Import local modules
from app.schemas import (
    PredictRequest, PredictResponse, TopReason,
    CustomerHistoryResponse, ApplicationsListResponse, ApplicationLog,
    HealthCheckResponse, ModelInfoResponse, ErrorResponse, RepurchaseLogCreate
)
from app.preprocess_utils import OlistPreprocessor, get_top_reasons, load_preprocessor
from app.database import db_manager, init_database, check_db_health, get_db


# ==========================================
# GLOBAL VARIABLES - MODEL & PREPROCESSOR
# ==========================================
from supabase import create_client, Client
from dotenv import load_dotenv

load_dotenv()

# Lấy thông tin từ file .env
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")

# Khởi tạo Supabase Client
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

model = None
preprocessor = None
feature_importance_df = None
threshold = 0.0806  # Ngưỡng mặc định từ training
MODEL_VERSION = "1.0.0"
MODEL_TYPE = "LightGBM"


def load_models():
    """
    Load model, preprocessor và feature importance từ files
    """
    global model, preprocessor, feature_importance_df, threshold
    
    models_dir = "./models"
    
    try:
        # Load model
        model_path = os.path.join(models_dir, "best_model.pkl")
        if os.path.exists(model_path):
            model = joblib.load(model_path)
            print(f"✅ Đã load model từ {model_path}")
        else:
            print(f"⚠️ Không tìm thấy model tại {model_path}")
        
        # Load preprocessor
        preprocessor_path = os.path.join(models_dir, "preprocessor.pkl")
        if os.path.exists(preprocessor_path):
            preprocessor = load_preprocessor(preprocessor_path)
            print(f"✅ Đã load preprocessor từ {preprocessor_path}")
        else:
            # Fallback: tạo preprocessor mới
            preprocessor = OlistPreprocessor()
            print(f"⚠️ Sử dụng preprocessor mặc định")
        
        # Load feature importance
        fi_path = os.path.join(models_dir, "feature_importance.csv")
        if os.path.exists(fi_path):
            feature_importance_df = pd.read_csv(fi_path)
            print(f"✅ Đã load feature importance từ {fi_path}")
        else:
            print(f"⚠️ Không tìm thấy feature importance")
        
        # Load threshold
        threshold_path = os.path.join(models_dir, "optimal_threshold.pkl")
        if os.path.exists(threshold_path):
            threshold = joblib.load(threshold_path)
            print(f"✅ Đã load threshold: {threshold}")
        else:
            print(f"⚠️ Sử dụng threshold mặc định: {threshold}")
            
    except Exception as e:
        print(f"❌ Lỗi load models: {str(e)}")
        raise


def get_potential_level(probability: float) -> str:
    """
    Phân loại mức độ tiềm năng dựa trên probability
    
    Theo đề bài:
    - High: > 15% (Khách hàng cực kỳ tiềm năng)
    - Medium: 8% - 15% (Khách hàng cần chăm sóc thêm)
    - Low: < 8% (Khách hàng ít khả năng quay lại)
    
    Args:
        probability: Xác suất mua lại (0-1)
    
    Returns:
        str: "High", "Medium", hoặc "Low"
    """
    if probability > 0.15:
        return "High"
    elif probability >= 0.08:
        return "Medium"
    else:
        return "Low"


# ==========================================
# LIFESPAN - Startup & Shutdown
# ==========================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Lifespan context manager cho startup và shutdown
    """
    # Startup
    print("🚀 Starting up...")
    
    # Load models
    load_models()
    
    # Connect to database
    init_database()
    
    print("✅ Startup complete!")
    
    yield
    
    # Shutdown
    print("🛑 Shutting down...")


# ==========================================
# APP INITIALIZATION
# ==========================================

app = FastAPI(
    title="Olist Repurchase Prediction API",
    description="API dự báo khả năng mua lại của khách hàng Olist",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan
)

# CORS Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://127.0.0.1:3000",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ==========================================
# API ENDPOINTS
# ==========================================

@app.get("/", tags=["Root"])
async def root():
    """Root endpoint - API information"""
    return {
        "message": "Olist Repurchase Prediction API",
        "version": "1.0.0",
        "docs_url": "/docs",
        "endpoints": {
            "predict": "POST /predict",
            "health": "GET /health",
            "history": "GET /customers/{id}/history",
            "applications": "GET /applications"
        }
    }


@app.get("/health", response_model=HealthCheckResponse, tags=["Health"])
async def health_check():
    """
    Health check endpoint - Kiểm tra trạng thái hệ thống
    
    Returns:
        Trạng thái model, preprocessor, và database
    """
    db_connected = check_db_health()
    
    return HealthCheckResponse(
        status="healthy" if all([model is not None, preprocessor is not None, db_connected]) else "degraded",
        model_loaded=model is not None,
        preprocessor_loaded=preprocessor is not None,
        database_connected=db_connected,
        model_version=MODEL_VERSION,
        threshold=threshold
    )


@app.post("/predict", response_model=PredictResponse, tags=["Prediction"])
async def predict(request: PredictRequest):
    try:
        # 1. Kiểm tra Model & Preprocessor
        if model is None or preprocessor is None:
            raise HTTPException(status_code=503, detail="Model hoặc Preprocessor chưa sẵn sàng")
        
        # 2. Trích xuất thông tin từ Request
        # Lưu ý: dùng .get() hoặc truy cập trực tiếp tùy theo cách bạn định nghĩa Pydantic model
        customer_id = request.customer_info.get("customer_unique_id")
        snapshot_date = request.customer_info.get("snapshot_date")
        transactions_list = [t.dict() for t in request.transactions]
        
        # 3. Tiền xử lý dữ liệu
        features_df = preprocessor.transform_api_input(
            transactions=transactions_list,
            customer_id=customer_id,
            snapshot_date=snapshot_date
        )
        
        # 4. Thực hiện dự báo
        probability = float(model.predict_proba(features_df)[0][1])
        is_repurchase = bool(probability >= threshold)
        potential_level = get_potential_level(probability)
        
        # 5. Lấy danh sách lý do ảnh hưởng (Top Reasons)
        top_reasons_data = []
        if feature_importance_df is not None:
            reasons = get_top_reasons(feature_importance_df, features_df.iloc[0], n=3)
            top_reasons_data = reasons # List các dict
        
        # 6. LƯU VÀO SUPABASE (Khớp với Schema của bạn)
        prediction_id = None
        try:
            log_entry = {
                "customer_id": str(customer_id), # Đảm bảo là string
                "input_data": {
                    "transactions": transactions_list, 
                    "snapshot_date": str(snapshot_date)
                },
                "probability": float(probability),  # ÉP KIỂU: float64 -> float
                "is_repurchase": bool(is_repurchase), # ÉP KIỂU: bool_ -> bool
                "potential_level": str(potential_level),
                "top_reasons": [
                    {
                        "feature": str(r["feature"]),
                        "importance_percent": float(r["importance_percent"]),
                        "value": float(r["value"])
                    } for r in top_reasons_data
                ]
            }
            
            # Thực hiện Insert
            # Đảm bảo bạn đã khởi tạo: supabase = create_client(url, key)
            db_response = supabase.table("repurchase_logs").insert(log_entry).execute()
            
            if db_response.data:
                prediction_id = db_response.data[0].get("id")
                print(f"✅ Đã lưu vào log thành công! ID: {prediction_id}")
                
        except Exception as db_err:
            print(f"⚠️ Cảnh báo: Không lưu được log vào Supabase. Lỗi: {str(db_err)}")
        
        # 7. Trả về kết quả cho Client
        return PredictResponse(
            success=True,
            customer_id=customer_id,
            snapshot_date=snapshot_date,
            probability=round(probability, 4),
            probability_percent=round(probability * 100, 2),
            is_repurchase=is_repurchase,
            potential_level=potential_level,
            threshold_used=threshold,
            top_reasons=[TopReason(**r) for r in top_reasons_data],
            prediction_id=str(prediction_id) if prediction_id else None,
            created_at=datetime.now().isoformat()
        )
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Lỗi hệ thống: {str(e)}")


@app.get("/customers/{customer_id}/history", response_model=CustomerHistoryResponse, tags=["Customer"])
async def get_customer_history(customer_id: str):
    """
    Lấy lịch sử giao dịch của một khách hàng từ database
    
    Dùng cho tính năng Quick-fill: Khi người dùng nhập customer_id,
    hệ thống tự động lấy dữ liệu và đổ vào bảng nhập liệu.
    
    **Path Parameter:**
    - customer_id: customer_unique_id của khách hàng
    
    **Response:**
    - Danh sách các giao dịch (order_items) của khách hàng
    """
    try:
        transactions = db_manager.get_customer_transactions(customer_id)
        
        if not transactions:
            return CustomerHistoryResponse(
                success=True,
                customer_id=customer_id,
                count=0,
                transactions=[]
            )
        
        # Chuyển đổi timestamp sang string format
        for t in transactions:
            if 'order_purchase_timestamp' in t:
                t['order_purchase_timestamp'] = str(t['order_purchase_timestamp'])
            if 'created_at' in t:
                t['created_at'] = str(t['created_at'])
            if 'id' in t:
                del t['id']  # Xóa internal id
        
        return CustomerHistoryResponse(
            success=True,
            customer_id=customer_id,
            count=len(transactions),
            transactions=transactions
        )
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Lỗi lấy lịch sử: {str(e)}"
        )


@app.get("/applications", response_model=ApplicationsListResponse, tags=["History"])
async def get_applications(
    page: int = Query(1, ge=1, description="Số trang"),
    page_size: int = Query(20, ge=1, le=100, description="Số records mỗi trang")
):
    """
    Lấy lịch sử các lượt dự báo với pagination
    
    **Query Parameters:**
    - page: Số trang (mặc định 1)
    - page_size: Số records mỗi trang (mặc định 20, max 100)
    
    **Response:**
    - Danh sách các lượt dự báo đã thực hiện
    """
    try:
        result = db_manager.get_predictions_history(page=page, page_size=page_size)
        
        # Chuyển đổi data sang schema
        applications = []
        for item in result.get("data", []):
            applications.append(ApplicationLog(
                id=item.get("id"),
                customer_id=item.get("customer_id"),
                probability=item.get("probability"),
                is_repurchase=item.get("is_repurchase"),
                potential_level=item.get("potential_level"),
                created_at=str(item.get("created_at"))
            ))
        
        return ApplicationsListResponse(
            success=True,
            count=result.get("count", 0),
            page=page,
            page_size=page_size,
            data=applications
        )
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Lỗi lấy lịch sử: {str(e)}"
        )


@app.get("/applications/{prediction_id}", tags=["History"])
async def get_application_detail(prediction_id: str):
    """
    Lấy chi tiết một lượt dự báo (Nâng cao)
    
    **Path Parameter:**
    - prediction_id: UUID của dự báo
    
    **Response:**
    - Chi tiết đầy đủ của một lượt dự báo
    """
    try:
        result = db_manager.get_prediction_by_id(prediction_id)
        
        if not result:
            raise HTTPException(
                status_code=404,
                detail=f"Không tìm thấy dự báo với ID: {prediction_id}"
            )
        
        return {
            "success": True,
            "data": result
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Lỗi lấy chi tiết: {str(e)}"
        )


@app.get("/model-info", response_model=ModelInfoResponse, tags=["Model"])
async def get_model_info():
    """
    Lấy thông tin về model (Bonus)
    
    Returns:
        Thông tin model: loại, phiên bản, ngưỡng threshold, danh sách features
    """
    try:
        feature_list = []
        if preprocessor and hasattr(preprocessor, 'final_features'):
            feature_list = preprocessor.final_features
        
        return ModelInfoResponse(
            model_type=MODEL_TYPE,
            model_version=MODEL_VERSION,
            training_date=None,  # Có thể thêm từ metadata
            threshold=threshold,
            total_features=len(feature_list),
            feature_list=feature_list
        )
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Lỗi lấy thông tin model: {str(e)}"
        )


# ==========================================
# ERROR HANDLERS
# ==========================================

@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    """Xử lý HTTP exceptions"""
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "success": False,
            "error_code": f"HTTP_{exc.status_code}",
            "message": exc.detail
        }
    )


@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    """Xử lý các exceptions chung"""
    return JSONResponse(
        status_code=500,
        content={
            "success": False,
            "error_code": "INTERNAL_ERROR",
            "message": str(exc)
        }
    )


# ==========================================
# MAIN
# ==========================================

if __name__ == "__main__":
    import uvicorn
    
    print("""
    ╔═══════════════════════════════════════════════════════════╗
    ║       Olist Repurchase Prediction API - FastAPI           ║
    ╠═══════════════════════════════════════════════════════════╣
    ║  Endpoints:                                               ║
    ║    • GET  /                    - API info                 ║
    ║    • GET  /health              - Health check            ║
    ║    • POST /predict             - Dự báo                   ║
    ║    • GET  /customers/{id}/history - Lịch sử khách hàng     ║
    ║    • GET  /applications        - Lịch sử dự báo           ║
    ║    • GET  /model-info          - Thông tin model          ║
    ╠═══════════════════════════════════════════════════════════╣
    ║  Swagger UI: http://localhost:8000/docs                    ║
    ╚═══════════════════════════════════════════════════════════╝
    """)
    
    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
