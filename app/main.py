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



from app.preprocess_utils import OnlineRetailPreprocessor, get_top_reasons, get_shap_reasons, load_preprocessor



from app.database import db_manager, init_database, check_db_health, get_db











# ==========================================



# GLOBAL VARIABLES - MODEL & PREPROCESSOR



# ==========================================



from supabase import create_client, Client



from dotenv import load_dotenv







load_dotenv(dotenv_path=os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../.env"))







# Lấy thông tin từ file .env



SUPABASE_URL = os.getenv("SUPABASE_URL")



SUPABASE_KEY = os.getenv("SUPABASE_KEY")







# Khởi tạo Supabase Client



supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)







model = None



preprocessor = None



feature_importance_df = None



threshold = None  # Sẽ được load từ optimal_threshold.pkl



MODEL_VERSION = "1.0.0"



MODEL_TYPE = "XGBoost"







def load_models():



    """



    Load model, preprocessor và feature importance từ files.



    Tự động xử lý đường dẫn để tương thích cả Local và Render.



    """



    global model, preprocessor, feature_importance_df, threshold







    # 1. Xác định vị trí file code đang chạy



    current_dir = os.path.dirname(os.path.abspath(__file__))



    # 2. Xây dựng đường dẫn đến thư mục 'models' nằm cùng cấp với file này

    models_dir = os.path.join(current_dir, "models")



    



    # Dự phòng (Fallback): Nếu không thấy, lùi lại 1 cấp (Trường hợp file nằm trong backend/app)



    if not os.path.exists(models_dir):



        models_dir = os.path.join(current_dir, "..", "models")



    



    print(f"🔍 Đang tìm kiếm tài nguyên tại: {models_dir}")



    



    try:



        # --- Load Model ---



        model_path = os.path.join(models_dir, "best_model.pkl")



        if os.path.exists(model_path):



            model = joblib.load(model_path)



            print(f"✅ Đã load model từ {model_path}")



            print(f"   Model type: {type(model).__name__}")



            print(f"   Model class: {model.__class__.__name__}")



            if hasattr(model, 'n_estimators'):



                print(f"   N estimators: {model.n_estimators}")



            if hasattr(model, 'get_params'):



                params = model.get_params()



                print(f"   Learning rate: {params.get('learning_rate', 'N/A')}")



                print(f"   Max depth: {params.get('max_depth', 'N/A')}")



                print(f"   Subsample: {params.get('subsample', 'N/A')}")



        else:



            print(f"⚠️ Không tìm thấy model tại {model_path}")



        



        # --- Load Preprocessor ---



        preprocessor_path = os.path.join(models_dir, "preprocessor.pkl")



        if os.path.exists(preprocessor_path):



            # Giả sử load_preprocessor là hàm helper bạn đã định nghĩa



            preprocessor = load_preprocessor(preprocessor_path)



            print(f"✅ Đã load preprocessor từ {preprocessor_path}")



        else:



            # Fallback: Tạo preprocessor mới nếu không tìm thấy file



            from app.preprocess_utils import OnlineRetailPreprocessor



            preprocessor = OnlineRetailPreprocessor()



            print(f"⚠️ Sử dụng preprocessor mặc định (Khởi tạo mới)")



        



        # --- Load Feature Importance ---



        fi_path = os.path.join(models_dir, "feature_importance.csv")



        if os.path.exists(fi_path):



            feature_importance_df = pd.read_csv(fi_path)



            print(f"✅ Đã load feature importance từ {fi_path}")



        else:



            print(f"⚠️ Không tìm thấy feature importance")



        



        # --- Load Threshold (Xác suất tối ưu) ---



        threshold_path = os.path.join(models_dir, "optimal_threshold.pkl")



        if os.path.exists(threshold_path):



            threshold = joblib.load(threshold_path)



            print(f"✅ Đã load threshold thành công: {threshold}")



        else:

            raise FileNotFoundError(f"Không tìm thấy file threshold: {threshold_path}. Vui lòng đảm bảo file optimal_threshold.pkl tồn tại.")



            



    except Exception as e:



        print(f"❌ Lỗi nghiêm trọng khi load tài nguyên: {str(e)}")



        raise











def get_potential_level(probability: float) -> str:

    """

    Phân loại mức độ tiềm năng dựa trên xác suất tái mua.

    

    Chiến lược 3 nhóm:

    - Nhóm Khách hàng Tự hành: Tự động quay lại mua hàng, xác suất cao

    - Nhóm Trọng tâm Tăng trưởng: Cần đẩy marketing để chuyển đổi

    - Nhóm Tối ưu Hóa Chi phí: Xác suất thấp, cần chiến lược tiết kiệm ngân sách

    """

    if probability >= 0.60:

        # Nhóm Tự hành: Khách tự quay lại, xác suất cao

        # Hành động: Tri ân nhẹ, giữ chân bằng loyalty program

        return "Nhóm Khách hàng Tự hành"

    

    elif probability >= 0.40:

        # Nhóm Trọng tâm Tăng trưởng: Cần đẩy marketing

        # Hành động: Voucher, ưu đãi đặc biệt, remarketing

        return "Nhóm Trọng tâm Tăng trưởng"

    

    else:

        # Nhóm Tối ưu Hóa Chi phí: Xác suất thấp

        # Hành động: Chiến lược low-cost, email thay vì SMS, không ưu đãi lớn

        return "Nhóm Tối ưu Hóa Chi phí"











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



    title="Online Retail Repurchase Prediction API",



    description="API dự báo khả năng mua lại của khách hàng Online Retail",



    version="1.0.0",



    docs_url="/docs",



    redoc_url="/redoc",



    lifespan=lifespan



)







# CORS Middleware



app.add_middleware(



    CORSMiddleware,



    allow_origins=["*"],



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



        "message": "Online Retail Repurchase Prediction API",



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



        customer_id = request.customer_info.get("customer_id")



        snapshot_date = request.customer_info.get("snapshot_date")



        transactions_list = [t.dict() for t in request.transactions]



        



        # 3. Tiền xử lý dữ liệu



        features_df = preprocessor.transform_api_input(



            transactions=transactions_list,



            customer_id=customer_id,



            snapshot_date=snapshot_date



        )



        



        # DEBUG: Log features để kiểm tra



        print(f"🔍 Features shape: {features_df.shape}")



        print(f"🔍 Features columns: {list(features_df.columns)}")



        print(f"🔍 Features values:\n{features_df.iloc[0].to_dict()}")



        



        # 4. Thực hiện dự báo (XGBoost Booster uses predict, not predict_proba)
        import xgboost as xgb
        dtest = xgb.DMatrix(features_df)
        probability = float(model.predict(dtest)[0])

        print(f"🎯 Raw probability from model: {probability:.4f}")



        is_repurchase = bool(probability >= threshold)

        potential_level = get_potential_level(probability)

        

        # 5. Lấy danh sách lý do ảnh hưởng (Top Reasons) - Dùng SHAP cho từng prediction

        top_reasons_data = []

        try:

            # Thử tính SHAP values cho từng khách hàng

            shap_reasons = get_shap_reasons(model, features_df, list(features_df.columns), n=3)

            if shap_reasons:

                top_reasons_data = shap_reasons

                print(f" Dùng SHAP values cho top reasons")

            else:

                # Fallback về global feature importance nếu SHAP fail

                if feature_importance_df is not None:

                    reasons = get_top_reasons(feature_importance_df, features_df.iloc[0], n=3)

                    top_reasons_data = reasons

                    print(f" Fallback về global feature importance")

        except Exception as shap_err:

            print(f" SHAP error: {shap_err}")

            # Fallback về global feature importance

            if feature_importance_df is not None:

                reasons = get_top_reasons(feature_importance_df, features_df.iloc[0], n=3)

                top_reasons_data = reasons

        

        # 6. LƯU VÀO SUPABASE (Khớp với Schema của bạn)

        prediction_id = None



        try:



            # Kiểm tra Supabase Key có hợp lệ không (phải bắt đầu bằng 'eyJ' - JWT token)



            if not SUPABASE_KEY or not SUPABASE_KEY.startswith('eyJ'):



                print(f"⚠️ CẢNH BÁO: SUPABASE_KEY không hợp lệ! Key hiện tại: {SUPABASE_KEY[:20] if SUPABASE_KEY else 'None'}...")



                print(f"   → Key phải bắt đầu bằng 'eyJ' (JWT token)")



                print(f"   → Vào Supabase Dashboard → Project Settings → API → copy 'anon public' key")



            



            log_entry = {



                "customer_id": str(customer_id),



                "input_data": {



                    "transactions": transactions_list, 



                    "snapshot_date": str(snapshot_date)



                },



                "probability": float(probability),



                "is_repurchase": bool(is_repurchase),



                "potential_level": str(potential_level),



                "top_reasons": [



                    {



                        "feature": str(r["feature"]),



                        "importance_percent": float(r["importance_percent"]) if pd.notna(r["importance_percent"]) else 0.0,



                        "value": float(r["value"]) if pd.notna(r["value"]) else 0.0



                    } for r in top_reasons_data



                ]



            }



            



            print(f"📝 Đang lưu vào Supabase: customer_id={customer_id}, probability={probability:.4f}, level={potential_level}")



            print(f"🔑 Supabase URL: {SUPABASE_URL[:30]}... if 'None' thì chưa load được .env")



            



            db_response = supabase.table("repurchase_logs").insert(log_entry).execute()



            



            print(f"📦 Supabase response type: {type(db_response)}")



            print(f"📦 Supabase response attrs: {dir(db_response)}")



            



            if hasattr(db_response, 'data') and db_response.data:



                prediction_id = db_response.data[0].get("id")



                print(f"✅ Đã lưu vào log thành công! ID: {prediction_id}")



            else:



                print(f"⚠️ Không có data trong response. Response: {db_response}")



                if hasattr(db_response, '__dict__'):



                    print(f"📦 Response __dict__: {db_response.__dict__}")



                



        except Exception as db_err:



            import traceback



            print(f"❌ Exception khi lưu vào Supabase: {str(db_err)}")



            print(f"📋 Chi tiết lỗi:\n{traceback.format_exc()}")



            # Không raise exception để API vẫn trả kết quả dự đoán



        



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

        

        # Graceful degradation: nếu DB lỗi, trả về mảng rỗng

        if transactions is None:

            transactions = []

        

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

        # Graceful degradation: Log lỗi nhưng vẫn trả về response rỗng

        print(f"⚠️ Lỗi DB khi lấy lịch sử KH {customer_id}: {str(e)}")

        return CustomerHistoryResponse(

            success=True,

            customer_id=customer_id,

            count=0,

            transactions=[]

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

        

        # Graceful degradation: nếu DB lỗi, trả về danh sách rỗng

        if result is None:

            result = {"count": 0, "data": []}

        

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

        # Graceful degradation: Log lỗi nhưng vẫn trả về response rỗng

        print(f"⚠️ Lỗi DB khi lấy lịch sử dự báo: {str(e)}")

        return ApplicationsListResponse(

            success=True,

            count=0,

            page=page,

            page_size=page_size,

            data=[]

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

        

        # Graceful degradation: nếu DB lỗi

        if result is None:

            raise HTTPException(

                status_code=404,

                detail=f"Không tìm thấy dự báo với ID: {prediction_id} (hoặc DB không kết nối được)"

            )

        

        return {

            "success": True,

            "data": result

        }

        

    except HTTPException:

        raise

        

    except Exception as e:

        # Graceful degradation: Log lỗi và trả về 404 thay vì 500

        print(f"⚠️ Lỗi DB khi lấy chi tiết dự báo {prediction_id}: {str(e)}")

        raise HTTPException(

            status_code=404,

            detail=f"Không tìm thấy dự báo với ID: {prediction_id}"

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



    ║       Online Retail Repurchase Prediction API             ║



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



