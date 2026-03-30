# Backend API - Olist Repurchase Prediction

Backend REST API cho ứng dụng dự đoán khả năng mua lại của khách hàng (Online Retail Repurchase Prediction App). Xây dựng bằng **FastAPI** và triển khai trên **Render.com**.

## Tổng quan

API cung cấp các endpoint để:
- Dự đoán khả năng mua lại của khách hàng dựa trên lịch sử giao dịch
- Lưu trữ kết quả dự đoán vào PostgreSQL (Supabase)
- Truy xuất lịch sử dự đoán

## Tech Stack

| Công nghệ | Mục đích |
|-----------|----------|
| **FastAPI** | Web framework cho REST API |
| **LightGBM** | Machine Learning model |
| **Supabase (PostgreSQL)** | Database lưu trữ dữ liệu |
| **Pydantic** | Data validation |
| **Uvicorn** | ASGI server |

## Cấu trúc thư mục

```
backend/
├── app/
│   ├── main.py              # Entry point, API endpoints
│   ├── schemas.py             # Pydantic models (input/output validation)
│   ├── preprocess_utils.py    # Class xử lý dữ liệu đầu vào
│   └── database.py            # Kết nối Supabase
├── models/
│   ├── best_model.pkl         # Model LightGBM đã train
│   ├── preprocessor.pkl       # Preprocessor object
│   ├── optimal_threshold.pkl  # Ngưỡng phân loại
│   └── feature_importance.csv # Feature importance của model
├── requirements.txt           # Python dependencies
└── .env.example               # Mẫu file môi trường
```

## Cài đặt Local

### 1. Clone repository

```bash
git clone <repo-url>
cd backend
```

### 2. Tạo virtual environment

```bash
python -m venv venv

# Windows
venv\Scripts\activate

# macOS/Linux
source venv/bin/activate
```

### 3. Cài đặt dependencies

```bash
pip install -r requirements.txt
```

### 4. Cấu hình environment variables

Copy file `.env.example` thành `.env` và điền thông tin:

```bash
cp .env.example .env
```

Cấu hình các biến trong `.env`:

```env
SUPABASE_URL=https://your-project.supabase.co
SUPABASE_KEY=your-anon-key
```

### 5. Chạy server

```bash
# Development mode
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000

# Production mode
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

Server chạy tại: `http://localhost:8000`

Swagger UI: `http://localhost:8000/docs`

## API Endpoints

### Health Check

| Method | Endpoint | Mô tả |
|--------|----------|-------|
| `GET` | `/health` | Kiểm tra trạng thái API, model và database |

**Response:**
```json
{
  "status": "healthy",
  "model_loaded": true,
  "database_connected": true,
  "version": "1.0.0"
}
```

### Dự đoán

| Method | Endpoint | Mô tả |
|--------|----------|-------|
| `POST` | `/predict` | Dự đoán khả năng mua lại của khách hàng |

**Request Body:**
```json
{
  "customer_info": {
    "Customer_id": "CUST001",
    "snapshot_date": "2010-09-30"
  },
  "transactions": [
    {
      "Order_id": "ORD001",
      "Total_items": 3,
      "Log_items": 1.0986,
      "Order_date": "2010-09-15",
      "Order_value": 150.50,
      "Canceled_value": 0,
      "Order_n_categories": 2,
      "Order_n_lines": 3,
      "Is_canceled": 0,
      "Country": "United Kingdom"
    }
  ]
}
```

**Response:**
```json
{
  "probability": 0.75,
  "is_repurchase": true,
  "potential_level": "High",
  "top_reasons": [
    "L5M_total_value",
    "L3M_avg_value",
    "L1M_n_orders"
  ],
  "customer_id": "CUST001",
  "id": "550e8400-e29b-41d4-a716-446655440000"
}
```

### Lịch sử dự đoán

| Method | Endpoint | Mô tả |
|--------|----------|-------|
| `GET` | `/applications` | Lấy lịch sử dự đoán (có phân trang) |
| `GET` | `/applications/{id}` | Chi tiết một bản ghi |

**Query Parameters cho `/applications`:**
- `page`: Số trang (mặc định: 1)
- `limit`: Số bản ghi mỗi trang (mặc định: 10)

### Model Info

| Method | Endpoint | Mô tả |
|--------|----------|-------|
| `GET` | `/model-info` | Thông tin về model (version, ngày train, threshold) |

## Logic Phân loại Potential Level

| Xác suất | Level | Mô tả |
|----------|-------|-------|
| >= 60% | **High** | Khách hàng cực kỳ tiềm năng |
| 40% - 60% | **Medium** | Khách hàng cần chăm sóc |
| < 40% | **Low** | Khách hàng ít khả năng quay lại |

## Triển khai lên Render.com

### 1. Push code lên GitHub

```bash
git add .
git commit -m "Initial commit"
git push origin main
```

### 2. Tạo Web Service trên Render

1. Đăng nhập [render.com](https://render.com)
2. Click **New +** → **Web Service**
3. Chọn repository GitHub
4. Cấu hình:
   - **Name**: `olist-repurchase-api`
   - **Environment**: `Python 3`
   - **Build Command**: `pip install -r requirements.txt`
   - **Start Command**: `uvicorn app.main:app --host 0.0.0.0 --port 10000`
5. Thêm Environment Variables (từ Supabase)
6. Click **Create Web Service**

### 3. Kiểm tra deployment

- API URL: `https://your-api.onrender.com`
- Swagger UI: `https://your-api.onrender.com/docs`

## Cấu hình CORS

API đã cấu hình CORS cho phép frontend từ Vercel:

```python
app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://your-frontend.vercel.app"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
```

## Tài liệu tham khảo

- [FastAPI Documentation](https://fastapi.tiangolo.com/)
- [Supabase Python Client](https://supabase.com/docs/reference/python/)
- [LightGBM Documentation](https://lightgbm.readthedocs.io/)

## License

MIT License - VTI Academy Mini Project
