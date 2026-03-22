"""
==========================================
DATABASE MODULE - Supabase Connection
==========================================
Kết nối và tương tác với Supabase PostgreSQL Database
==========================================
"""

import os
from typing import Optional, Dict, Any, List
from supabase import create_client, Client
from dotenv import load_dotenv
import logging

# Load environment variables
load_dotenv()

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DatabaseManager:
    """
    Manager class để quản lý kết nối và tương tác với Supabase
    
    Attributes:
        client (Client): Supabase client instance
        is_connected (bool): Trạng thái kết nối
    """
    
    _instance = None
    
    def __new__(cls):
        """Singleton pattern - đảm bảo chỉ có 1 instance"""
        if cls._instance is None:
            cls._instance = super(DatabaseManager, cls).__new__(cls)
            cls._instance.client = None
            cls._instance.is_connected = False
        return cls._instance
    
    def connect(self) -> bool:
        """
        Khởi tạo kết nối đến Supabase
        
        Returns:
            bool: True nếu kết nối thành công, False nếu thất bại
        """
        try:
            supabase_url = os.getenv("SUPABASE_URL")
            supabase_key = os.getenv("SUPABASE_KEY")
            
            if not supabase_url or not supabase_key:
                logger.error("Thiếu SUPABASE_URL hoặc SUPABASE_KEY trong environment variables")
                return False
            
            self.client = create_client(supabase_url, supabase_key)
            self.is_connected = True
            logger.info("✅ Kết nối Supabase thành công")
            return True
            
        except Exception as e:
            logger.error(f"❌ Lỗi kết nối Supabase: {str(e)}")
            self.is_connected = False
            return False
    
    def check_connection(self) -> bool:
        """
        Kiểm tra kết nối đang hoạt động
        
        Returns:
            bool: True nếu kết nối OK
        """
        if not self.is_connected or not self.client:
            return False
        
        try:
            # Thử query để kiểm tra
            result = self.client.table("repurchase_logs").select("id", count="exact").limit(1).execute()
            return True
        except Exception as e:
            logger.warning(f"⚠️ Kết nối không ổn định: {str(e)}")
            self.is_connected = False
            return False
    
    def get_customer_transactions(self, customer_id: str) -> List[Dict[str, Any]]:
        """
        Lấy lịch sử giao dịch của một khách hàng từ raw_transactions
        
        Args:
            customer_id: customer_unique_id cần tìm
            
        Returns:
            List các dict chứa thông tin giao dịch
        """
        if not self.is_connected:
            self.connect()
        
        if not self.is_connected:
            return []
        
        try:
            result = self.client.table("raw_transactions") \
                .select("*") \
                .eq("customer_unique_id", customer_id) \
                .order("order_purchase_timestamp", desc=False) \
                .execute()
            
            return result.data if result.data else []
            
        except Exception as e:
            logger.error(f"❌ Lỗi lấy transactions: {str(e)}")
            return []
    
    def save_prediction(self, data: Dict[str, Any]) -> Optional[str]:
        """
        Lưu kết quả dự báo vào bảng repurchase_logs
        
        Args:
            data: Dict chứa thông tin dự báo
            
        Returns:
            str: ID của record đã tạo, hoặc None nếu lỗi
        """
        if not self.is_connected:
            self.connect()
        
        if not self.is_connected:
            logger.warning("⚠️ Không thể lưu prediction - Database không kết nối")
            return None
        
        try:
            result = self.client.table("repurchase_logs").insert(data).execute()
            
            if result.data and len(result.data) > 0:
                prediction_id = result.data[0].get('id')
                logger.info(f"✅ Đã lưu prediction: {prediction_id}")
                return prediction_id
            
            return None
            
        except Exception as e:
            logger.error(f"❌ Lỗi lưu prediction: {str(e)}")
            return None
    
    def get_predictions_history(self, page: int = 1, page_size: int = 20) -> Dict[str, Any]:
        """
        Lấy lịch sử các lượt dự báo với pagination
        
        Args:
            page: Số trang (bắt đầu từ 1)
            page_size: Số records mỗi trang
            
        Returns:
            Dict chứa count, data
        """
        if not self.is_connected:
            self.connect()
        
        if not self.is_connected:
            return {"count": 0, "data": []}
        
        try:
            # Tính range
            start = (page - 1) * page_size
            end = start + page_size - 1
            
            # Query với range
            result = self.client.table("repurchase_logs") \
                .select("*", count="exact") \
                .order("created_at", desc=True) \
                .range(start, end) \
                .execute()
            
            count = result.count if hasattr(result, 'count') else len(result.data)
            
            return {
                "count": count,
                "data": result.data if result.data else []
            }
            
        except Exception as e:
            logger.error(f"❌ Lỗi lấy history: {str(e)}")
            return {"count": 0, "data": []}
    
    def get_prediction_by_id(self, prediction_id: str) -> Optional[Dict[str, Any]]:
        """
        Lấy chi tiết một lượt dự báo theo ID
        
        Args:
            prediction_id: UUID của dự báo
            
        Returns:
            Dict chứa thông tin hoặc None
        """
        if not self.is_connected:
            self.connect()
        
        if not self.is_connected:
            return None
        
        try:
            result = self.client.table("repurchase_logs") \
                .select("*") \
                .eq("id", prediction_id) \
                .single() \
                .execute()
            
            return result.data if result.data else None
            
        except Exception as e:
            logger.error(f"❌ Lỗi lấy prediction: {str(e)}")
            return None


# Global instance
db_manager = DatabaseManager()


def get_db() -> DatabaseManager:
    """
    Dependency để lấy DatabaseManager instance
    
    Usage trong FastAPI:
        @app.get("/some-endpoint")
        async def some_endpoint(db: DatabaseManager = Depends(get_db)):
            ...
    """
    return db_manager


# ==========================================
# HELPER FUNCTIONS
# ==========================================

def init_database() -> bool:
    """
    Khởi tạo kết nối database khi startup
    
    Returns:
        bool: True nếu thành công
    """
    return db_manager.connect()


def check_db_health() -> bool:
    """
    Kiểm tra sức khỏe database connection
    
    Returns:
        bool: True nếu OK
    """
    return db_manager.check_connection()


# ==========================================
# TEST
# ==========================================
if __name__ == "__main__":
    print("Testing Database Connection...")
    
    # Test connection
    success = init_database()
    
    if success:
        print("✅ Kết nối thành công!")
        
        # Test lấy sample data
        sample = db_manager.get_predictions_history(page=1, page_size=3)
        print(f"📊 Sample predictions: {len(sample['data'])} records")
    else:
        print("❌ Kết nối thất bại!")
