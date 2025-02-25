from pymilvus import connections, utility
import os
from dotenv import load_dotenv

# Load biến môi trường từ file .env
load_dotenv()

# Kết nối đến Zilliz Cloud
print("🔄 Đang kết nối đến Zilliz Cloud...")
connections.connect(
    alias="default",
    uri=f"https://{os.getenv('ZILLIZ_HOST')}",
    token=os.getenv("ZILLIZ_API_KEY")  # Dùng API Key để xác thực
)

# Kiểm tra kết nối
collections = utility.list_collections()

# In ra danh sách collections
if collections:
    print(f"✅ Kết nối thành công! Collections hiện có: {collections}")
else:
    print("⚠️ Không có collection nào hoặc kết nối không thành công!")
    
