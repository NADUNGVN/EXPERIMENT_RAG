from pymilvus import connections, utility, Collection, DataType
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def check_vector_database(host: str = "localhost", port: str = "19530"):
    """
    Kiểm tra toàn diện vector database
    """
    try:
        # Kết nối đến Milvus
        connections.connect("default", host=host, port=port)
        logger.info("✅ Kết nối Milvus thành công")
        
        # Lấy danh sách collections
        collections = utility.list_collections()
        logger.info(f"📑 Tìm thấy {len(collections)} collections")
        
        # Kiểm tra chi tiết từng collection
        for coll_name in collections:
            collection = Collection(coll_name)
            collection.load()
            
            try:
                # Thông tin cơ bản
                print(f"\n🔍 Collection: {coll_name}")
                print(f"   ├── Số lượng entities: {collection.num_entities}")
                print(f"   ├── Primary field: {collection.schema.primary_field.name}")
                
                # Thông tin về fields
                print("   ├── Fields:")
                for field in collection.schema.fields:
                    if field.dtype == DataType.FLOAT_VECTOR:
                        print(f"   │   ├── {field.name} (Vector dim={field.params['dim']})")
                    else:
                        print(f"   │   ├── {field.name} ({field.dtype})")
                
                # Thông tin về indexes
                print("   └── Indexes:")
                for idx in collection.indexes:
                    print(f"       └── {idx.field_name}: {idx.params}")
                    
            finally:
                collection.release()
                
    except Exception as e:
        logger.error(f"❌ Lỗi: {str(e)}")
        raise
    finally:
        connections.disconnect("default")

if __name__ == "__main__":
    check_vector_database()
