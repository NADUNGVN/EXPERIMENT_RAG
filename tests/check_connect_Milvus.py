from pymilvus import connections, Collection, utility

# 🔹 Kết nối với Milvus trước khi thao tác với Collection
connections.connect(alias="default", host="localhost", port="19530")

# 🔹 Kiểm tra kết nối đã thành công chưa
if not connections.has_connection("default"):
    print("❌ Không thể kết nối Milvus. Hãy kiểm tra Milvus server!")
    exit()

# 🔹 Kiểm tra danh sách collections
collections_list = utility.list_collections()
print(f"📂 Danh sách collections: {collections_list}")

# 🔹 Kiểm tra collection tồn tại không
collection_name = "tthc_vectors"
if collection_name not in collections_list:
    print(f"❌ Collection {collection_name} không tồn tại!")
    exit()

# 🔹 Load collection để kiểm tra dữ liệu
collection = Collection(collection_name)
collection.load()

# 🔹 Kiểm tra schema của collection
print(f"📑 Schema của collection: {collection.schema}")

# 🔹 Kiểm tra số lượng vector
print(f"📊 Số lượng vector trong collection: {collection.num_entities}")

print("📌 Danh sách fields trong collection:")
for field in collection.schema.fields:
    print(f"- Field: {field.name}, Type: {field.dtype}")

# 🔹 Kiểm tra index có tồn tại không
print("📌 Danh sách index:")
for index in collection.indexes:
    print(f"🔹 Index trên field: {index.field_name}, Params: {index.params}")

print("✅ Kết nối Milvus thành công!")

# Tạo một vector giả lập để test truy vấn
search_vector = [[0.0] * 768]

# Thực hiện truy vấn
results = collection.search(
    data=search_vector,
    anns_field="embedding",  # Trường chứa vector embeddings
    param={"metric_type": "L2", "params": {"nprobe": 10}},  # Các tham số tìm kiếm
    limit=4,  # Giới hạn số lượng kết quả trả về
    output_fields=["content"]  # Lấy nội dung văn bản từ kết quả tìm kiếm
)

# In kết quả tìm kiếm
print("🔍 Kết quả truy vấn:")
if results:
    for i, res in enumerate(results):
        print(f"\n🔹 Kết quả {i+1}:")
        for hit in res:
            print(f"   🏷️ Distance: {hit.distance}")
            print(f"   📄 Content: {hit.entity.get('content')}")
else:
    print("❌ Không tìm thấy kết quả nào!")
