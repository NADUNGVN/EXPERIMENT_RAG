import os
from dotenv import load_dotenv
from langchain_together import ChatTogether, TogetherEmbeddings

# Load API keys từ .env
load_dotenv()
TOGETHER_API_KEY = os.getenv("TOGETHER_API_KEY_2")  # Dùng chung một API key

# Khởi tạo mô hình chat từ Together AI thông qua LangChain
chat_model = ChatTogether(
    model="meta-llama/Llama-3.3-70B-Instruct-Turbo",
    together_api_key=TOGETHER_API_KEY
)

# Khởi tạo model tạo embedding
embedding_model = TogetherEmbeddings(
    model="togethercomputer/m2-bert-80M-32k-retrieval",
    together_api_key=TOGETHER_API_KEY
)

def test_llm_api(query):
    """Hàm để kiểm tra API LLM với một truy vấn cụ thể."""
    try:
        response = chat_model.invoke(query)
        print("✅ Kết quả từ LLM:", response)
    except Exception as e:
        print(f"❌ Lỗi khi gọi API LLM: {str(e)}")

def create_embedding(input_text):
    """Hàm để tạo embedding cho một đoạn văn bản."""
    try:
        embedding = embedding_model.embed_query(input_text)
        print("✅ Embedding:", embedding)
    except Exception as e:
        print(f"❌ Lỗi khi tạo embedding: {str(e)}")

if __name__ == "__main__":
    test_query = "Thủ tục làm hộ chiếu cần những giấy tờ gì?"
    test_llm_api(test_query)
    
    # Tạo embedding cho một đoạn văn bản
    create_embedding("Thủ tục làm hộ chiếu cần những giấy tờ gì?")
