import os
from dotenv import load_dotenv
from together import Together

# Load API keys từ .env
load_dotenv()
TOGETHER_API_KEY_1 = os.getenv("TOGETHER_API_KEY_1")
TOGETHER_API_KEY_2 = os.getenv("TOGETHER_API_KEY_2")

def test_llm_api_2(query):
    """Hàm để kiểm tra API LLM với một truy vấn cụ thể, sử dụng API key 2."""
    try:
        client = Together(api_key=TOGETHER_API_KEY_2)
        response = client.chat.completions.create(
            model="meta-llama/Llama-3.3-70B-Instruct-Turbo",
            messages=[{"role": "user", "content": query}],
        )
        print("Kết quả từ LLM (API key 2):", response.choices[0].message.content)
    except Exception as e:
        print(f"❌ Lỗi khi gọi API LLM (API key 2): {str(e)}")

def create_embedding(input_text):
    """Hàm để tạo embedding cho một đoạn văn bản, sử dụng model embedding."""
    try:
        client = Together(api_key=TOGETHER_API_KEY_2)  # Hoặc dùng API key 2 nếu cần
        response = client.embeddings.create(
            model="togethercomputer/m2-bert-80M-32k-retrieval",  # Model embedding
            input=input_text,
        )
        print("Embedding:", response.data[0].embedding)
    except Exception as e:
        print(f"❌ Lỗi khi tạo embedding: {str(e)}")

if __name__ == "__main__":
    test_query = "Thủ tục làm hộ chiếu cần những giấy tờ gì?"
    test_llm_api_2(test_query)
    
    # Tạo embedding cho một đoạn văn bản
    create_embedding("Thủ tục làm hộ chiếu cần những giấy tờ gì?")
