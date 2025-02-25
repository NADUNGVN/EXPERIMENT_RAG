import os
from dotenv import load_dotenv
from together import Together

# ✅ Load API key từ .env
load_dotenv()
TOGETHER_API_KEY = os.getenv("TOGETHER_API_KEY")
if not TOGETHER_API_KEY:
    raise ValueError("❌ API key is missing! Make sure .env is correctly set.")

def test_llm_api(query):
    """Hàm để kiểm tra API LLM với một truy vấn cụ thể."""
    try:
        client = Together(api_key=TOGETHER_API_KEY)  # Khởi tạo client với API key

        response = client.chat.completions.create(
            model="meta-llama/Llama-3.3-70B-Instruct-Turbo",
            messages=[{"role": "user", "content": query}],
        )

        print("Kết quả từ LLM:", response.choices[0].message.content)
    except Exception as e:
        print(f"❌ Lỗi khi gọi API LLM: {str(e)}")

if __name__ == "__main__":
    test_query = "Thủ tục làm hộ chiếu cần những giấy tờ gì?"
    test_llm_api(test_query)
