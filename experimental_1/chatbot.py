import os
import logging
from dotenv import load_dotenv
from langchain_community.vectorstores import Zilliz
from langchain_together import ChatTogether, TogetherEmbeddings
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA

# Cấu hình logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load API Key từ .env
load_dotenv()

class RAGChatbot:
    def __init__(self):
        self.api_key = os.getenv("TOGETHER_API_KEY_1")
        self.llm_api_key = os.getenv("TOGETHER_API_KEY_2")

        # Khởi tạo model embedding
        self.embeddings = TogetherEmbeddings(
            model="togethercomputer/m2-bert-80M-32k-retrieval",
            together_api_key=self.api_key
        )

        # Khởi tạo LLM (Llama 3)
        self.llm = ChatTogether(
            model="meta-llama/Llama-3.3-70B-Instruct-Turbo",
            together_api_key=self.llm_api_key
        )

        # Kết nối đến Zilliz
        self.vector_store = Zilliz(
            embedding_function=self.embeddings,
            collection_name="tthc_vectors_1",
            connection_args={"uri": f"https://{os.getenv('ZILLIZ_HOST')}", "token": os.getenv("ZILLIZ_API_KEY")},
            vector_field="embedding",
            text_field="content"
        )

        # Tạo template cho prompt
        self.create_prompt_template()
        self.setup_retrieval_qa()

    def create_prompt_template(self):
        """Tạo template cho LLM"""
        self.prompt = PromptTemplate(
            template="""Dựa trên thông tin dưới đây, hãy trả lời câu hỏi một cách ngắn gọn và chính xác:\n\n{context}\n\nCâu hỏi: {question}\n\nTrả lời:""",
            input_variables=["context", "question"]
        )

    def setup_retrieval_qa(self):
        """Thiết lập truy vấn tìm kiếm"""
        self.qa_chain = RetrievalQA.from_chain_type(
            llm=self.llm,
            retriever=self.vector_store.as_retriever(search_kwargs={"k": 5}),
            return_source_documents=True,
            chain_type_kwargs={"prompt": self.prompt}
        )

    def analyze_question(self, question):  
        """
        Phân tích câu hỏi để trích xuất `intent` và `section_name` cần tìm kiếm.
        """
        question = question.lower()
        intent_mapping = {
            "trình tự": ["bước thực hiện", "quy trình", "trình tự"],
            "cách thức": ["làm như thế nào", "nộp hồ sơ", "thực hiện như thế nào"],
            "hồ sơ": ["giấy tờ", "hồ sơ cần có", "cần chuẩn bị những gì"],
            "thời hạn": ["mất bao lâu", "thời gian xử lý"],
            "phí": ["chi phí", "phí bao nhiêu"],
            "pháp lý": ["quy định pháp luật", "căn cứ pháp lý"],
            "điều kiện": ["điều kiện cần có", "ai được phép"]
        }

        detected_intent = "khác"
        detected_section = "Thông tin chung"

        for intent, keywords in intent_mapping.items():
            for keyword in keywords:
                if keyword in question:
                    detected_intent = intent
                    detected_section = keyword
                    break

        return {"intent": detected_intent, "section_name": detected_section}

    def answer_question(self, question):
        """Tìm kiếm dữ liệu dựa trên intent & section_name từ phân tích câu hỏi."""
        analysis = self.analyze_question(question)
        intent = analysis["intent"]
        section_name = analysis["section_name"]

        print(f"\n🔍 **PHÂN TÍCH CÂU HỎI:**")
        print(f"🛠 Intent: {intent}")
        print(f"📌 Section Name: {section_name}")

        # Tạo bộ lọc dựa vào intent & section_name
        filter_dict = {
            "intent": intent,
            "section_name": section_name
        }

        # Tìm kiếm dữ liệu theo bộ lọc đã phân tích
        docs = self.vector_store.similarity_search(question, k=10, filter=filter_dict)

        # Hiển thị kết quả tìm được từ Zilliz
        if docs:
            print("\n🔍 **KẾT QUẢ EMBEDDING TÌM ĐƯỢC:**")
            for idx, doc in enumerate(docs, 1):
                print(f"\n=== 📝 **TÀI LIỆU {idx}** ===")
                print(f"📜 **Nội dung (content):**\n{doc.page_content}")  # 🔥 In toàn bộ nội dung từ Zilliz
                print(f"📌 **Metadata:** {doc.metadata}")
                print(f"🔢 **Embedding:** {doc.metadata.get('embedding', 'Không có')[:3]}...")  
                print("=" * 50)

        else:
            print("\n❌ **Không tìm thấy tài liệu liên quan!**")

        # Nếu không tìm thấy dữ liệu, trả về thông báo
        if not docs:
            return "❌ Không tìm thấy thông tin phù hợp."

        # Gửi dữ liệu đến LLM để tạo câu trả lời
        response = self.qa_chain.invoke({"query": question, "input_documents": docs})
        return response["result"]

    def chat(self):
        """Khởi động chatbot"""
        print("🤖 Chatbot sẵn sàng! Nhập câu hỏi của bạn:")
        while True:
            question = input("👤 Bạn: ").strip()
            if question.lower() == "quit":
                break
            print(f"🤖 Bot: {self.answer_question(question)}")

if __name__ == "__main__":
    RAGChatbot().chat()
