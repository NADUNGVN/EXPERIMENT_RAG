import os
from typing import List, Dict
from dotenv import load_dotenv
from langchain_together import ChatTogether, TogetherEmbeddings
from langchain_community.vectorstores import Zilliz
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
import logging
from datetime import datetime
import asyncio
from concurrent.futures import TimeoutError
import numpy as np
from pymilvus import Collection, DataType, FieldSchema, CollectionSchema, connections, utility

# Cấu hình logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class RAGChatbot:
    def __init__(self, debug_mode=False):
        self.debug_mode = debug_mode
        self.chat_history = []
        load_dotenv()
        self.init_components()
        self.create_prompt_template()
        self.setup_retrieval_qa()

    def init_milvus_collection(self):
        """Kiểm tra kết nối tới Zilliz Cloud"""
        try:
            connections.connect(
                uri=f"https://{os.getenv('ZILLIZ_HOST')}",
                token=os.getenv("ZILLIZ_API_KEY")
            )
            
            collection_name = "tthc_vectors"
            
            if collection_name in utility.list_collections():
                logger.info(f"Đã kết nối thành công tới collection {collection_name}")
            else:
                raise Exception(f"Collection {collection_name} không tồn tại")
                
        except Exception as e:
            logger.error(f"Lỗi khi kết nối tới Zilliz: {str(e)}")
            raise

    def init_components(self):
        """Khởi tạo các thành phần chính của chatbot"""
        try:
            logger.info("Bắt đầu khởi tạo embedding model...")
            self.embeddings = TogetherEmbeddings(
                model="togethercomputer/m2-bert-80M-32k-retrieval",
                together_api_key=os.getenv("TOGETHER_API_KEY_2")
            )
            logger.info("Khởi tạo embedding model thành công")

            logger.info("Bắt đầu khởi tạo LLM...")
            self.llm = ChatTogether(
                model="meta-llama/Llama-3.3-70B-Instruct-Turbo",
                together_api_key=os.getenv("TOGETHER_API_KEY_2"),
                callbacks=[StreamingStdOutCallbackHandler()]
            )
            logger.info("Khởi tạo LLM thành công")

            self.init_milvus_collection()

            logger.info("Kết nối vector store...")
            self.vector_store = Zilliz(
                embedding_function=self.embeddings,
                collection_name="tthc_vectors",
                connection_args={
                    "uri": f"https://{os.getenv('ZILLIZ_HOST')}",
                    "token": os.getenv("ZILLIZ_API_KEY"),
                    "secure": True
                },
                vector_field="embedding",
                text_field="content"
            )
            logger.info("Khởi tạo các thành phần thành công")

        except Exception as e:
            logger.error(f"Lỗi khởi tạo components: {str(e)}")
            raise

    def create_prompt_template(self):
        """Tạo template cho prompt"""
        template = """Sử dụng thông tin sau để trả lời câu hỏi. Nếu không thể trả lời được câu hỏi, hãy nói rằng bạn không biết.

Thông tin: {context}

Câu hỏi: {question}

Trả lời:"""

        self.prompt = PromptTemplate(
            template=template,
            input_variables=["context", "question"]
        )

    def setup_retrieval_qa(self):
        """Thiết lập chain RetrievalQA"""
        try:
            self.qa_chain = RetrievalQA.from_chain_type(
                llm=self.llm,
                chain_type="stuff",
                retriever=self.vector_store.as_retriever(
                    search_kwargs={"k": 3}
                ),
                return_source_documents=True,
                chain_type_kwargs={"prompt": self.prompt}
            )
            logger.info("Khởi tạo QA chain thành công")
        except Exception as e:
            logger.error(f"Lỗi khởi tạo QA chain: {str(e)}")
            raise

    async def get_answer_with_timeout(self, question: str, timeout: int = 30) -> Dict:
        """Lấy câu trả lời với timeout"""
        try:
            result = await asyncio.wait_for(
                self.qa_chain.ainvoke({"query": question}),
                timeout=timeout
            )
            return {
                "answer": result["result"],
                "sources": [doc.page_content for doc in result["source_documents"]]
            }
        except TimeoutError:
            return {
                "answer": "Xin lỗi, câu trả lời mất quá nhiều thời gian để xử lý.",
                "sources": []
            }
        except Exception as e:
            logger.error(f"Lỗi khi lấy câu trả lời: {str(e)}")
            return {
                "answer": "Xin lỗi, tôi không thể trả lời câu hỏi này lúc này.",
                "sources": []
            }

    def analyze_embedding_process(self, question: str):
        """Phân tích và hiển thị chi tiết quá trình embedding"""
        try:
            print("\n🔍 Phân tích quá trình embedding:")
            
            # 1. Tạo embedding cho câu hỏi
            print("\n1. Embedding câu hỏi:")
            query_embedding = self.embeddings.embed_query(question)
            print(f"- Câu hỏi: {question}")
            print(f"- Độ dài vector: {len(query_embedding)}")
            print(f"- Một phần vector: {query_embedding[:5]}...")

            # 2. Tìm kiếm trong Zilliz
            print("\n2. Kết quả tìm kiếm từ Zilliz:")
            docs = self.vector_store.similarity_search(
                question,
                k=3,
                return_metadata=True
            )

            # 3. Hiển thị kết quả tìm kiếm
            for idx, doc in enumerate(docs, 1):
                print(f"\nDocument {idx}:")
                print(f"- Nội dung: {doc.page_content[:100]}...")
                if hasattr(doc, 'metadata'):
                    print(f"- Metadata: {doc.metadata}")

            return {
                "query_embedding": query_embedding,
                "results": docs
            }

        except Exception as e:
            logger.error(f"Lỗi khi phân tích embedding: {str(e)}")
            return None

    def save_chat_history(self, question: str, answer: Dict):
        """Lưu lịch sử chat"""
        self.chat_history.append({
            "timestamp": datetime.now().isoformat(),
            "question": question,
            "answer": answer["answer"],
            "sources": answer["sources"]
        })

    def analyze_question(self, question: str) -> Dict:
        """Phân tích câu hỏi để xác định lĩnh vực, tên thủ tục và ý định người dùng"""
        # Từ điển mapping từ khóa với thủ tục
        keywords = {
            "khai tử": {
                "linh_vuc": "Hộ tịch",
                "ten_thu_tuc": ["đăng ký khai tử", "khai tử"]
            },
            "khai sinh": {
                "linh_vuc": "Hộ tịch",
                "ten_thu_tuc": ["đăng ký khai sinh"]
            }
        }
        
        # Từ điển mapping ý định với section_name
        intent_mapping = {
            "trình tự": {
                "section_name": "Trình tự thực hiện",
                "keywords": ["trình tự", "các bước", "bước", "quy trình", "thực hiện như thế nào", 
                        "làm như thế nào", "thực hiện ra sao", "tiến hành"]
            },
            "cách thức": {
                "section_name": "Cách thức thực hiện",
                "keywords": ["cách thức", "hình thức", "thực hiện ở đâu", "nộp ở đâu", "địa điểm"]
            },
            "hồ sơ": {
                "section_name": "Thành phần hồ sơ",
                "keywords": ["hồ sơ", "giấy tờ", "tài liệu", "văn bản", "cần những gì", 
                        "cần chuẩn bị", "cần mang theo"]
            },
            "thời hạn": {
                "section_name": "Thời hạn giải quyết",
                "keywords": ["thời hạn", "bao lâu", "mất bao nhiêu thời gian", "trong vòng", 
                        "thời gian"]
            },
            "phí": {
                "section_name": "Phí, lệ phí",
                "keywords": ["phí", "lệ phí", "chi phí", "tốn", "mất bao nhiêu tiền", 
                        "bao nhiêu tiền"]
            },
            "pháp lý": {
                "section_name": "Căn cứ pháp lý",
                "keywords": ["căn cứ", "pháp lý", "luật", "nghị định", "quy định"]
            },
            "điều kiện": {
                "section_name": "Yêu cầu, điều kiện thực hiện",
                "keywords": ["điều kiện", "yêu cầu", "đối tượng", "ai được", "điều kiện gì"]
            }
        }
        
        result = {
            "linh_vuc": None,
            "ten_thu_tuc": None,
            "section_name": None,
            "intent": None
        }
        
        # Phân tích lĩnh vực và tên thủ tục
        question_lower = question.lower()
        for key, value in keywords.items():
            if key in question_lower:
                result["linh_vuc"] = value["linh_vuc"]
                for proc_name in value["ten_thu_tuc"]:
                    if proc_name in question_lower:
                        result["ten_thu_tuc"] = proc_name
                        break
                break
        
        # Phân tích ý định người dùng
        for intent, mapping in intent_mapping.items():
            for keyword in mapping["keywords"]:
                if keyword in question_lower:
                    result["intent"] = intent
                    result["section_name"] = mapping["section_name"]
                    break
            if result["intent"]:  # Nếu đã tìm thấy ý định thì dừng
                break
                
        # Nếu không tìm thấy ý định cụ thể, mặc định là "Trình tự thực hiện"
        if not result["section_name"]:
            result["section_name"] = "Trình tự thực hiện"
            result["intent"] = "trình tự"
        
        logger.info(f"Kết quả phân tích câu hỏi: {result}")
        return result

    def get_relevant_context(self, question: str) -> List[Dict]:
        """Lấy context liên quan từ vector store với filter metadata"""
        try:
            # Phân tích câu hỏi
            analysis = self.analyze_question(question)
            
            # Tạo filter dựa trên kết quả phân tích
            filter_dict = {}
            
            # Chỉ thêm các điều kiện filter nếu có giá trị
            if analysis.get("linh_vuc"):
                filter_dict["linh_vuc"] = analysis["linh_vuc"]
                
            if analysis.get("ten_thu_tuc"):
                filter_dict["ten_thu_tuc"] = {"$like": f"%{analysis['ten_thu_tuc']}%"}
                
            if analysis.get("section_name"):
                filter_dict["section_name"] = analysis["section_name"]

            logger.info(f"Áp dụng filter: {filter_dict}")
            
            # Thực hiện tìm kiếm với filter
            docs = self.vector_store.similarity_search(
                question,
                k=3,
                filter=filter_dict
            )
            
            return docs

        except Exception as e:
            logger.error(f"Lỗi khi tìm kiếm context: {str(e)}")
            return []

    def answer_question(self, question: str) -> Dict:
        """Trả lời câu hỏi dựa trên context phù hợp"""
        try:
            # Lấy context với filter
            relevant_docs = self.get_relevant_context(question)
            
            if not relevant_docs:
                return {
                    "answer": "Xin lỗi, tôi không tìm thấy thông tin liên quan đến câu hỏi của bạn.",
                    "sources": []
                }

            # In ra context tìm được để debug
            print("\n🔍 Context tìm được từ vector store:")
            for idx, doc in enumerate(relevant_docs, 1):
                print(f"\n=== Document {idx} ===")
                print(f"Nội dung: {doc.page_content}")
                print(f"Metadata: {doc.metadata}")
                print("=" * 50)

            # Sử dụng QA chain với context đã lọc
            result = self.qa_chain(
                {
                    "query": question,
                    "input_documents": relevant_docs
                },
                return_only_outputs=False
            )

            # Chuẩn bị response với metadata đầy đủ
            response = {
                "answer": result["result"],
                "sources": [
                    {
                        "content": doc.page_content,
                        "metadata": doc.metadata
                    } for doc in result["source_documents"]
                ]
            }

            # Lưu lịch sử chat
            self.save_chat_history(question, response)

            return response

        except Exception as e:
            logger.error(f"Lỗi khi xử lý câu hỏi: {str(e)}")
            return {
                "error": "Có lỗi xảy ra khi xử lý câu hỏi của bạn",
                "details": str(e)
            }

        except Exception as e:
            logger.error(f"Lỗi khi xử lý câu hỏi: {str(e)}")
            return {
                "error": "Có lỗi xảy ra khi xử lý câu hỏi của bạn",
                "details": str(e)
            }

    def chat(self, question: str):
        """Interface chính để tương tác với chatbot"""
        try:
            # Hiển thị kết quả phân tích nếu ở chế độ debug
            if self.debug_mode:
                analysis = self.analyze_question(question)
                print("\n🔍 Phân tích câu hỏi:")
                print(f"- Lĩnh vực: {analysis['linh_vuc']}")
                print(f"- Thủ tục: {analysis['ten_thu_tuc']}")
                print(f"- Section: {analysis['section_name']}")
                
            response = self.answer_question(question)
            
            if "error" in response:
                print(f"❌ Lỗi: {response['error']}")
                return
            
            print("\n🤖 Trả lời:")
            print(response["answer"])
            
            if response["sources"]:
                print("\n📚 Nguồn tham khảo:")
                for idx, source in enumerate(response["sources"], 1):
                    print(f"\n{idx}. Từ thủ tục: {source['metadata'].get('ten_thu_tuc', 'Không xác định')}")
                    print(f"   Lĩnh vực: {source['metadata'].get('linh_vuc', 'Không xác định')}")
                    print(f"   Phần: {source['metadata'].get('section_name', 'Không xác định')}")

                print("\n🔍 Nội dung chi tiết:")
                for idx, source in enumerate(response["sources"], 1):
                    print(f"\n--- Đoạn {idx} ---")
                    print(f"{source['content']}")
                    print("-" * 50)

        except Exception as e:
            logger.error(f"Lỗi trong quá trình chat: {str(e)}")
            print("❌ Có lỗi xảy ra, vui lòng thử lại sau")

def main():
    # Khởi tạo chatbot với debug mode
    chatbot = RAGChatbot(debug_mode=True)
    
    print("🤖 Chatbot đã sẵn sàng! Gõ 'quit' để thoát.")
    
    while True:
        question = input("\n👤 Câu hỏi của bạn: ").strip()
        
        if question.lower() == 'quit':
            print("👋 Tạm biệt!")
            break
            
        chatbot.chat(question)

if __name__ == "__main__":
    main()
