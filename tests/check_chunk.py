import json
from datetime import datetime  
import re
from typing import Dict, List, Tuple
import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter
import os

def check_chunks_quality(file_path: str = r"C:\Users\nguye\OneDrive\Máy tính\CHATBOT\data\chunks\chunks_20250204_122842.json"):
    class ChunkQualityChecker:
        def __init__(self):
            self.min_content_words = 20
            self.max_content_words = 200
            self.required_metadata_fields = [
                "file_name", "processed_date", "ma_thu_tuc",
                "ten_thu_tuc", "cap_thuc_hien", "linh_vuc",
                "section_name", "created_at"
            ]
            self.results = []
            
        def validate_datetime(self, date_str: str) -> bool:
            try:
                datetime.fromisoformat(date_str)
                return True
            except ValueError:
                return False
        
        def validate_ma_thu_tuc(self, ma_thu_tuc: str) -> bool:
            pattern = r'^\d+\.\d{6}\.\d{3}\.\d{2}\.\d{2}\.[A-Z0-9]+$'
            return bool(re.match(pattern, ma_thu_tuc))
        
        def check_content_quality(self, content: str) -> Tuple[bool, List[str]]:
            issues = []
            word_count = len(content.split())
            
            if word_count < self.min_content_words:
                issues.append(f"Nội dung quá ngắn ({word_count} từ < {self.min_content_words})")
            if word_count > self.max_content_words:
                issues.append(f"Nội dung quá dài ({word_count} từ > {self.max_content_words})")
                
            if not content.strip().endswith(('.', '?', '!')):
                issues.append("Nội dung không kết thúc bằng dấu câu phù hợp")
                
            if len(set(content.split())) / len(content.split()) < 0.3:
                issues.append("Nội dung có nhiều từ lặp lại")
                
            return len(issues) == 0, issues
        
        def check_chunk(self, chunk: Dict) -> Dict:
            result = {
                "chunk_id": len(self.results) + 1,
                "metadata_issues": [],
                "content_issues": [],
                "status": "PASS"
            }
            
            metadata = chunk.get("metadata", {})
            for field in self.required_metadata_fields:
                if field not in metadata:
                    result["metadata_issues"].append(f"Thiếu trường metadata: {field}")
                    
            if "processed_date" in metadata and not self.validate_datetime(metadata["processed_date"]):
                result["metadata_issues"].append("Định dạng processed_date không hợp lệ")
                
            if "created_at" in metadata and not self.validate_datetime(metadata["created_at"]):
                result["metadata_issues"].append("Định dạng created_at không hợp lệ")
                
            if "ma_thu_tuc" in metadata and not self.validate_ma_thu_tuc(metadata["ma_thu_tuc"]):
                result["metadata_issues"].append("Format mã thủ tục không hợp lệ")
                
            content = chunk.get("content", "")
            content_valid, content_issues = self.check_content_quality(content)
            result["content_issues"].extend(content_issues)
            
            if result["metadata_issues"] or result["content_issues"]:
                result["status"] = "FAIL"
                
            self.results.append(result)
            return result
        
        def analyze_chunks(self, chunks: List[Dict]) -> None:
            for chunk in chunks:
                self.check_chunk(chunk)
                
        def generate_report(self) -> None:
            df = pd.DataFrame(self.results)
            
            total_chunks = len(self.results)
            passed_chunks = len([r for r in self.results if r["status"] == "PASS"])
            failed_chunks = total_chunks - passed_chunks
            
            plt.figure(figsize=(15, 10))
            
            plt.subplot(2, 2, 1)
            plt.pie([passed_chunks, failed_chunks], 
                    labels=['PASS', 'FAIL'],
                    autopct='%1.1f%%',
                    colors=['green', 'red'])
            plt.title('Tỷ lệ Chunks Pass/Fail')
            
            all_issues = []
            for r in self.results:
                all_issues.extend(r["metadata_issues"])
                all_issues.extend(r["content_issues"])
            
            issue_counts = Counter(all_issues)
            if issue_counts:
                plt.subplot(2, 2, 2)
                issues, counts = zip(*issue_counts.most_common(5))
                plt.barh(issues, counts)
                plt.title('Top 5 vấn đề phổ biến nhất')
            
            plt.tight_layout()

            # 🔹 Tạo thư mục "figure" nếu chưa tồn tại
            base_dir = r"C:\Users\nguye\OneDrive\Máy tính\CHATBOT\data"
            figure_dir = os.path.join(base_dir, "figures")
            os.makedirs(figure_dir, exist_ok=True)

            # 🔹 Lưu kết quả vào thư mục "figure"
            current_time = datetime.now().strftime('%Y%m%d_%H%M%S')
            figure_path = os.path.join(figure_dir, f'chunk_quality_report_{current_time}.png')

            plt.savefig(figure_path, dpi=300, bbox_inches='tight')
            plt.show()
            
            print("\n=== BÁO CÁO KIỂM TRA CHẤT LƯỢNG CHUNKS ===")
            print(f"Tổng số chunks: {total_chunks}")
            print(f"Số chunks đạt yêu cầu: {passed_chunks} ({passed_chunks/total_chunks*100:.1f}%)")
            print(f"Số chunks không đạt: {failed_chunks} ({failed_chunks/total_chunks*100:.1f}%)")
            print(f"\n📁 Biểu đồ đã được lưu tại: {figure_path}")
            
            if issue_counts:
                print("\nCác vấn đề phổ biến nhất:")
                for issue, count in issue_counts.most_common(5):
                    print(f"- {issue}: {count} lần")

            # 🔹 Lưu báo cáo dạng text
            report_path = os.path.join(figure_dir, f'chunk_quality_report_{current_time}.txt')
            with open(report_path, 'w', encoding='utf-8') as f:
                f.write("=== BÁO CÁO KIỂM TRA CHẤT LƯỢNG CHUNKS ===\n")
                f.write(f"Tổng số chunks: {total_chunks}\n")
                f.write(f"Số chunks đạt yêu cầu: {passed_chunks} ({passed_chunks/total_chunks*100:.1f}%)\n")
                f.write(f"Số chunks không đạt: {failed_chunks} ({failed_chunks/total_chunks*100:.1f}%)\n")
                if issue_counts:
                    f.write("\nCác vấn đề phổ biến nhất:\n")
                    for issue, count in issue_counts.most_common(5):
                        f.write(f"- {issue}: {count} lần\n")

    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            chunks = json.load(f)
    except Exception as e:
        print(f"Lỗi khi đọc file: {str(e)}")
        return
    
    checker = ChunkQualityChecker()
    checker.analyze_chunks(chunks)
    checker.generate_report()
    
    return checker.results

# Chạy kiểm tra
if __name__ == "__main__":
    results = check_chunks_quality()
