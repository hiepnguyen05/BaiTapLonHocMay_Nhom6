# 📩 Spam Message Classification

Dự án phân loại tin nhắn rác (**Spam**) và tin nhắn hợp lệ (**Ham**) bằng **Machine Learning** và triển khai ứng dụng web với **Flask**.

---

## 🔬 Phần 1: Huấn luyện mô hình

Quy trình huấn luyện mô hình gồm các bước:

1. **Tiền xử lý dữ liệu**: xóa URL, email, số điện thoại, ký tự thừa, viết thường, xử lý từ lặp.  
2. **Trích xuất đặc trưng**:
   - Bag of Words (BoW), TF-IDF  
   - Đặc trưng thủ công: độ dài văn bản, số từ, số chữ hoa, số lần xuất hiện `$`, `!`, `?`, có URL, có số điện thoại...
3. **Huấn luyện mô hình**:
   - Thử nghiệm: MultinomialNB, Logistic Regression, SVM  
   - Mô hình tốt nhất: **Multinomial Naive Bayes (BoW)**

```python
import joblib
joblib.dump(best_model, 'spam_model.pkl')
joblib.dump(best_vectorizer, 'spam_vectorizer.pkl')
