import joblib
from flask import Flask, render_template, request
from scipy.sparse import hstack, csr_matrix
import pandas as pd
import re
import string

model = joblib.load('spam_model.pkl')
vectorizer = joblib.load('spam_vectorizer.pkl')

app = Flask(__name__)

# Tái tạo hàm tiền xử lý và trích xuất đặc trưng thủ công
# Đây là các bước quan trọng để đảm bảo đầu vào cho mô hình khớp với quá trình huấn luyện
def advanced_preprocess_text(text):
    """Hàm tiền xử lý văn bản nâng cao"""
    if pd.isna(text):
        return ""
    
    text = str(text)
    
    # Xử lý URL, email, số điện thoại, tiền
    text = re.sub(r'http\S+|www\S+|https\S+', ' URL ', text, flags=re.MULTILINE)
    text = re.sub(r'\S+@\S+', ' EMAIL ', text)
    text = re.sub(r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b', ' PHONE ', text)
    text = re.sub(r'\$\d+|\d+\$', ' MONEY ', text)
    
    # Xử lý từ lặp và chuyển về chữ thường
    text = re.sub(r'(.)\1{2,}', r'\1\1', text)
    text = text.lower()
    
    # Giữ lại một số ký tự đặc biệt quan trọng
    text = re.sub(r'[^\w\s!?$%]', ' ', text)
    text = re.sub(r'\b\d+\b', ' NUM ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    
    # Tokenization và xử lý (không áp dụng stop words và stemming để giữ độ dài và ngữ cảnh)
    # Vì bạn đã gặp vấn đề với mô hình, chúng ta sẽ giữ lại toàn bộ văn bản sau khi làm sạch cơ bản
    # Điều này sẽ giúp mô hình có thêm thông tin
    return text

def extract_manual_features(text):
    """Trích xuất các features thủ công"""
    if pd.isna(text):
        text = ""
    text = str(text)
    
    return {
        'length': len(text),
        'word_count': len(text.split()),
        'num_caps': sum(c.isupper() for c in text),
        'num_exclamations': text.count('!'),
        'num_questions': text.count('?'),
        'num_dollars': text.count('$'),
        'caps_ratio': sum(c.isupper() for c in text) / len(text) if len(text) > 0 else 0,
        'has_url': 1 if any(url in text.lower() for url in ['http', 'www', '.com']) else 0,
        'has_phone': 1 if re.search(r'\d{3}[-.]?\d{3}[-.]?\d{4}', text) else 0,
        'has_money': 1 if '$' in text else 0
    }

# Trang chủ của ứng dụng
@app.route('/')
def home():
    return render_template('index.html')

# Xử lý yêu cầu phân loại
@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        message = request.form['message']
        if not message:
            return render_template('index.html', prediction_text="Vui lòng nhập tin nhắn để phân loại.")
        
        # 1. Tiền xử lý văn bản
        processed_message = advanced_preprocess_text(message)
        
        # 2. Vector hóa tin nhắn (từ BoW hoặc TF-IDF)
        # Bước này tạo ra 5000 đặc trưng từ văn bản
        message_vectorized = vectorizer.transform([processed_message])
        
        # 3. Trích xuất các đặc trưng thủ công
        # Bước này tạo ra 10 đặc trưng bổ sung
        manual_features_dict = extract_manual_features(message)
        manual_features_df = pd.DataFrame([manual_features_dict])
        
        # 4. Kết hợp hai loại đặc trưng
        # Đây là bước quan trọng nhất để giải quyết lỗi không khớp đặc trưng
        # Tổng số đặc trưng đầu vào sẽ là 5000 + 10 = 5010
        combined_features = hstack([message_vectorized, csr_matrix(manual_features_df)])
        
        # 5. Dự đoán
        prediction = model.predict(combined_features)
        
        # 6. Trả về kết quả
        if prediction[0] == 1:
            result = "Đây là tin nhắn Spam"
        else:
            result = "Đây là tin nhắn Không phải Spam (Ham)"
            
        return render_template('index.html', prediction_text=result)

if __name__ == '__main__':
    app.run(debug=True)