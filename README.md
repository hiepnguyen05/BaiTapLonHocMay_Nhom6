
<!DOCTYPE html>
<html lang="vi">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width,initial-scale=1" />
  <title>Phân loại tin nhắn rác — Tài liệu dự án</title>
  <style>
    /* Kiểu chữ & layout cơ bản */
    :root{
      --bg:#f7f8fa;
      --card:#ffffff;
      --text:#222;
      --muted:#666;
      --accent:#0b5fff;
      --code-bg:#f4f6f8;
      --box-shadow: 0 8px 24px rgba(12, 24, 40, 0.06);
    }
    html,body{height:100%;margin:0;padding:0;background:var(--bg);color:var(--text);font-family:Inter, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif;}
    .wrapper{max-width:980px;margin:36px auto;padding:28px;background:var(--card);border-radius:10px;box-shadow:var(--box-shadow);line-height:1.6;}
    h1{margin:0 0 8px;font-size:28px;font-weight:600}
    h2{margin:24px 0 8px;font-size:20px;font-weight:600}
    h3{margin:16px 0 6px;font-size:16px;font-weight:600}
    p{margin:8px 0;color:var(--muted)}
    ul,ol{margin:8px 0 8px 20px;color:var(--muted)}
    code{background:var(--code-bg);padding:2px 6px;border-radius:6px;font-family:SFMono-Regular, Menlo, Monaco, "Roboto Mono", monospace;font-size:13px}
    pre{background:#0f1724;color:#e6eef8;padding:14px;border-radius:8px;overflow:auto;font-family:SFMono-Regular, Menlo, Monaco, "Roboto Mono", monospace;font-size:13px}
    .meta{display:flex;gap:12px;flex-wrap:wrap;color:var(--muted);font-size:13px;margin-bottom:6px}
    .section-note{background:#f3f6ff;border-left:4px solid #cfe0ff;padding:10px;border-radius:6px;color:#1f2f6f;margin:10px 0}
    .images {display:flex;gap:12px;flex-wrap:wrap;margin:12px 0}
    .images img{max-width:100%;height:auto;border-radius:6px;box-shadow:0 6px 18px rgba(12,24,40,0.06)}
    .img-holder{flex:1 1 48%;min-width:220px}
    .small{font-size:13px;color:var(--muted)}
    .footer{margin-top:22px;padding-top:14px;border-top:1px solid #eef2f6;color:var(--muted);font-size:13px}
    .code-block{margin:12px 0}
    .kbd{display:inline-block;border:1px solid #e6edf3;background:#fbfdff;padding:2px 8px;border-radius:6px;font-family:inherit;font-size:13px}
    @media (max-width:640px){
      .wrapper{margin:12px;padding:18px}
      .images{flex-direction:column}
      .img-holder{flex-basis:100%}
    }
  </style>
</head>
<body>
  <div class="wrapper">

    <h1>Phân loại tin nhắn rác (Spam Message Classification)</h1>
    <div class="meta">
      <div class="small">Tài liệu: mô tả huấn luyện mô hình & triển khai web</div>
      <div class="small">Ngôn ngữ: Python / Flask / scikit-learn</div>
    </div>

    <p>
      Tài liệu này tổng hợp nội dung dự án phân loại tin nhắn văn bản nhằm phát hiện <strong>tin nhắn rác (Spam)</strong> và phân loại
      các tin nhắn hợp lệ (<strong>Ham</strong>). Nội dung gồm: mô tả quy trình huấn luyện, cách lưu và tải mô hình, cấu trúc ứng dụng web và hướng dẫn chạy.
    </p>

    <h2>1. Giới thiệu tổng quan</h2>
    <p>
      Hệ thống sử dụng các kỹ thuật xử lý ngôn ngữ tự nhiên (NLP) cơ bản để chuyển văn bản thành đặc trưng số, kết hợp với mô hình phân loại máy học.
      Mục tiêu là có một mô hình dễ triển khai, có tốc độ dự đoán nhanh và độ chính xác tốt trên bộ dữ liệu tin nhắn SMS.
    </p>

    <h2>2. Phần A — Huấn luyện mô hình</h2>

    <h3>2.1. Quy trình huấn luyện</h3>
    <ol>
      <li><strong>Tiền xử lý văn bản</strong>
        <ul>
          <li>Chuẩn hóa: chuyển toàn bộ về chữ thường.</li>
          <li>Thay thế token có cấu trúc: URL → <code>URL</code>, email → <code>EMAIL</code>, số điện thoại → <code>PHONE</code>, tiền → <code>MONEY</code>.</li>
          <li>Loại bỏ ký tự đặc biệt không cần thiết, xử lý từ lặp (ví dụ: <em>helloooo</em> → <em>helloo</em>).</li>
        </ul>
      </li>
      <li><strong>Trích xuất đặc trưng</strong>
        <ul>
          <li>Vector hóa bằng <strong>Bag-of-Words (CountVectorizer)</strong> hoặc <strong>TF-IDF</strong>.</li>
          <li>Thêm các đặc trưng thủ công (numeric features): độ dài chuỗi, số từ, tỷ lệ chữ hoa, số dấu chấm than, số ký tự tiền tệ, phát hiện URL, phát hiện số điện thoại, ...</li>
        </ul>
      </li>
      <li><strong>Huấn luyện & lựa chọn mô hình</strong>
        <ul>
          <li>Thử nghiệm các thuật toán: Multinomial Naive Bayes, Logistic Regression, SVM.</li>
          <li>Sử dụng cross-validation và GridSearch khi cần để chọn hyper-parameters.</li>
          <li>Mô hình cuối cùng được lựa chọn: <strong>Multinomial Naive Bayes (đã huấn luyện trên BoW)</strong>.</li>
        </ul>
      </li>
    </ol>

    <h3>2.2. Ví dụ lưu mô hình</h3>
    <div class="code-block">
      <pre><code>import joblib
joblib.dump(best_model, 'spam_model.pkl')
joblib.dump(best_vectorizer, 'spam_vectorizer.pkl')</code></pre>
    </div>
    <p class="small">
      Lưu ý: cần lưu cả <code>model</code> và <code>vectorizer</code> cùng lúc để đảm bảo pipeline tiền xử lý & vector hóa khi dự đoán khớp với lúc huấn luyện.
    </p>

    <h3>2.3. Kết quả huấn luyện (minh họa)</h3>
    <p class="small">Chèn các biểu đồ, ma trận nhầm lẫn hoặc bảng kết quả vào hình dưới.</p>
    <div class="images">
      <div class="img-holder">
        <img src="images/training_results.png" alt="Biểu đồ kết quả huấn luyện (ví dụ: Confusion Matrix / ROC / Accuracy)">
      </div>
      <div class="img-holder">
        <img src="images/metrics_table.png" alt="Bảng số liệu - precision/recall/f1/accuracy">
      </div>
    </div>

    <div class="section-note">
      <strong>Tóm tắt kết quả (tham khảo):</strong> Accuracy ~97%, Precision (Spam) ~95%, Recall (Spam) ~94%. Kết quả thay đổi tùy dữ liệu và pipeline xử lý.
    </div>

    <h2>3. Phần B — Ứng dụng Web phân loại tin nhắn</h2>

    <h3>3.1. Mục tiêu</h3>
    <p>
      Xây dựng một ứng dụng web đơn giản cho phép người dùng nhập tin nhắn (text) và nhận nhãn dự đoán: <code>Spam</code> hoặc <code>Ham</code>.
      Ứng dụng sử dụng mô hình đã lưu (file <code>.pkl</code>) để dự đoán realtime.
    </p>

    <h3>3.2. Luồng xử lý khi có yêu cầu phân loại</h3>
    <ol>
      <li>Nhận input text từ form trên giao diện.</li>
      <li>Tiền xử lý (hàm <code>advanced_preprocess_text</code>) để đảm bảo format giống dữ liệu huấn luyện.</li>
      <li>Vector hóa bằng <code>vectorizer.transform([...])</code> → trả về ma trận sparse.</li>
      <li>Trích xuất đặc trưng thủ công thành DataFrame và chuyển sang <code>csr_matrix</code>.</li>
      <li>Kết hợp hai nguồn đặc trưng bằng <code>hstack</code> → đưa vào <code>model.predict()</code>.</li>
      <li>Hiển thị kết quả trên giao diện (Spam / Ham).</li>
    </ol>

    <h3>3.3. Đoạn mã chính (ví dụ triển khai Flask)</h3>
    <div class="code-block">
      <pre><code>import joblib
from flask import Flask, render_template, request
from scipy.sparse import hstack, csr_matrix
import pandas as pd
import re

model = joblib.load('spam_model.pkl')
vectorizer = joblib.load('spam_vectorizer.pkl')

app = Flask(__name__)

# advanced_preprocess_text(...)  # hàm làm sạch giống pipeline huấn luyện
# extract_manual_features(...)  # hàm tạo các đặc trưng numeric

@app.route('/', methods=['GET'])
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    message = request.form.get('message', '')
    processed = advanced_preprocess_text(message)
    vec = vectorizer.transform([processed])
    manual = extract_manual_features(message)
    manual_df = pd.DataFrame([manual])
    combined = hstack([vec, csr_matrix(manual_df)])
    pred = model.predict(combined)
    return render_template('index.html', prediction=('Spam' if pred[0]==1 else 'Ham'))</code></pre>
    </div>

    <h3>3.4. Giao diện & minh họa</h3>
    <p class="small">Chèn ảnh giao diện (screenshot) vào phần dưới.</p>
    <div class="images">
      <div class="img-holder">
        <img src="images/web_demo.png" alt="Screenshot giao diện ứng dụng phân loại spam">
      </div>
      <div class="img-holder">
        <img src="images/web_demo_2.png" alt="Screenshot giao diện - form nhập và kết quả">
      </div>
    </div>

    <h2>4. Cấu trúc repository & cách chạy</h2>

    <h3>4.1. Cấu trúc gợi ý</h3>
    <pre><code>spam-message-classification/
├── data/                    # (tùy chọn) bộ dữ liệu gốc
├── images/                  # ảnh minh họa cho README
│   ├── training_results.png
│   └── web_demo.png
├── models/
│   ├── spam_model.pkl
│   └── spam_vectorizer.pkl
├── app.py                   # Flask app
├── train_model.ipynb        # Notebook / script huấn luyện
├── requirements.txt
└── README.html</code></pre>

    <h3>4.2. Hướng dẫn chạy</h3>
    <ol>
      <li>Clone repository:
        <pre><code>git clone https://github.com/your-username/spam-message-classification.git
cd spam-message-classification</code></pre>
      </li>
      <li>Cài đặt môi trường:
        <pre><code>python -m venv .venv
source .venv/bin/activate   # hoặc .venv\Scripts\activate trên Windows
pip install -r requirements.txt</code></pre>
      </li>
      <li>Chạy Flask:
        <pre><code>python app.py
# Mở http://127.0.0.1:5000/</code></pre>
      </li>
    </ol>

    <h2>5. Đánh giá, hạn chế và hướng phát triển</h2>
    <h3>5.1. Đánh giá</h3>
    <p class="small">Mô hình Naive Bayes với BoW cho kết quả ổn định trên tập SMS nhỏ/ vừa; nhanh và dễ triển khai.</p>

    <h3>5.2. Hạn chế</h3>
    <ul>
      <li>Mô hình được huấn luyện trên SMS (dữ liệu giới hạn) — có thể giảm hiệu năng khi áp dụng cho email hoặc mạng xã hội.</li>
      <li>Không xử lý đa ngôn ngữ/biến thể ngôn ngữ phức tạp (chỉ phù hợp tiếng Anh đơn giản nếu dataset là tiếng Anh).</li>
      <li>Phụ thuộc vào chất lượng và tính đại diện của dữ liệu huấn luyện.</li>
    </ul>

    <h3>5.3. Hướng phát triển</h3>
    <ul>
      <li>Mở rộng dữ liệu (email, comment, chat logs) và huấn luyện lại.</li>
      <li>Sử dụng mô hình ngôn ngữ tiên tiến (BERT, fine-tuning) để cải thiện khả năng hiểu ngữ cảnh.</li>
      <li>Xây dựng API REST để tích hợp dễ dàng vào hệ thống lớn.</li>
      <li>Thêm cơ chế học trực tuyến (online learning) hoặc pipeline retraining định kỳ.</li>
    </ul>

    <div class="footer">
      Tài liệu này là bản tóm tắt kỹ thuật cho dự án phân loại tin nhắn rác. Nếu cần bản chi tiết hơn (script huấn luyện, notebook, hoặc hướng dẫn deploy lên dịch vụ PaaS), vui lòng yêu cầu để mình bổ sung.
    </div>

  </div>
</body>
</html>
