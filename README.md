
<div>
  <h1>ỨNG DỤNG PHÂN LOẠI TIN NHẮN RÁC</h1>

  <h2>Giới thiệu</h2>
  <p>
    Đây là một dự án <strong>Machine Learning</strong> và <strong>Web Application</strong> gồm hai phần chính:
  </p>
  <ul>
    <li><strong>Huấn luyện mô hình</strong>: Xây dựng mô hình phân loại tin nhắn rác (Spam/Ham).</li>
    <li><strong>Ứng dụng Web</strong>: Cho phép người dùng nhập tin nhắn và nhận kết quả phân loại.</li>
  </ul>

  <h2>1. Huấn luyện mô hình</h2>
  <p>Mô hình sử dụng <strong>Naive Bayes</strong> kết hợp với Bag-of-Words và TF-IDF để trích xuất đặc trưng.</p>
  <ol>
    <li>Tiền xử lý dữ liệu: loại bỏ ký tự thừa, chuẩn hóa văn bản.</li>
    <li>Vector hóa văn bản: BoW, TF-IDF.</li>
    <li>Huấn luyện mô hình Naive Bayes trên tập dữ liệu tin nhắn.</li>
    <li>Đánh giá mô hình bằng Accuracy, Precision, Recall, F1-score.</li>
    <li>Lưu mô hình và vectorizer bằng <code>joblib</code>.</li>
  </ol>


  <h2>2. Ứng dụng Web phân loại tin nhắn</h2>
  <p>Ứng dụng được xây dựng bằng <strong>Flask</strong>. Người dùng có thể nhập tin nhắn để phân loại.</p>
  <ul>
    <li>Tiền xử lý đầu vào với cùng pipeline như khi huấn luyện.</li>
    <li>Vector hóa tin nhắn bằng CountVectorizer đã lưu.</li>
    <li>Kết hợp với đặc trưng thủ công (số từ, số ký tự in hoa, ký tự đặc biệt...).</li>
    <li>Dự đoán bằng mô hình Naive Bayes.</li>
    <li>Hiển thị kết quả: Spam hoặc Ham.</li>
  </ul>

  <h3>Giao diện minh họa</h3>
  <img src="https://github.com/hiepnguyen05/BaiTapLonHocMay_Nhom6/blob/main/mainimg.png?raw=true" alt="Giao diện web" width="600">
  <img src="https://github.com/hiepnguyen05/BaiTapLonHocMay_Nhom6/blob/main/ham.png?raw=true" alt="Giao diện web" width="600">
  <img src="https://github.com/hiepnguyen05/BaiTapLonHocMay_Nhom6/blob/main/spam.png?raw=true" alt="Giao diện web" width="600">

  <h2>Cách chạy dự án</h2>
  <ol>
    <li>Clone project: <code>git clone https://github.com/hiepnguyen05/BaiTapLonHocMay_Nhom6</code></li>
    <li>Cài thư viện: <code>pip install -r requirements.txt</code></li>
    <li>Huấn luyện mô hình (nếu cần): chạy notebook huấn luyện.</li>
    <li>Chạy ứng dụng web: <code>python app.py</code></li>
    <li>Mở trình duyệt: <code>http://127.0.0.1:5000/</code></li>
  </ol>

  <h2>Mục tiêu dự án</h2>
  <p>
    Dự án giúp thực hành các bước xây dựng hệ thống học máy từ huấn luyện đến triển khai.
    Qua đó, người học nắm vững cách xử lý dữ liệu văn bản, xây dựng mô hình phân loại,
    và triển khai ứng dụng thực tế bằng Flask.
  </p>
</div>
