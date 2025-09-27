from flask import Flask, render_template, request
from flask_socketio import SocketIO, emit
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB, GaussianNB
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sentence_transformers import SentenceTransformer
import matplotlib.pyplot as plt
import seaborn as sns
import io
import base64
import json
import re
import nltk
from underthesea import word_tokenize
import os
from datetime import datetime
import threading
import time
import pickle
from collections import Counter

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

app = Flask(__name__)
app.config['SECRET_KEY'] = 'spam_classifier_secret_key'
socketio = SocketIO(app, cors_allowed_origins="*", async_mode='threading')

# Global variables to store model and data
model_data = {
    'vectorizers': {},
    'models': {},
    'results': {},
    'processed_data': None,
    'original_data': None,
    'training_times': {},
    'start_time': None,
    'total_time': 0
}

class TextPreprocessor:
    def __init__(self):
        self.english_stopwords = set(nltk.corpus.stopwords.words('english'))
        self.vietnamese_stopwords = {
            'và', 'của', 'có', 'là', 'được', 'một', 'trong', 'cho', 'với', 'từ', 
            'này', 'đó', 'những', 'các', 'để', 'khi', 'sẽ', 'đã', 'bị', 'về',
            'tôi', 'bạn', 'anh', 'chị', 'em', 'chúng', 'họ', 'nó', 'mình'
        }
    
    def detect_language(self, text):
        """Simple language detection based on character patterns"""
        if not text or pd.isna(text):
            return 'en'
        vietnamese_chars = re.findall(r'[àáạảãâầấậẩẫăằắặẳẵèéẹẻẽêềếệểễìíịỉĩòóọỏõôồốộổỗơờớợởỡùúụủũưừứựửữỳýỵỷỹđ]', text.lower())
        return 'vi' if len(vietnamese_chars) > 0 else 'en'
    
    def clean_text(self, text):
        """Clean and normalize text"""
        if pd.isna(text):
            return ""
        
        text = str(text).lower()
        text = re.sub(r'\s+', ' ', text).strip()
        
        if len(text) < 2:
            return ""
            
        text = re.sub(r'[^a-zA-Zàáạảãâầấậẩẫăằắặẳẵèéẹẻẽêềếệểễìíịỉĩòóọỏõôồốộổỗơờớợởỡùúụủũưừứựửữỳýỵỷỹđ\s]', ' ', text)
        text = re.sub(r'\s+', ' ', text).strip()
        return text
    
    def tokenize_text(self, text, language='auto'):
        """Tokenize text based on language"""
        if not text or len(text.strip()) < 2:
            return ""
            
        if language == 'auto':
            language = self.detect_language(text)
        
        try:
            if language == 'vi':
                try:
                    tokens = word_tokenize(text)
                except Exception as e:
                    print(f"Underthesea error: {e}")
                    tokens = text.split()
                tokens = [token for token in tokens if token not in self.vietnamese_stopwords and len(token) > 1]
            else:
                try:
                    tokens = nltk.word_tokenize(text)
                except Exception as e:
                    print(f"NLTK tokenize error: {e}")
                    tokens = text.split()
                tokens = [token for token in tokens if token not in self.english_stopwords and len(token) > 1]
        except Exception as e:
            print(f"Tokenization error: {e}")
            tokens = [word for word in text.split() if len(word) > 1]
        
        return ' '.join(tokens)
    
    def preprocess(self, texts, socket_callback=None):
        """Complete preprocessing pipeline with real-time updates"""
        processed_texts = []
        total = len(texts)
        
        for i, text in enumerate(texts):
            try:
                cleaned = self.clean_text(text)
                if not cleaned:
                    processed_texts.append("")
                    continue
                    
                tokenized = self.tokenize_text(cleaned)
                processed_texts.append(tokenized if tokenized else cleaned)
                
                # Real-time progress update
                if socket_callback and (i + 1) % 50 == 0:
                    progress = (i + 1) / total * 100
                    socket_callback('preprocessing_progress', {
                        'progress': progress,
                        'current': i + 1,
                        'total': total,
                        'message': f'Đã xử lý {i + 1}/{total} tin nhắn'
                    })
                    
            except Exception as e:
                print(f"Error processing text {i}: {e}")
                processed_texts.append(str(text) if text else "")
                
        return processed_texts

class SpamClassifier:
    def __init__(self):
        self.preprocessor = TextPreprocessor()
        self.sentence_model = None
    
    def load_sentence_model(self):
        """Load sentence transformer model"""
        try:
            if self.sentence_model is None:
                print("Loading sentence transformer model...")
                # Try different models in order of preference
                models_to_try = [
                    'all-MiniLM-L6-v2',
                    'paraphrase-MiniLM-L6-v2', 
                    'all-mpnet-base-v2'
                ]
                
                for model_name in models_to_try:
                    try:
                        print(f"Trying to load model: {model_name}")
                        self.sentence_model = SentenceTransformer(model_name)
                        print(f"Successfully loaded: {model_name}")
                        break
                    except Exception as e:
                        print(f"Failed to load {model_name}: {e}")
                        continue
                
                if self.sentence_model is None:
                    raise Exception("Could not load any sentence transformer model")
                    
            return self.sentence_model
        except Exception as e:
            print(f"Error loading sentence model: {e}")
            raise e
    
    def create_bow_features(self, texts, max_features=5000):
        """Create Bag of Words features"""
        vectorizer = CountVectorizer(max_features=max_features, ngram_range=(1, 2))
        features = vectorizer.fit_transform(texts)
        return features, vectorizer
    
    def create_tfidf_features(self, texts, max_features=5000):
        """Create TF-IDF features"""
        vectorizer = TfidfVectorizer(max_features=max_features, ngram_range=(1, 2))
        features = vectorizer.fit_transform(texts)
        return features, vectorizer
    
    def create_sentence_embeddings(self, texts):
        """Create sentence embeddings"""
        try:
            model = self.load_sentence_model()
            print(f"Creating embeddings for {len(texts)} texts...")
            
            # Process in batches to avoid memory issues
            batch_size = 32
            all_embeddings = []
            
            for i in range(0, len(texts), batch_size):
                batch = texts[i:i + batch_size]
                print(f"Processing batch {i//batch_size + 1}/{(len(texts) + batch_size - 1)//batch_size}")
                
                try:
                    batch_embeddings = model.encode(batch, show_progress_bar=False)
                    all_embeddings.extend(batch_embeddings)
                except Exception as e:
                    print(f"Error in batch {i//batch_size + 1}: {e}")
                    # Create dummy embeddings for failed batch
                    dummy_embeddings = np.zeros((len(batch), 384))  # 384 is typical embedding size
                    all_embeddings.extend(dummy_embeddings)
            
            embeddings = np.array(all_embeddings)
            print(f"Created embeddings shape: {embeddings.shape}")
            return embeddings, model
            
        except Exception as e:
            print(f"Error creating sentence embeddings: {e}")
            raise e
    
    def train_and_evaluate(self, X, y, method_name, socket_callback=None):
        """Train Naive Bayes model and evaluate with real-time updates"""
        start_time = time.time()
        
        if socket_callback:
            socket_callback('training_progress', {
                'method': method_name,
                'status': 'splitting_data',
                'message': f'Chia dữ liệu cho {method_name}...'
            })
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
        
        if socket_callback:
            socket_callback('training_progress', {
                'method': method_name,
                'status': 'training',
                'message': f'Huấn luyện mô hình {method_name}...'
            })
        
        # Choose appropriate Naive Bayes based on data type
        if method_name == 'Sentence Embeddings':
            # GaussianNB for continuous features (can handle negative values)
            model = GaussianNB()
        else:
            # MultinomialNB for discrete features (BoW, TF-IDF)
            model = MultinomialNB()
            
        model.fit(X_train, y_train)
        
        if socket_callback:
            socket_callback('training_progress', {
                'method': method_name,
                'status': 'evaluating',
                'message': f'Đánh giá mô hình {method_name}...'
            })
        
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        report = classification_report(y_test, y_pred, output_dict=True)
        cm = confusion_matrix(y_test, y_pred)
        
        training_time = time.time() - start_time
        
        return {
            'model': model,
            'accuracy': accuracy,
            'classification_report': report,
            'confusion_matrix': cm,
            'y_test': y_test,
            'y_pred': y_pred,
            'training_time': training_time,
            'X_train': X_train,
            'y_train': y_train
        }

classifier = SpamClassifier()

@app.route('/')
def index():
    return render_template('index.html')

@socketio.on('connect')
def handle_connect():
    print('Client connected')
    emit('connected', {'message': 'Kết nối thành công!'})

@socketio.on('disconnect')
def handle_disconnect():
    print('Client disconnected')

@socketio.on('upload_file')
def handle_file_upload(data):
    """Handle file upload via Socket.IO"""
    try:
        # Decode base64 file content
        file_content = data['file_content']
        filename = data['filename']
        
        if not filename.lower().endswith('.csv'):
            emit('upload_error', {'message': 'Chỉ chấp nhận file CSV'})
            return
        
        # Decode base64
        import base64
        file_data = base64.b64decode(file_content.split(',')[1])
        
        # Try different encodings
        content_str = None
        for encoding in ['utf-8', 'latin-1', 'cp1252', 'iso-8859-1']:
            try:
                content_str = file_data.decode(encoding)
                break
            except UnicodeDecodeError:
                continue
        
        if content_str is None:
            emit('upload_error', {'message': 'Không thể đọc file. Vui lòng kiểm tra encoding.'})
            return
        
        # Parse CSV with different separators
        df = None
        for sep in [',', '\t', ';', '|']:
            try:
                df_temp = pd.read_csv(io.StringIO(content_str), sep=sep)
                if len(df_temp.columns) >= 2:
                    df = df_temp
                    print(f"Successfully parsed with separator: '{sep}'")
                    break
            except Exception as e:
                continue
        
        if df is None or df.empty:
            emit('upload_error', {'message': 'Không thể phân tích file CSV'})
            return
        
        # Smart column detection
        def detect_label_column(df):
            for col in df.columns:
                if df[col].dtype == 'object':
                    unique_vals = df[col].str.lower().str.strip().unique()
                    label_indicators = {'ham', 'spam', '0', '1', 'normal', 'legitimate'}
                    if any(val in label_indicators for val in unique_vals if pd.notna(val)):
                        return col
            return None
        
        def detect_message_column(df, label_col):
            text_cols = [col for col in df.columns if col != label_col and df[col].dtype == 'object']
            if not text_cols:
                return None
            return max(text_cols, key=lambda col: df[col].str.len().mean())
        
        # Handle different CSV formats
        print(f"Original columns: {list(df.columns)}")
        print(f"Sample data:\n{df.head()}")
        
        if 'v1' in df.columns and 'v2' in df.columns:
            # Standard SMS spam dataset format
            df = df[['v1', 'v2']].copy()
            df.columns = ['label', 'message']
        elif 'label' in df.columns and 'message' in df.columns:
            # Already correct format
            df = df[['label', 'message']].copy()
        elif 'sms' in df.columns and 'label' in df.columns:
            # Kaggle format: sms, label (need to swap)
            df = df[['label', 'sms']].copy()
            df.columns = ['label', 'message']
        else:
            # Smart detection for any 2-column format
            if len(df.columns) < 2:
                emit('upload_error', {'message': 'File CSV cần ít nhất 2 cột'})
                return
            
            # Try to detect which column contains labels
            label_col = None
            message_col = None
            
            for i, col in enumerate(df.columns[:2]):
                if df[col].dtype == 'object':
                    # Check if this column contains typical labels
                    sample_values = df[col].dropna().astype(str).str.lower().str.strip()
                    label_keywords = {'ham', 'spam', '0', '1', 'normal', 'legitimate'}
                    
                    # Count how many values look like labels
                    label_matches = sum(1 for val in sample_values.unique()[:10] 
                                      if any(keyword in val for keyword in label_keywords))
                    
                    # If more than 50% of unique values look like labels
                    if label_matches > 0 and label_matches >= len(sample_values.unique()[:10]) * 0.3:
                        label_col = col
                        # The other column should be messages
                        other_cols = [c for c in df.columns[:2] if c != col]
                        if other_cols:
                            message_col = other_cols[0]
                        break
            
            if label_col and message_col:
                print(f"Detected: label_col='{label_col}', message_col='{message_col}'")
                df = df[[label_col, message_col]].copy()
                df.columns = ['label', 'message']
            else:
                # Fallback: assume first 2 columns, first is label, second is message
                print("Using fallback: first column as label, second as message")
                df = df.iloc[:, :2].copy()
                df.columns = ['label', 'message']
        
        print(f"After column detection:")
        print(f"Columns: {list(df.columns)}")
        print(f"Sample labels: {df['label'].unique()[:5]}")
        print(f"Sample message: {df['message'].iloc[0][:100] if len(df) > 0 else 'No data'}")
        
        # Clean and validate data
        df = df.dropna()
        if df.empty:
            emit('upload_error', {'message': 'Không có dữ liệu hợp lệ'})
            return
        
        # Map labels
        original_labels = df['label'].unique()
        print(f"Original labels found: {original_labels}")
        
        label_mapping = {}
        
        for label in original_labels:
            label_str = str(label).lower().strip()
            print(f"Processing label: '{label}' -> '{label_str}'")
            
            if label_str in ['ham', '0'] or label == 0:
                label_mapping[label] = 0
            elif label_str in ['spam', '1'] or label == 1:
                label_mapping[label] = 1
            else:
                # More flexible matching
                if 'ham' in label_str or 'normal' in label_str or 'legitimate' in label_str:
                    label_mapping[label] = 0
                elif 'spam' in label_str:
                    label_mapping[label] = 1
                else:
                    # If we can't identify the label, show more helpful error
                    emit('upload_error', {
                        'message': f'Nhãn không xác định: "{label}". Chỉ chấp nhận: ham, spam, 0, 1, normal, legitimate'
                    })
                    return
        
        print(f"Label mapping: {label_mapping}")
        df['label'] = df['label'].map(label_mapping)
        df = df.dropna()
        
        if df.empty:
            emit('upload_error', {'message': 'Không có nhãn hợp lệ'})
            return
        
        # Store data
        model_data['original_data'] = df
        
        # Calculate stats
        total_messages = len(df)
        spam_count = int(df['label'].sum())
        ham_count = int(total_messages - spam_count)
        spam_percentage = (spam_count / total_messages * 100) if total_messages > 0 else 0.0
        
        stats = {
            'total_messages': total_messages,
            'spam_count': spam_count,
            'ham_count': ham_count,
            'spam_percentage': round(spam_percentage, 2)
        }
        
        emit('upload_success', {
            'message': 'Tải file thành công!',
            'filename': filename,
            'stats': stats,
            'sample_data': df.head().fillna('').to_dict('records')
        })
        
    except Exception as e:
        print(f"Upload error: {str(e)}")
        emit('upload_error', {'message': f'Lỗi tải file: {str(e)}'})

@socketio.on('process_data')
def handle_process_data():
    """Process data and train models with real-time updates"""
    try:
        if model_data['original_data'] is None:
            emit('process_error', {'message': 'Chưa có dữ liệu. Vui lòng tải file trước.'})
            return
        
        model_data['start_time'] = time.time()
        df = model_data['original_data'].copy()
        
        # Step 1: Preprocessing
        emit('step_update', {'step': 1, 'status': 'active', 'message': 'Bắt đầu tiền xử lý văn bản...'})
        
        def progress_callback(event, data):
            socketio.emit(event, data)
        
        processed_texts = classifier.preprocessor.preprocess(
            df['message'].tolist(), 
            socket_callback=progress_callback
        )
        
        df['processed_message'] = processed_texts
        df = df[df['processed_message'].str.len() > 0]
        
        if df.empty:
            emit('process_error', {'message': 'Không có văn bản hợp lệ sau tiền xử lý'})
            return
        
        model_data['processed_data'] = df
        emit('step_update', {'step': 1, 'status': 'completed', 'message': 'Tiền xử lý hoàn thành!'})
        
        # Step 2: Vectorization
        emit('step_update', {'step': 2, 'status': 'active', 'message': 'Bắt đầu vector hóa...'})
        
        processed_texts = df['processed_message'].tolist()
        results = {}
        
        # Train models
        emit('step_update', {'step': 3, 'status': 'active', 'message': 'Bắt đầu huấn luyện mô hình...'})
        
        # Bag of Words
        try:
            emit('training_progress', {'method': 'Bag of Words', 'status': 'vectorizing', 'message': 'Tạo Bag of Words features...'})
            bow_features, bow_vectorizer = classifier.create_bow_features(processed_texts)
            bow_results = classifier.train_and_evaluate(bow_features, df['label'], 'Bag of Words', progress_callback)
            
            model_data['vectorizers']['bow'] = bow_vectorizer
            model_data['models']['bow'] = bow_results['model']
            model_data['training_times']['bow'] = bow_results['training_time']
            results['bow'] = {
                'name': 'Bag of Words',
                'accuracy': float(bow_results['accuracy']),
                'precision': float(bow_results['classification_report']['weighted avg']['precision']),
                'recall': float(bow_results['classification_report']['weighted avg']['recall']),
                'f1_score': float(bow_results['classification_report']['weighted avg']['f1-score']),
                'training_time': bow_results['training_time']
            }
            emit('model_completed', {'method': 'bow', 'results': results['bow']})
        except Exception as e:
            print(f"BoW training failed: {e}")
            emit('model_error', {'method': 'bow', 'error': str(e)})
        
        # TF-IDF
        try:
            emit('training_progress', {'method': 'TF-IDF', 'status': 'vectorizing', 'message': 'Tạo TF-IDF features...'})
            tfidf_features, tfidf_vectorizer = classifier.create_tfidf_features(processed_texts)
            tfidf_results = classifier.train_and_evaluate(tfidf_features, df['label'], 'TF-IDF', progress_callback)
            
            model_data['vectorizers']['tfidf'] = tfidf_vectorizer
            model_data['models']['tfidf'] = tfidf_results['model']
            model_data['training_times']['tfidf'] = tfidf_results['training_time']
            results['tfidf'] = {
                'name': 'TF-IDF',
                'accuracy': float(tfidf_results['accuracy']),
                'precision': float(tfidf_results['classification_report']['weighted avg']['precision']),
                'recall': float(tfidf_results['classification_report']['weighted avg']['recall']),
                'f1_score': float(tfidf_results['classification_report']['weighted avg']['f1-score']),
                'training_time': tfidf_results['training_time']
            }
            emit('model_completed', {'method': 'tfidf', 'results': results['tfidf']})
        except Exception as e:
            print(f"TF-IDF training failed: {e}")
            emit('model_error', {'method': 'tfidf', 'error': str(e)})
        
        # Sentence Embeddings
        try:
            emit('training_progress', {'method': 'Sentence Embeddings', 'status': 'vectorizing', 'message': 'Tạo Sentence Embeddings...'})
            
            # Check if sentence-transformers is available
            try:
                import sentence_transformers
                sentence_embeddings, sentence_model = classifier.create_sentence_embeddings(processed_texts)
                sentence_results = classifier.train_and_evaluate(sentence_embeddings, df['label'], 'Sentence Embeddings', progress_callback)
                
                model_data['vectorizers']['sentence'] = sentence_model
                model_data['models']['sentence'] = sentence_results['model']
                model_data['training_times']['sentence'] = sentence_results['training_time']
                results['sentence'] = {
                    'name': 'Sentence Embeddings',
                    'accuracy': float(sentence_results['accuracy']),
                    'precision': float(sentence_results['classification_report']['weighted avg']['precision']),
                    'recall': float(sentence_results['classification_report']['weighted avg']['recall']),
                    'f1_score': float(sentence_results['classification_report']['weighted avg']['f1-score']),
                    'training_time': sentence_results['training_time']
                }
                emit('model_completed', {'method': 'sentence', 'results': results['sentence']})
                
            except ImportError:
                emit('model_error', {'method': 'sentence', 'error': 'sentence-transformers not installed'})
                print("sentence-transformers not available, skipping...")
                
        except Exception as e:
            print(f"Sentence Embeddings training failed: {e}")
            import traceback
            traceback.print_exc()
            emit('model_error', {'method': 'sentence', 'error': str(e)})
        
        # Store results
        model_data['results'] = {}
        if 'bow' in results:
            model_data['results']['bow'] = bow_results
        if 'tfidf' in results:
            model_data['results']['tfidf'] = tfidf_results
        if 'sentence' in results:
            model_data['results']['sentence'] = sentence_results
        
        model_data['total_time'] = time.time() - model_data['start_time']
        
        emit('step_update', {'step': 2, 'status': 'completed', 'message': 'Vector hóa hoàn thành!'})
        emit('step_update', {'step': 3, 'status': 'completed', 'message': 'Huấn luyện mô hình hoàn thành!'})
        
        # Generate charts and summary
        if results:
            best_method = max(results.keys(), key=lambda x: results[x]['accuracy'])
            
            # Generate and emit competition files
            competition_data = generate_competition_files(best_method)
            if competition_data:
                socketio.emit('test_files_ready', competition_data)

            confusion_matrix_img = generate_confusion_matrix(best_method)
            comparison_chart = generate_comparison_chart(results)
            feature_analysis = generate_feature_analysis()
            summary = generate_summary(results, model_data['total_time'])
            
            emit('process_completed', {
                'message': 'Xử lý hoàn thành!',
                'results': results,
                'best_method': best_method,
                'confusion_matrix': confusion_matrix_img,
                'comparison_chart': comparison_chart,
                'feature_analysis': feature_analysis,
                'summary': summary
            })
        else:
            emit('process_error', {'message': 'Không thể huấn luyện mô hình nào'})
        
    except Exception as e:
        print(f"Processing error: {str(e)}")
        import traceback
        traceback.print_exc()
        emit('process_error', {'message': f'Lỗi xử lý: {str(e)}'})

def generate_competition_files(best_method):
    """Generate test.csv and submission.csv files for competition"""
    try:
        if model_data['processed_data'] is None:
            return None
        
        # Read the actual test file that was provided
        test_df = pd.read_csv('test (6).csv')
        print(f"Test file loaded with {len(test_df)} rows")
        print(f"Test file columns: {test_df.columns.tolist()}")
        print(f"Sample test data:\n{test_df.head()}")
        
        # The test.csv already exists with correct format (id, sms)
        # We just need to create predictions for it
        test_messages = test_df['sms'].tolist()
        
        # Generate predictions for submission.csv using best model
        print(f"Processing {len(test_messages)} messages for prediction...")
        processed_texts = classifier.preprocessor.preprocess(test_messages)
        
        print(f"Processed {len(processed_texts)} texts")
        print(f"Sample processed text: {processed_texts[0][:100] if processed_texts else 'None'}")
        
        if best_method == 'bow':
            features = model_data['vectorizers']['bow'].transform(processed_texts)
        elif best_method == 'tfidf':
            features = model_data['vectorizers']['tfidf'].transform(processed_texts)
        elif best_method == 'sentence':
            features = model_data['vectorizers']['sentence'].encode(processed_texts)
        else:
            raise ValueError(f"Unknown method: {best_method}")
        
        model = model_data['models'][best_method]
        predictions = model.predict(features)
        
        print(f"Generated {len(predictions)} predictions")
        print(f"Prediction distribution: {np.bincount(predictions)}")
        
        # Create submission.csv with same ids as test.csv
        submission_csv = pd.DataFrame({
            'id': test_df['id'].values,  # Use the same IDs from test.csv
            'label': predictions
        })
        
        print(f"Submission CSV shape: {submission_csv.shape}")
        print(f"Submission sample:\n{submission_csv.head()}")
        
        # Convert to CSV strings
        test_csv_str = test_df.to_csv(index=False)  # Use original test data
        submission_csv_str = submission_csv.to_csv(index=False)
        
        # Calculate statistics
        spam_count = int(np.sum(predictions))
        ham_count = len(predictions) - spam_count
        spam_percentage = (spam_count / len(predictions)) * 100
        
        print(f"Final stats - Total: {len(predictions)}, Spam: {spam_count}, Ham: {ham_count}, Spam%: {spam_percentage:.2f}")
        
        return {
            'test_csv': test_csv_str,
            'submission_csv': submission_csv_str,
            'test_filename': f'test_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv',
            'submission_filename': f'submission_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv',
            'count': len(predictions),
            'best_method': best_method,
            'stats': {
                'total': len(predictions),
                'spam_count': spam_count,
                'ham_count': ham_count,
                'spam_percentage': round(spam_percentage, 2)
            }
        }
        
    except Exception as e:
        print(f"Error generating competition files: {e}")
        import traceback
        traceback.print_exc()
        return None

def generate_confusion_matrix(method):
    """Generate confusion matrix as base64 image"""
    try:
        if method not in model_data['results']:
            return None
        
        results = model_data['results'][method]
        cm = results['confusion_matrix']
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=['Ham', 'Spam'], 
                   yticklabels=['Ham', 'Spam'])
        plt.title(f'Ma trận Nhầm lẫn - {method.upper()}')
        plt.ylabel('Nhãn Thực tế')
        plt.xlabel('Nhãn Dự đoán')
        
        # Convert to base64
        img_buffer = io.BytesIO()
        plt.savefig(img_buffer, format='png', bbox_inches='tight', dpi=150)
        img_buffer.seek(0)
        img_base64 = base64.b64encode(img_buffer.getvalue()).decode()
        plt.close()
        
        return f"data:image/png;base64,{img_base64}"
        
    except Exception as e:
        print(f"Error generating confusion matrix: {e}")
        return None

def generate_comparison_chart(results):
    """Generate model comparison chart"""
    try:
        methods = list(results.keys())
        metrics = ['accuracy', 'precision', 'recall', 'f1_score']
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
        axes = [ax1, ax2, ax3, ax4]
        
        colors = ['#667eea', '#764ba2', '#48bb78']
        
        for i, metric in enumerate(metrics):
            values = [results[method][metric] * 100 for method in methods]
            method_names = [results[method]['name'] for method in methods]
            
            bars = axes[i].bar(method_names, values, color=colors[:len(methods)])
            axes[i].set_title(f'{metric.replace("_", " ").title()} (%)', fontsize=14, fontweight='bold')
            axes[i].set_ylim(0, 100)
            axes[i].grid(True, alpha=0.3)
            
            # Add value labels on bars
            for bar, value in zip(bars, values):
                axes[i].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                           f'{value:.2f}%', ha='center', va='bottom', fontweight='bold')
        
        plt.tight_layout()
        
        # Convert to base64
        img_buffer = io.BytesIO()
        plt.savefig(img_buffer, format='png', bbox_inches='tight', dpi=150)
        img_buffer.seek(0)
        img_base64 = base64.b64encode(img_buffer.getvalue()).decode()
        plt.close()
        
        return f"data:image/png;base64,{img_base64}"
        
    except Exception as e:
        print(f"Error generating comparison chart: {e}")
        return None

def generate_feature_analysis():
    """Generate feature importance analysis for BoW"""
    try:
        if 'bow' not in model_data['results'] or 'bow' not in model_data['vectorizers']:
            return None
            
        model = model_data['models']['bow']
        vectorizer = model_data['vectorizers']['bow']
        
        # Get feature names and importance
        feature_names = vectorizer.get_feature_names_out()
        
        # For MultinomialNB, feature_log_prob_ gives log probabilities
        # We want features that are most indicative of spam (class 1)
        spam_features = model.feature_log_prob_[1]  # Class 1 (spam)
        ham_features = model.feature_log_prob_[0]   # Class 0 (ham)
        
        # Calculate feature importance as difference
        feature_importance = spam_features - ham_features
        
        # Get top 15 features for spam
        top_indices = np.argsort(feature_importance)[-15:]
        top_features = [feature_names[i] for i in top_indices]
        top_scores = [feature_importance[i] for i in top_indices]
        
        # Create plot
        plt.figure(figsize=(12, 8))
        y_pos = np.arange(len(top_features))
        
        bars = plt.barh(y_pos, top_scores, color='#e53e3e', alpha=0.7)
        plt.yticks(y_pos, top_features)
        plt.xlabel('Mức độ quan trọng cho SPAM detection')
        plt.title('Top 15 từ quan trọng cho SPAM detection - BoW', fontsize=16, fontweight='bold')
        plt.grid(True, alpha=0.3)
        
        # Add value labels
        for i, (bar, score) in enumerate(zip(bars, top_scores)):
            plt.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height()/2,
                    f'{score:.3f}', va='center', fontweight='bold')
        
        plt.tight_layout()
        
        # Convert to base64
        img_buffer = io.BytesIO()
        plt.savefig(img_buffer, format='png', bbox_inches='tight', dpi=150)
        img_buffer.seek(0)
        img_base64 = base64.b64encode(img_buffer.getvalue()).decode()
        plt.close()
        
        return f"data:image/png;base64,{img_base64}"
        
    except Exception as e:
        print(f"Error generating feature analysis: {e}")
        return None

def generate_summary(results, total_time):
    """Generate processing summary"""
    try:
        if not results:
            return None
            
        # Find best model
        best_method = max(results.keys(), key=lambda x: results[x]['f1_score'])
        best_result = results[best_method]
        
        # Performance assessment
        f1_score = best_result['f1_score']
        if f1_score >= 0.95:
            performance = "Xuất sắc! 🏆"
        elif f1_score >= 0.90:
            performance = "Rất tốt! 🎯"
        elif f1_score >= 0.85:
            performance = "Tốt! 👍"
        else:
            performance = "Cần cải thiện 📈"
        
        summary = {
            'best_model': best_result['name'],
            'best_f1': f1_score,
            'total_models': len(results),
            'total_time': total_time,
            'performance': performance,
            'training_times': model_data['training_times']
        }
        
        return summary
        
    except Exception as e:
        print(f"Error generating summary: {e}")
        return None

@socketio.on('get_confusion_matrix')
def handle_get_confusion_matrix(data):
    """Get confusion matrix for specific method"""
    try:
        method = data['method']
        confusion_matrix_img = generate_confusion_matrix(method)
        
        if confusion_matrix_img:
            emit('confusion_matrix_result', {
                'method': method,
                'image': confusion_matrix_img
            })
        else:
            emit('confusion_matrix_error', {
                'method': method,
                'message': 'Không thể tạo ma trận nhầm lẫn'
            })
            
    except Exception as e:
        emit('confusion_matrix_error', {
            'method': data.get('method', 'unknown'),
            'message': f'Lỗi: {str(e)}'
        })

@socketio.on('download_results')
def handle_download_results():
    """Prepare download data"""
    try:
        if model_data['processed_data'] is None:
            emit('download_error', {'message': 'Chưa có dữ liệu đã xử lý'})
            return
            
        # Create download package
        download_data = {
            'processed_data': model_data['processed_data'].to_csv(index=False),
            'results_summary': model_data.get('results', {}),
            'training_times': model_data.get('training_times', {}),
            'total_time': model_data.get('total_time', 0)
        }
        
        emit('download_ready', {
            'data': download_data,
            'filename': f'spam_classification_results_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
        })
        
    except Exception as e:
        emit('download_error', {'message': f'Lỗi chuẩn bị tải xuống: {str(e)}'})

@socketio.on('download_model')
def handle_download_model(data):
    """Prepare model download"""
    try:
        method = data['method']
        
        if method not in model_data['models'] or method not in model_data['vectorizers']:
            emit('download_error', {'message': f'Mô hình {method} không tồn tại'})
            return
            
        # Serialize model and vectorizer
        model_package = {
            'model': pickle.dumps(model_data['models'][method]).hex(),
            'vectorizer': pickle.dumps(model_data['vectorizers'][method]).hex(),
            'method': method,
            'results': model_data['results'][method] if method in model_data['results'] else None,
            'created_at': datetime.now().isoformat()
        }
        
        emit('model_download_ready', {
            'data': model_package,
            'filename': f'spam_model_{method}_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
        })
        
    except Exception as e:
        emit('download_error', {'message': f'Lỗi chuẩn bị tải mô hình: {str(e)}'})
if __name__ == '__main__':
    socketio.run(app, host='0.0.0.0', port=8000, debug=True, allow_unsafe_werkzeug=True)