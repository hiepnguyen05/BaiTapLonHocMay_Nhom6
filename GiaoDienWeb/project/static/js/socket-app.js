class SocketSpamClassifier {
    constructor() {
        this.socket = io();
        this.initializeElements();
        this.bindSocketEvents();
        this.bindUIEvents();
        this.isProcessed = false;
    }

    initializeElements() {
        this.connectionStatus = document.getElementById('connectionStatus');
        this.fileInput = document.getElementById('fileInput');
        this.uploadArea = document.getElementById('uploadArea');
        this.fileInfo = document.getElementById('fileInfo');
        this.processBtn = document.getElementById('processBtn');
        this.resultsSection = document.getElementById('resultsSection');
        this.resultsGrid = document.getElementById('resultsGrid');
        this.confusionMatrix = document.getElementById('confusionMatrix');
        this.comparisonChart = document.getElementById('comparisonChart');
        this.featureAnalysis = document.getElementById('featureAnalysis');
        this.summarySection = document.getElementById('summarySection');
        this.summaryContent = document.getElementById('summaryContent');
        this.competitionSection = document.getElementById('competitionSection');
        this.competitionResults = document.getElementById('competitionResults');
        this.downloadTestBtn = document.getElementById('downloadTestBtn');
        this.downloadSubmissionBtn = document.getElementById('downloadSubmissionBtn');
        
        this.progressSteps = {
            step1: document.getElementById('step1'),
            step2: document.getElementById('step2'),
            step3: document.getElementById('step3'),
            step4: document.getElementById('step4')
        };
        this.preprocessProgress = document.getElementById('preprocessProgress');
        this.trainingProgress = document.getElementById('trainingProgress');
        this.logsContainer = document.getElementById('logsContainer');
    }

    bindSocketEvents() {
        // Connection events
        this.socket.on('connect', () => {
            this.updateConnectionStatus(true);
            this.addLog('Kết nối thành công với server!', 'success');
        });

        this.socket.on('disconnect', () => {
            this.updateConnectionStatus(false);
            this.addLog('Mất kết nối với server', 'error');
        });

        this.socket.on('connected', (data) => this.addLog(data.message, 'info'));
        this.socket.on('upload_success', (data) => this.handleUploadSuccess(data));
        this.socket.on('upload_error', (data) => this.handleUploadError(data));
        this.socket.on('step_update', (data) => this.updateProgressStep(data.step, data.status, data.message));
        this.socket.on('preprocessing_progress', (data) => this.updatePreprocessingProgress(data));
        this.socket.on('training_progress', (data) => this.updateTrainingProgress(data));
        this.socket.on('model_completed', (data) => this.handleModelCompleted(data));
        this.socket.on('model_error', (data) => this.handleModelError(data));
        this.socket.on('process_completed', (data) => this.handleProcessCompleted(data));
        this.socket.on('process_error', (data) => this.handleProcessError(data));
        this.socket.on('confusion_matrix_result', (data) => this.displayConfusionMatrix(data));
        this.socket.on('confusion_matrix_error', (data) => this.addLog(`❌ Lỗi tạo ma trận: ${data.message}`, 'error'));
        this.socket.on('test_files_ready', (data) => this.handleTestFilesReady(data));
    }

    bindUIEvents() {
        this.fileInput.addEventListener('change', (e) => this.handleFileSelect(e));
        this.uploadArea.addEventListener('click', () => this.fileInput.click());
        this.uploadArea.addEventListener('dragover', (e) => this.handleDragOver(e));
        this.uploadArea.addEventListener('dragleave', (e) => this.handleDragLeave(e));
        this.uploadArea.addEventListener('drop', (e) => this.handleDrop(e));
        this.processBtn.addEventListener('click', () => this.processData());
        this.downloadTestBtn.addEventListener('click', () => this.downloadTestFile());
        this.downloadSubmissionBtn.addEventListener('click', () => this.downloadSubmissionFile());
    }

    updateConnectionStatus(connected) {
        const statusText = this.connectionStatus.querySelector('span');
        if (connected) {
            this.connectionStatus.className = 'connection-status connected';
            statusText.textContent = 'Đã kết nối';
        } else {
            this.connectionStatus.className = 'connection-status disconnected';
            statusText.textContent = 'Mất kết nối';
        }
    }

    addLog(message, type = 'info') {
        const logItem = document.createElement('div');
        logItem.className = `log-item ${type}`;
        const timestamp = new Date().toLocaleTimeString('vi-VN');
        logItem.innerHTML = `<span class="log-time">[${timestamp}]</span> <span class="log-message">${message}</span>`;
        this.logsContainer.appendChild(logItem);
        this.logsContainer.scrollTop = this.logsContainer.scrollHeight;
    }

    handleDragOver(e) {
        e.preventDefault();
        this.uploadArea.classList.add('dragover');
    }

    handleDragLeave(e) {
        e.preventDefault();
        this.uploadArea.classList.remove('dragover');
    }

    handleDrop(e) {
        e.preventDefault();
        this.uploadArea.classList.remove('dragover');
        const files = e.dataTransfer.files;
        if (files.length > 0) {
            this.fileInput.files = files;
            this.handleFileSelect({ target: { files } });
        }
    }

    handleFileSelect(e) {
        const file = e.target.files[0];
        if (!file) return;
        if (!file.name.toLowerCase().endsWith('.csv')) {
            this.addLog('Chỉ chấp nhận file CSV', 'error');
            return;
        }
        this.addLog(`Đang tải file: ${file.name}`, 'info');
        this.uploadArea.classList.add('loading');
        this.uploadArea.innerHTML = `<i class="fas fa-spinner fa-spin"></i><p>Đang tải và xử lý file...</p>`;
        const reader = new FileReader();
        reader.onload = (e) => {
            this.socket.emit('upload_file', {
                filename: file.name,
                file_content: e.target.result
            });
        };
        reader.readAsDataURL(file);
    }

    handleUploadSuccess(data) {
        this.addLog(`Tải file thành công: ${data.filename}`, 'success');
        this.updateProgressStep(1, 'completed', 'Tải file hoàn thành!');
        this.uploadArea.classList.remove('loading');
        this.uploadArea.innerHTML = `
            <i class="fas fa-check-circle" style="color: #48bb78;"></i>
            <p style="color: #48bb78;">Tải file thành công!</p>
            <p style="font-size: 0.9rem; color: #718096;">${data.filename}</p>
            <button class="btn btn-primary" onclick="window.socketApp.resetUpload()">
                <i class="fas fa-upload"></i> Tải file khác
            </button>
        `;
        this.displayFileInfo(data);
        this.processBtn.disabled = false;
    }

    handleUploadError(data) {
        this.addLog(`Lỗi tải file: ${data.message}`, 'error');
        this.resetUploadArea();
    }

    displayFileInfo(data) {
        const { stats } = data;
        this.fileInfo.innerHTML = `
            <h3><i class="fas fa-info-circle"></i> Thông tin Dữ liệu</h3>
            <div class="stats-grid">
                <div class="stat-item"><div class="stat-value">${stats.total_messages}</div><div class="stat-label">Tổng Tin nhắn</div></div>
                <div class="stat-item"><div class="stat-value">${stats.ham_count}</div><div class="stat-label">Bình thường</div></div>
                <div class="stat-item"><div class="stat-value">${stats.spam_count}</div><div class="stat-label">Spam</div></div>
                <div class="stat-item"><div class="stat-value">${stats.spam_percentage}%</div><div class="stat-label">Tỷ lệ Spam</div></div>
            </div>`;
        this.fileInfo.classList.remove('hidden');
    }

    resetUpload() {
        this.resetUploadArea();
        this.fileInput.value = '';
        this.fileInfo.classList.add('hidden');
        this.processBtn.disabled = true;
        this.resultsSection.classList.add('hidden');
        this.competitionSection.classList.add('hidden');
        this.downloadTestBtn.disabled = true;
        this.downloadSubmissionBtn.disabled = true;
        this.addLog('Đã reset để tải file mới', 'info');
    }

    resetUploadArea() {
        this.uploadArea.classList.remove('loading');
        this.uploadArea.innerHTML = `
            <i class="fas fa-cloud-upload-alt"></i>
            <p>Kéo thả file CSV vào đây hoặc nhấn để chọn file</p>
            <button class="btn btn-primary" onclick="document.getElementById('fileInput').click()">
                <i class="fas fa-file-csv"></i> Chọn File CSV
            </button>`;
    }

    processData() {
        this.addLog('Bắt đầu xử lý dữ liệu...', 'info');
        this.processBtn.disabled = true;
        this.processBtn.innerHTML = '<i class="fas fa-spinner fa-spin"></i> Đang xử lý...';
        this.socket.emit('process_data');
    }

    updateProgressStep(step, status, message = '') {
        const stepElement = this.progressSteps[`step${step}`];
        if (!stepElement) return;
        stepElement.classList.remove('active', 'completed', 'error');
        if (status !== 'pending') stepElement.classList.add(status);
        const statusIcon = stepElement.querySelector('.step-status i');
        switch (status) {
            case 'active': statusIcon.className = 'fas fa-spinner fa-spin'; break;
            case 'completed': statusIcon.className = 'fas fa-check'; break;
            case 'error': statusIcon.className = 'fas fa-times'; break;
            default: statusIcon.className = 'fas fa-clock';
        }
        if (message) this.addLog(`Bước ${step}: ${message}`, status === 'error' ? 'error' : 'info');
    }

    updatePreprocessingProgress(data) {
        this.preprocessProgress.classList.remove('hidden');
        this.preprocessProgress.querySelector('.progress-fill').style.width = `${data.progress}%`;
        this.preprocessProgress.querySelector('.progress-text').textContent = `${Math.round(data.progress)}%`;
    }

    updateTrainingProgress(data) {
        this.trainingProgress.classList.remove('hidden');
        this.addLog(`${data.method}: ${data.message}`, 'info');
    }

    handleModelCompleted(data) {
        this.addLog(`✅ ${data.method} hoàn thành - Độ chính xác: ${(data.results.accuracy * 100).toFixed(2)}%`, 'success');
        const trainingItem = document.createElement('div');
        trainingItem.className = 'training-item completed';
        trainingItem.innerHTML = `<span>${data.results.name}</span><span>✅ ${(data.results.accuracy * 100).toFixed(2)}%</span>`;
        this.trainingProgress.appendChild(trainingItem);
    }

    handleModelError(data) {
        this.addLog(`❌ ${data.method} thất bại: ${data.error}`, 'error');
        const trainingItem = document.createElement('div');
        trainingItem.className = 'training-item error';
        trainingItem.innerHTML = `<span>${data.method}</span><span>❌ Lỗi</span>`;
        this.trainingProgress.appendChild(trainingItem);
    }

    handleProcessCompleted(data) {
        this.addLog('🎉 Xử lý hoàn thành!', 'success');
        this.processBtn.innerHTML = '<i class="fas fa-play"></i> Bắt đầu Xử lý';
        this.processBtn.disabled = false;
        
        this.displayResults(data.results);
        
        if (data.comparison_chart) {
            this.comparisonChart.src = data.comparison_chart;
            this.comparisonChart.classList.remove('hidden');
        }
        if (data.feature_analysis) {
            this.featureAnalysis.src = data.feature_analysis;
            this.featureAnalysis.classList.remove('hidden');
        }
        if (data.confusion_matrix) {
            this.confusionMatrix.src = data.confusion_matrix;
            this.confusionMatrix.classList.remove('hidden');
        }
        if (data.summary) {
            this.displaySummary(data.summary);
        }
        
        this.resultsSection.classList.remove('hidden');
        this.competitionSection.classList.remove('hidden');
    }

    handleProcessError(data) {
        this.addLog(`❌ Lỗi xử lý: ${data.message}`, 'error');
        this.processBtn.innerHTML = '<i class="fas fa-play"></i> Bắt đầu Xử lý';
        this.processBtn.disabled = false;
    }

    displayResults(results) {
        let html = '';
        Object.values(results).forEach(metrics => {
            html += `
                <div class="result-card">
                    <h3>${metrics.name}</h3>
                    <div class="metric"><span class="metric-label">Độ chính xác</span><span class="metric-value">${(metrics.accuracy * 100).toFixed(2)}%</span></div>
                    <div class="metric"><span class="metric-label">Precision</span><span class="metric-value">${(metrics.precision * 100).toFixed(2)}%</span></div>
                    <div class="metric"><span class="metric-label">Recall</span><span class="metric-value">${(metrics.recall * 100).toFixed(2)}%</span></div>
                    <div class="metric"><span class="metric-label">F1-Score</span><span class="metric-value">${(metrics.f1_score * 100).toFixed(2)}%</span></div>
                    <div class="metric"><span class="metric-label">Thời gian</span><span class="metric-value">${metrics.training_time.toFixed(2)}s</span></div>
                </div>`;
        });
        this.resultsGrid.innerHTML = html;
    }

    displaySummary(summary) {
        this.summaryContent.innerHTML = `
            <div class="summary-item"><div class="summary-value">${summary.best_model}</div><div class="summary-label">Mô hình tốt nhất</div></div>
            <div class="summary-item"><div class="summary-value">${(summary.best_f1 * 100).toFixed(2)}%</div><div class="summary-label">F1-Score cao nhất</div></div>
            <div class="summary-item"><div class="summary-value">${summary.total_models}</div><div class="summary-label">Tổng số mô hình</div></div>
            <div class="summary-item"><div class="summary-value">${summary.total_time.toFixed(2)}s</div><div class="summary-label">Tổng thời gian</div></div>
            <div class="summary-item"><div class="summary-value">${summary.performance}</div><div class="summary-label">Đánh giá</div></div>
        `;
        this.summarySection.classList.remove('hidden');
    }
    
    displayConfusionMatrix(data) {
        this.confusionMatrix.src = data.image;
        this.confusionMatrix.classList.remove('hidden');
    }
    
    handleTestFilesReady(data) {
        this.addLog(`🏆 File cuộc thi đã sẵn sàng`, 'success');
        this.competitionData = data;
        this.downloadTestBtn.disabled = false;
        this.downloadSubmissionBtn.disabled = false;
    }
    
    downloadFile(content, filename, mimeType) {
        const blob = new Blob([content], { type: mimeType });
        const url = URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = filename;
        document.body.appendChild(a);
        a.click();
        document.body.removeChild(a);
        URL.revokeObjectURL(url);
        this.addLog(`✅ Đã tải xuống: ${filename}`, 'success');
    }

    downloadTestFile() {
        if (this.competitionData) {
            this.downloadFile(this.competitionData.test_csv, this.competitionData.test_filename, 'text/csv');
        } else {
            this.addLog('❌ Chưa có dữ liệu file test', 'error');
        }
    }
    
    downloadSubmissionFile() {
        if (this.competitionData) {
            this.downloadFile(this.competitionData.submission_csv, this.competitionData.submission_filename, 'text/csv');
        } else {
            this.addLog('❌ Chưa có dữ liệu file submission', 'error');
        }
    }
}

document.addEventListener('DOMContentLoaded', () => {
    window.socketApp = new SocketSpamClassifier();
});