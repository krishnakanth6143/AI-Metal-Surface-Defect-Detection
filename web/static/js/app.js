// Modern UI - Metal Defect Detection System

let currentFilename = null;
let selectedImage = null;

// Initialize on DOM load
document.addEventListener('DOMContentLoaded', () => {
    setupUploadArea();
    setupThemeToggle();
    updateStats();
    setInterval(updateStats, 5000);
    
    // Restore saved theme
    const savedTheme = localStorage.getItem('theme') || 'light';
    applyTheme(savedTheme);
});

// ============================================
// THEME TOGGLE
// ============================================

function setupThemeToggle() {
    const themeToggle = document.getElementById('themeToggle');
    if (themeToggle) {
        themeToggle.addEventListener('click', () => {
            const isDark = document.body.classList.contains('dark-theme');
            const newTheme = isDark ? 'light' : 'dark';
            applyTheme(newTheme);
            localStorage.setItem('theme', newTheme);
        });
    }
}

function applyTheme(theme) {
    if (theme === 'dark') {
        document.body.classList.add('dark-theme');
    } else {
        document.body.classList.remove('dark-theme');
    }
}

// ============================================
// UPLOAD AREA SETUP
// ============================================

function setupUploadArea() {
    const uploadArea = document.getElementById('uploadArea');
    const fileInput = document.getElementById('fileInput');

    if (!uploadArea || !fileInput) return;

    // Click to browse
    uploadArea.addEventListener('click', () => fileInput.click());

    // Drag over
    uploadArea.addEventListener('dragover', (e) => {
        e.preventDefault();
        uploadArea.classList.add('dragover');
    });

    // Drag leave
    uploadArea.addEventListener('dragleave', () => {
        uploadArea.classList.remove('dragover');
    });

    // Drop
    uploadArea.addEventListener('drop', (e) => {
        e.preventDefault();
        uploadArea.classList.remove('dragover');
        
        const files = e.dataTransfer.files;
        if (files.length > 0) {
            handleFileSelect(files[0]);
        }
    });

    // File input change
    fileInput.addEventListener('change', (e) => {
        if (e.target.files.length > 0) {
            handleFileSelect(e.target.files[0]);
        }
    });
}

// ============================================
// FILE HANDLING
// ============================================

function handleFileSelect(file) {
    // Validate file type
    if (!file.type.startsWith('image/')) {
        showError('Please select an image file (JPG, PNG, BMP)');
        return;
    }

    // Validate file size
    if (file.size > 10 * 1024 * 1024) {
        showError('File size must be less than 10MB');
        return;
    }

    // Show preview
    const reader = new FileReader();
    reader.onload = (e) => {
        selectedImage = file;
        document.getElementById('imagePreview').src = e.target.result;
        document.getElementById('previewArea').style.display = 'block';
        
        // Reset result area
        document.getElementById('resultArea').innerHTML = 
            '<p class="text-muted text-center">Click "Analyze Image" to process</p>';
    };
    reader.readAsDataURL(file);
}

// ============================================
// ANALYZE IMAGE
// ============================================

function analyzeImage() {
    if (!selectedImage) {
        showError('Please select an image first');
        return;
    }

    // Show loading state
    document.getElementById('loadingArea').style.display = 'block';
    document.getElementById('previewArea').style.display = 'none';
    
    // Upload and analyze
    uploadFile(selectedImage);
}

function uploadFile(file) {
    const formData = new FormData();
    formData.append('file', file);

    fetch('/upload', {
        method: 'POST',
        body: formData
    })
    .then(response => response.json())
    .then(data => {
        document.getElementById('loadingArea').style.display = 'none';
        document.getElementById('previewArea').style.display = 'block';
        
        if (data.success) {
            currentFilename = data.filename;
            displayResults(data.prediction);
            updateStats();
            showSuccess('Analysis complete!');
        } else {
            showError(data.error || 'Analysis failed');
        }
    })
    .catch(error => {
        document.getElementById('loadingArea').style.display = 'none';
        showError('Error: ' + error.message);
    });
}

// ============================================
// DISPLAY RESULTS
// ============================================

function displayResults(prediction) {
    const resultArea = document.getElementById('resultArea');
    
    const defectClass = prediction.class;
    const confidence = (parseFloat(prediction.confidence) * 100).toFixed(1);
    const isDefect = prediction.is_defect;
    const probabilities = prediction.probabilities;

    // Build result HTML
    const badgeClass = isDefect ? 'danger' : 'success';
    const badgeIcon = isDefect ? '<i class="fas fa-exclamation-triangle me-2"></i>' : '<i class="fas fa-check-circle me-2"></i>';
    const badgeText = isDefect ? 'DEFECT DETECTED' : 'NO DEFECT';
    const recommendation = isDefect ? 'REJECT' : 'ACCEPT';
    
    let html = `
        <div class="alert alert-${badgeClass} border-0 border-start border-4 mb-4" role="alert">
            <h5 class="mb-0">${badgeIcon}${badgeText}</h5>
        </div>
        
        <div class="mb-4">
            <h6>Predicted Class</h6>
            <p class="fs-5 fw-bold text-primary">${defectClass}</p>
        </div>

        <div class="mb-4">
            <h6>Confidence Score</h6>
            <div class="progress" style="height: 25px;">
                <div class="progress-bar bg-primary" role="progressbar" 
                     style="width: ${confidence}%" 
                     aria-valuenow="${confidence}" aria-valuemin="0" aria-valuemax="100">
                    <span class="fw-bold" style="color: white;">${confidence}%</span>
                </div>
            </div>
        </div>

        <div>
            <h6>Quality Control</h6>
            <span class="badge bg-${badgeClass} fs-6 p-2">
                <i class="fas fa-${isDefect ? 'times-circle' : 'check-circle'} me-2"></i>
                ${recommendation}
            </span>
        </div>
    `;
    
    resultArea.innerHTML = html;

    // Display probabilities
    displayProbabilities(probabilities);
    
    // Add to history
    addToHistory(defectClass, confidence, isDefect, currentFilename);
}

// ============================================
// DISPLAY PROBABILITIES
// ============================================

function displayProbabilities(probabilities) {
    const probabilitiesArea = document.getElementById('probabilitiesArea');
    
    // Sort by probability
    const sorted = Object.entries(probabilities).sort((a, b) => b[1] - a[1]);
    
    let html = '<div class="row">';
    
    for (const [className, prob] of sorted) {
        const percentage = (prob * 100).toFixed(1);
        const colors = [
            'bg-primary', 'bg-success', 'bg-danger', 
            'bg-warning', 'bg-info', 'bg-secondary'
        ];
        const colorClass = colors[sorted.indexOf([className, prob]) % colors.length];
        
        html += `
            <div class="col-md-6 mb-3">
                <div class="p-3 rounded-3 bg-light">
                    <div class="d-flex justify-content-between mb-2">
                        <small class="fw-bold">${className}</small>
                        <small class="fw-bold text-primary">${percentage}%</small>
                    </div>
                    <div class="progress" style="height: 6px;">
                        <div class="progress-bar ${colorClass}" style="width: ${percentage}%" role="progressbar"></div>
                    </div>
                </div>
            </div>
        `;
    }
    
    html += '</div>';
    probabilitiesArea.innerHTML = html;
}

// ============================================
// HISTORY MANAGEMENT
// ============================================

function addToHistory(className, confidence, isDefect, filename) {
    // This will be synced with updateHistory which fetches from backend
    updateStats();
}

function updateStats() {
    fetch('/history', {
        method: 'GET'
    })
    .then(response => response.json())
    .then(data => {
        if (data.success) {
            const summary = data.summary;
            
            // Update stat cards with animation
            animateCounter('totalScans', parseInt(summary.total_predictions));
            animateCounter('defectsFound', parseInt(summary.defects_found));
            animateCounter('normalSurfaces', parseInt(summary.normal_surfaces));
            
            updateHistory(data.history);
        }
    })
    .catch(error => console.error('Error fetching stats:', error));
}

function animateCounter(elementId, newValue) {
    const element = document.getElementById(elementId);
    if (!element) return;
    
    const currentValue = parseInt(element.textContent) || 0;
    
    if (currentValue !== newValue) {
        element.style.color = '#0066ff';
        element.textContent = newValue;
        
        element.style.animation = 'none';
        setTimeout(() => {
            element.style.animation = 'scale-up 0.3s ease-out';
        }, 10);
    }
}

function updateHistory(history) {
    const historyArea = document.getElementById('historyArea');
    
    if (history.length === 0) {
        historyArea.innerHTML = `
            <div class="text-center text-muted py-4">
                <i class="fas fa-inbox fa-2x mb-2 opacity-50 d-block"></i>
                <p>No analysis history yet</p>
            </div>
        `;
        return;
    }
    
    let html = '';
    const recentHistory = history.slice(-10).reverse(); // Show last 10
    
    for (const item of recentHistory) {
        const itemClass = item.is_defect ? 'defect' : 'normal';
        const icon = item.is_defect ? 'fas fa-exclamation-triangle text-danger' : 'fas fa-check-circle text-success';
        const badgeClass = item.is_defect ? 'danger' : 'success';
        
        html += `
            <div class="history-item ${itemClass} fade-in">
                <div class="d-flex justify-content-between align-items-start mb-2">
                    <div class="flex-grow-1">
                        <div class="d-flex align-items-center gap-2 mb-1">
                            <i class="${icon}"></i>
                            <strong>${item.predicted_class}</strong>
                        </div>
                        <small class="text-muted">${item.filename}</small>
                    </div>
                    <span class="badge bg-${badgeClass}">
                        ${(parseFloat(item.confidence_score) * 100 || item.confidence_score).toFixed(1)}%
                    </span>
                </div>
            </div>
        `;
    }
    
    historyArea.innerHTML = html;
}

// ============================================
// CLEAR PREVIEW
// ============================================

function clearPreview() {
    document.getElementById('previewArea').style.display = 'none';
    document.getElementById('loadingArea').style.display = 'none';
    document.getElementById('fileInput').value = '';
    document.getElementById('resultArea').innerHTML = 
        '<p class="text-muted text-center">Upload an image to start</p>';
    document.getElementById('probabilitiesArea').innerHTML = 
        '<p class="text-muted text-center">Analyze an image to see classification results</p>';
    selectedImage = null;
    currentFilename = null;
}

// ============================================
// NOTIFICATIONS
// ============================================

function showError(message) {
    showAlert(message, 'danger');
}

function showSuccess(message) {
    showAlert(message, 'success');
}

function showAlert(message, type = 'info') {
    const alertId = 'alert-' + Date.now();
    const alert = `
        <div id="${alertId}" class="alert alert-${type} alert-dismissible fade show shadow-sm rounded-3" role="alert" style="animation: slideIn 0.3s ease-out;">
            <i class="fas fa-${type === 'danger' ? 'exclamation-circle' : 'check-circle'} me-2"></i>
            ${message}
            <button type="button" class="btn-close" data-bs-dismiss="alert"></button>
        </div>
    `;
    
    const container = document.querySelector('.container-fluid');
    if (container) {
        container.insertAdjacentHTML('afterbegin', alert);
        
        setTimeout(() => {
            const element = document.getElementById(alertId);
            if (element) {
                element.classList.add('fade');
                setTimeout(() => element.remove(), 150);
            }
        }, 4000);
    }
}

// ============================================
// ANIMATIONS
// ============================================

const style = document.createElement('style');
style.textContent = `
    @keyframes scale-up {
        0% {
            transform: scale(1);
        }
        50% {
            transform: scale(1.1);
        }
        100% {
            transform: scale(1);
        }
    }
    
    @keyframes slideIn {
        from {
            transform: translateX(-100%);
            opacity: 0;
        }
        to {
            transform: translateX(0);
            opacity: 1;
        }
    }
`;
document.head.appendChild(style);
