// Main JavaScript file for Radioactive Watermark Detector

// Global variables
let systemStatus = {
    modelLoaded: false,
    modelExists: false,
    datasetAvailable: false,
    device: 'cpu'
};

// Initialize the application
document.addEventListener('DOMContentLoaded', function() {
    initializeApp();
});

function initializeApp() {
    // Check system status
    checkSystemStatus();
    
    // Set up periodic status checks
    setInterval(checkSystemStatus, 30000); // Check every 30 seconds
    
    // Initialize tooltips
    initializeTooltips();
    
    // Set up form validations
    setupFormValidations();
}

// Check system status
function checkSystemStatus() {
    fetch('/api/status')
        .then(response => response.json())
        .then(data => {
            systemStatus = data;
            updateStatusIndicator();
        })
        .catch(error => {
            console.error('Error checking system status:', error);
            updateStatusIndicator(true);
        });
}

// Update status indicator in navbar
function updateStatusIndicator(error = false) {
    const indicator = document.getElementById('status-indicator');
    if (!indicator) return;
    
    if (error) {
        indicator.className = 'badge bg-danger';
        indicator.innerHTML = '<i class="fas fa-circle"></i> Error';
        return;
    }
    
    if (systemStatus.modelLoaded && systemStatus.datasetAvailable) {
        indicator.className = 'badge bg-success';
        indicator.innerHTML = '<i class="fas fa-circle"></i> Ready';
    } else if (systemStatus.modelExists || systemStatus.datasetAvailable) {
        indicator.className = 'badge bg-warning';
        indicator.innerHTML = '<i class="fas fa-circle"></i> Setup Required';
    } else {
        indicator.className = 'badge bg-secondary';
        indicator.innerHTML = '<i class="fas fa-circle"></i> Not Ready';
    }
}

// Initialize Bootstrap tooltips
function initializeTooltips() {
    const tooltipTriggerList = [].slice.call(document.querySelectorAll('[data-bs-toggle="tooltip"]'));
    tooltipTriggerList.map(function (tooltipTriggerEl) {
        return new bootstrap.Tooltip(tooltipTriggerEl);
    });
}

// Set up form validations
function setupFormValidations() {
    // Training form validation
    const trainingForm = document.getElementById('trainingForm');
    if (trainingForm) {
        trainingForm.addEventListener('submit', function(e) {
            if (!validateTrainingForm()) {
                e.preventDefault();
            }
        });
    }
    
    // Upload form validation
    const uploadForm = document.getElementById('uploadForm');
    if (uploadForm) {
        uploadForm.addEventListener('submit', function(e) {
            if (!validateUploadForm()) {
                e.preventDefault();
            }
        });
    }
}

// Validate training form
function validateTrainingForm() {
    const sampleSize = document.getElementById('sampleSize');
    const replaceCount = document.getElementById('replaceCount');
    const epochs = document.getElementById('epochs');
    const batchSize = document.getElementById('batchSize');
    const learningRate = document.getElementById('learningRate');
    
    let isValid = true;
    
    // Validate sample size
    if (sampleSize && (parseInt(sampleSize.value) < 10 || parseInt(sampleSize.value) > 1000)) {
        showFieldError(sampleSize, 'Sample size must be between 10 and 1000');
        isValid = false;
    } else {
        clearFieldError(sampleSize);
    }
    
    // Validate replace count
    if (replaceCount && parseInt(replaceCount.value) < 1) {
        showFieldError(replaceCount, 'Replace count must be at least 1');
        isValid = false;
    } else {
        clearFieldError(replaceCount);
    }
    
    // Validate epochs
    if (epochs && (parseInt(epochs.value) < 1 || parseInt(epochs.value) > 100)) {
        showFieldError(epochs, 'Epochs must be between 1 and 100');
        isValid = false;
    } else {
        clearFieldError(epochs);
    }
    
    // Validate batch size
    if (batchSize && (parseInt(batchSize.value) < 1 || parseInt(batchSize.value) > 64)) {
        showFieldError(batchSize, 'Batch size must be between 1 and 64');
        isValid = false;
    } else {
        clearFieldError(batchSize);
    }
    
    // Validate learning rate
    if (learningRate && (parseFloat(learningRate.value) < 0.0001 || parseFloat(learningRate.value) > 0.1)) {
        showFieldError(learningRate, 'Learning rate must be between 0.0001 and 0.1');
        isValid = false;
    } else {
        clearFieldError(learningRate);
    }
    
    return isValid;
}

// Validate upload form
function validateUploadForm() {
    const fileInput = document.getElementById('imageFile');
    
    if (!fileInput.files || fileInput.files.length === 0) {
        showFieldError(fileInput, 'Please select an image file');
        return false;
    }
    
    const file = fileInput.files[0];
    
    // Check file size (16MB max)
    if (file.size > 16 * 1024 * 1024) {
        showFieldError(fileInput, 'File size must be less than 16MB');
        return false;
    }
    
    // Check file type
    const allowedTypes = ['image/jpeg', 'image/jpg', 'image/png', 'image/bmp'];
    if (!allowedTypes.includes(file.type)) {
        showFieldError(fileInput, 'File type must be JPG, PNG, or BMP');
        return false;
    }
    
    clearFieldError(fileInput);
    return true;
}

// Show field error
function showFieldError(field, message) {
    field.classList.add('is-invalid');
    
    // Remove existing error message
    const existingError = field.parentNode.querySelector('.invalid-feedback');
    if (existingError) {
        existingError.remove();
    }
    
    // Add new error message
    const errorDiv = document.createElement('div');
    errorDiv.className = 'invalid-feedback';
    errorDiv.textContent = message;
    field.parentNode.appendChild(errorDiv);
}

// Clear field error
function clearFieldError(field) {
    field.classList.remove('is-invalid');
    
    // Remove error message
    const errorDiv = field.parentNode.querySelector('.invalid-feedback');
    if (errorDiv) {
        errorDiv.remove();
    }
}

// Show notification
function showNotification(message, type = 'info') {
    const alertDiv = document.createElement('div');
    alertDiv.className = `alert alert-${type} alert-dismissible fade show`;
    alertDiv.innerHTML = `
        <i class="fas fa-${getIconForType(type)}"></i>
        ${message}
        <button type="button" class="btn-close" data-bs-dismiss="alert"></button>
    `;
    
    const container = document.querySelector('.container');
    if (container) {
        container.insertBefore(alertDiv, container.firstChild);
        
        // Auto-dismiss after 5 seconds
        setTimeout(() => {
            if (alertDiv.parentNode) {
                alertDiv.remove();
            }
        }, 5000);
    }
}

// Get icon for notification type
function getIconForType(type) {
    const icons = {
        'success': 'check-circle',
        'danger': 'exclamation-triangle',
        'warning': 'exclamation-triangle',
        'info': 'info-circle'
    };
    return icons[type] || 'info-circle';
}

// Format file size
function formatFileSize(bytes) {
    if (bytes === 0) return '0 Bytes';
    
    const k = 1024;
    const sizes = ['Bytes', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    
    return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
}

// Format date
function formatDate(date) {
    return new Date(date).toLocaleString();
}

// Download file
function downloadFile(url, filename) {
    const a = document.createElement('a');
    a.href = url;
    a.download = filename;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
}

// Copy to clipboard
function copyToClipboard(text) {
    navigator.clipboard.writeText(text).then(() => {
        showNotification('Copied to clipboard!', 'success');
    }).catch(err => {
        console.error('Failed to copy text: ', err);
        showNotification('Failed to copy to clipboard', 'danger');
    });
}

// Debounce function
function debounce(func, wait) {
    let timeout;
    return function executedFunction(...args) {
        const later = () => {
            clearTimeout(timeout);
            func(...args);
        };
        clearTimeout(timeout);
        timeout = setTimeout(later, wait);
    };
}

// Throttle function
function throttle(func, limit) {
    let inThrottle;
    return function() {
        const args = arguments;
        const context = this;
        if (!inThrottle) {
            func.apply(context, args);
            inThrottle = true;
            setTimeout(() => inThrottle = false, limit);
        }
    };
}

// Export functions for use in other files
window.RadioactiveWatermarkDetector = {
    showNotification,
    formatFileSize,
    formatDate,
    downloadFile,
    copyToClipboard,
    debounce,
    throttle,
    systemStatus
};
