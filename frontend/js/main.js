/**
 * Multi-Omics Pharmacogenomics Platform - Main JavaScript
 */

// Global state management
const AppState = {
    currentSection: 'home',
    uploadedFiles: [],
    predictions: [],
    models: {},
    activeTab: 'shap',
    selectedFiles: [] // Store selected files globally
};

// DOM Ready
document.addEventListener('DOMContentLoaded', function() {
    initializeApp();
});

/**
 * Initialize the application
 */
function initializeApp() {
    // Set up navigation
    setupNavigation();
    
    // Set up file upload functionality
    setupFileUpload();
    
    // Set up smooth scrolling
    setupSmoothScrolling();
    
    // Initialize API health check (wait for ApiClient to be ready)
    waitForApiClient(checkAPIHealth);
    
    // Add direct event listener to upload button as backup
    setTimeout(() => {
        const uploadBtn = document.querySelector('button[onclick="uploadFiles()"]');
        if (uploadBtn) {
            uploadBtn.addEventListener('click', function(e) {
                console.log('🖱️ Upload button clicked via event listener!');
                e.preventDefault();
                uploadFiles();
            });
            console.log('✅ Upload button event listener attached');
        } else {
            console.error('❌ Upload button not found for event listener');
        }
    }, 1000);
    
    console.log('Multi-Omics Platform initialized');
}

function getApi() {
    const api = window.ApiClient || window.apiClient || window.API;
    console.log('🔍 getApi() called, returning:', api);
    console.log('🔍 Available window objects:', {
        ApiClient: window.ApiClient,
        apiClient: window.apiClient,
        API: window.API
    });
    
    if (!api) {
        console.error('❌ API client not found! Check console for details.');
    } else {
        console.log('✅ API client found:', api.constructor?.name || 'Unknown');
        console.log('API client methods:', Object.getOwnPropertyNames(Object.getPrototypeOf(api)));
        console.log('Has uploadOmicsData?', typeof api.uploadOmicsData);
    }
    
    return api;
}

function waitForApiClient(callback, retries = 20) {
    const api = getApi();
    if (api && typeof api.getHealth === 'function') {
        callback();
        return;
    }
    if (retries <= 0) return;
    setTimeout(() => waitForApiClient(callback, retries - 1), 150);
}

/**
 * Navigation setup
 */
function setupNavigation() {
    const navLinks = document.querySelectorAll('.nav-link');
    
    navLinks.forEach(link => {
        link.addEventListener('click', function(e) {
            e.preventDefault();
            const target = this.getAttribute('href').substring(1);
            scrollToSection(target);
            updateActiveNav(this);
        });
    });
}

/**
 * Update active navigation item
 */
function updateActiveNav(activeLink) {
    document.querySelectorAll('.nav-link').forEach(link => {
        link.classList.remove('active');
    });
    activeLink.classList.add('active');
}

/**
 * Smooth scrolling setup
 */
function setupSmoothScrolling() {
    // Handle scroll events to update navigation
    window.addEventListener('scroll', throttle(updateNavOnScroll, 100));
}

/**
 * Update navigation based on scroll position
 */
function updateNavOnScroll() {
    const sections = document.querySelectorAll('section[id]');
    const scrollPos = window.scrollY + 100;
    
    sections.forEach(section => {
        const sectionTop = section.offsetTop;
        const sectionHeight = section.offsetHeight;
        const sectionId = section.getAttribute('id');
        
        if (scrollPos >= sectionTop && scrollPos < sectionTop + sectionHeight) {
            const activeNav = document.querySelector(`[href="#${sectionId}"]`);
            if (activeNav) {
                updateActiveNav(activeNav);
            }
        }
    });
}

/**
 * Scroll to section
 */
function scrollToSection(sectionId) {
    const element = document.getElementById(sectionId);
    if (element) {
        element.scrollIntoView({
            behavior: 'smooth',
            block: 'start'
        });
        AppState.currentSection = sectionId;
    }
}

/**
 * File upload setup
 */
function setupFileUpload() {
    const fileUploadArea = document.getElementById('fileUploadArea');
    const fileInput = document.getElementById('fileInput');
    
    console.log('Setting up file upload:', { fileUploadArea, fileInput });
    
    if (fileUploadArea && fileInput) {
        // Remove existing event listeners to avoid duplicates
        fileUploadArea.removeEventListener('click', handleFileUploadClick);
        fileInput.removeEventListener('change', handleFileSelection);
        fileUploadArea.removeEventListener('dragover', handleDragOver);
        fileUploadArea.removeEventListener('drop', handleFileDrop);
        fileUploadArea.removeEventListener('dragleave', handleDragLeave);
        
        // Click to browse files
        fileUploadArea.addEventListener('click', handleFileUploadClick);
        
        // File input change
        fileInput.addEventListener('change', handleFileSelection);
        
        // Drag and drop functionality
        fileUploadArea.addEventListener('dragover', handleDragOver);
        fileUploadArea.addEventListener('drop', handleFileDrop);
        fileUploadArea.addEventListener('dragleave', handleDragLeave);
        
        console.log('✅ File upload setup complete');
    } else {
        console.error('❌ File upload setup failed - missing elements:', { fileUploadArea, fileInput });
    }
}

/**
 * Handle file upload area click
 */
function handleFileUploadClick() {
    const fileInput = document.getElementById('fileInput');
    if (fileInput) {
        fileInput.click();
    }
}

/**
 * Handle file selection
 */
function handleFileSelection(event) {
    const files = event.target.files;
    if (files.length > 0) {
        // Store files globally
        AppState.selectedFiles = Array.from(files);
        console.log('Files stored globally:', AppState.selectedFiles);
        displaySelectedFiles(files);
    }
}

/**
 * Handle drag over
 */
function handleDragOver(event) {
    event.preventDefault();
    event.currentTarget.classList.add('dragover');
}

/**
 * Handle drag leave
 */
function handleDragLeave(event) {
    event.currentTarget.classList.remove('dragover');
}

/**
 * Handle file drop
 */
function handleFileDrop(event) {
    event.preventDefault();
    event.currentTarget.classList.remove('dragover');
    
    const files = event.dataTransfer.files;
    if (files.length > 0) {
        document.getElementById('fileInput').files = files;
        displaySelectedFiles(files);
    }
}

/**
 * Display selected files
 */
function displaySelectedFiles(files) {
    const fileUploadArea = document.getElementById('fileUploadArea');
    const fileList = Array.from(files).map(file => `${file.name} (${formatFileSize(file.size)})`).join(', ');
    
    fileUploadArea.innerHTML = `
        <i class="fas fa-check-circle" style="color: var(--secondary-color);"></i>
        <p><strong>Selected files:</strong><br>${fileList}</p>
    `;
}

/**
 * Format file size
 */
function formatFileSize(bytes) {
    if (bytes === 0) return '0 Bytes';
    const k = 1024;
    const sizes = ['Bytes', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
}

/**
 * Show upload modal
 */
function showUploadModal(dataType = 'genomics') {
    console.log('📁 Show upload modal called with data type:', dataType);
    const modal = document.getElementById('uploadModal');
    if (modal) {
        modal.style.display = 'flex';
        
        // Reset form fields and set data type
        document.getElementById('uploadDataType').value = dataType;
        document.getElementById('uploadPatientId').value = '';
        
        // Clear selected files
        AppState.selectedFiles = [];
        
        // Ensure file upload area is properly set up
        const fileUploadArea = document.getElementById('fileUploadArea');
        if (fileUploadArea) {
            // Create the file input directly
            fileUploadArea.innerHTML = `
                <i class="fas fa-cloud-upload-alt"></i>
                <p>Drag & drop files here or click to browse</p>
                <input type="file" id="fileInput" multiple>
            `;
            
            // Setup immediately after creating
            setupFileUpload();
            console.log('✅ File input element created and setup complete');
        }
        
        console.log('✅ Upload modal displayed with data type:', dataType);
    } else {
        console.error('❌ Upload modal not found!');
    }
}

/**
 * Test file input (for debugging)
 */
function testFileInput() {
    const fileInput = document.getElementById('fileInput');
    console.log('Test file input:', fileInput);
    console.log('Global selected files:', AppState.selectedFiles);
    
    if (AppState.selectedFiles && AppState.selectedFiles.length > 0) {
        console.log('Global files found! Files:', AppState.selectedFiles);
        alert('Files found! Count: ' + AppState.selectedFiles.length + ', First file: ' + AppState.selectedFiles[0].name);
    } else if (fileInput && fileInput.files) {
        console.log('File input found! Files:', fileInput.files);
        alert('File input found! Files: ' + (fileInput.files?.length || 0));
    } else {
        console.log('No files found!');
        alert('No files found!');
    }
}

/**
 * Hide upload modal
 */
function hideUploadModal() {
    const modal = document.getElementById('uploadModal');
    if (modal) {
        modal.style.display = 'none';
        
        // Reset form
        const dataTypeSelect = document.getElementById('uploadDataType');
        const patientIdInput = document.getElementById('uploadPatientId');
        const fileInput = document.getElementById('fileInput');
        
        if (dataTypeSelect) dataTypeSelect.value = 'genomics';
        if (patientIdInput) patientIdInput.value = '';
        if (fileInput) fileInput.value = '';
        
        // Reset file upload area
        const fileUploadArea = document.getElementById('fileUploadArea');
        fileUploadArea.innerHTML = `
            <i class="fas fa-cloud-upload-alt"></i>
            <p>Drag & drop files here or click to browse</p>
            <input type="file" id="fileInput" multiple>
        `;
        
        // Re-attach event listeners
        setupFileUpload();
        
        // Verify file input was created
        const newFileInput = document.getElementById('fileInput');
        if (!newFileInput) {
            console.error('❌ Failed to recreate file input element');
        } else {
            console.log('✅ File input element recreated successfully');
        }
        
        // Reset progress
        const progressDiv = document.getElementById('uploadProgress');
        progressDiv.style.display = 'none';
        progressDiv.innerHTML = `
            <div class="progress-bar">
                <div class="progress-fill" id="progressFill"></div>
            </div>
            <span class="progress-text" id="progressText">0%</span>
        `;
        
        // Clear global files
        AppState.selectedFiles = [];
        console.log('Global files cleared');
    }
}

/**
 * Upload files
 */
async function uploadFiles() {
    console.log('🚀 Upload files function called!');
    
    const dataType = document.getElementById('uploadDataType').value;
    const patientId = document.getElementById('uploadPatientId').value;
    
    console.log('Upload params:', { dataType, patientId, selectedFiles: AppState.selectedFiles });
    
    // Validate inputs
    if (!patientId.trim()) {
        showNotification('Please enter a patient ID', 'warning');
        return;
    }
    
    if (!AppState.selectedFiles || AppState.selectedFiles.length === 0) {
        showNotification('Please select files to upload', 'warning');
        return;
    }
    
    // Check API client
    const api = getApi();
    if (!api) {
        showNotification('API client not available', 'error');
        return;
    }
    
    if (!api.uploadOmicsData) {
        showNotification('API client missing upload method', 'error');
        return;
    }
    
    // Show progress
    const progressDiv = document.getElementById('uploadProgress');
    if (progressDiv) {
        progressDiv.style.display = 'block';
    }
    
    try {
        console.log('Starting upload process...');
        
        for (let i = 0; i < AppState.selectedFiles.length; i++) {
            const file = AppState.selectedFiles[i];
            console.log(`Uploading file ${i + 1}/${AppState.selectedFiles.length}:`, file.name);
            
            try {
                await uploadSingleFile(file, dataType, patientId, i + 1, AppState.selectedFiles.length);
                console.log(`File ${i + 1} uploaded successfully!`);
            } catch (fileError) {
                console.error(`Error uploading file ${i + 1}:`, fileError);
                throw fileError;
            }
        }
        
        console.log('All files uploaded successfully');
        
        // Show success in modal
        const progressDiv = document.getElementById('uploadProgress');
        if (progressDiv) {
            progressDiv.innerHTML = `
                <div style="text-align: center; color: #10b981; font-weight: bold;">
                    <i class="fas fa-check-circle" style="font-size: 24px; margin-bottom: 10px;"></i>
                    <div>✅ Files uploaded successfully!</div>
                    <div style="font-size: 14px; margin-top: 5px;">Processing in background...</div>
                </div>
            `;
        }
        
        // Show notification
        showNotification('✅ Files uploaded successfully!', 'success');
        
        // Wait a moment before closing modal to show success
        setTimeout(() => {
            hideUploadModal();
        }, 2000);
        
    } catch (error) {
        console.error('Upload error:', error);
        showNotification('❌ Upload failed: ' + error.message, 'error');
        
        // Hide progress on error
        const progressDiv = document.getElementById('uploadProgress');
        if (progressDiv) {
            progressDiv.style.display = 'none';
        }
    }
}

/**
 * Upload single file
 */
async function uploadSingleFile(file, dataType, patientId, current, total) {
    console.log('📤 Uploading file:', file.name, 'Type:', dataType, 'Patient:', patientId);
    
    const formData = new FormData();
    formData.append('file', file);
    
    try {
        console.log('📤 Calling API uploadOmicsData...');
        const api = getApi();
        console.log('📤 API client:', api);
        
        if (!api || !api.uploadOmicsData) {
            throw new Error('API client not available or uploadOmicsData method missing');
        }
        
        const response = await api.uploadOmicsData(patientId, dataType, formData);
        console.log('✅ Upload response:', response);
        
        // Update progress
        const progress = (current / total) * 100;
        const progressFill = document.getElementById('progressFill');
        const progressText = document.getElementById('progressText');
        
        if (progressFill) progressFill.style.width = `${progress}%`;
        if (progressText) progressText.textContent = `${Math.round(progress)}%`;
        
        return response;
    } catch (error) {
        console.error('❌ Upload error in uploadSingleFile:', error);
        throw error;
    }
}

/**
 * Train model
 */
async function trainModel() {
    const uiModelType = document.getElementById('modelType').value;
    const learningRate = parseFloat(document.getElementById('learningRate').value);
    const batchSize = parseInt(document.getElementById('batchSize').value);
    const cvFolds = parseInt(document.getElementById('cvFolds').value);
    
    const hyperparameters = {
        learning_rate: learningRate,
        batch_size: batchSize
    };
    
    const dataTypes = ['genomics', 'transcriptomics']; // Default selection
    
    // Map UI model types to backend lightweight types
    let modelType = uiModelType;
    if (uiModelType === 'genomics_cnn') modelType = 'genomics';
    else if (uiModelType === 'transcriptomics_rnn') modelType = 'transcriptomics';
    else if (uiModelType === 'proteomics_fc') modelType = 'proteomics';

    try {
        showNotification('Starting model training...', 'info');
        
        const response = await getApi().trainModel(modelType, dataTypes, hyperparameters, cvFolds);
        
        showNotification(`Training started! Job ID: ${response.job_id}`, 'success');
        
        // Monitor training progress
        monitorTrainingProgress(response.job_id);
        
    } catch (error) {
        console.error('Training error:', error);
        showNotification('Training failed: ' + error.message, 'error');
    }
}

/**
 * Monitor training progress
 */
async function monitorTrainingProgress(jobId) {
    const interval = setInterval(async () => {
        try {
            const status = await getApi().getTrainingStatus(jobId);
            
            console.log('Training status:', status);
            
            if (status.status === 'completed') {
                clearInterval(interval);
                showNotification('Model training completed!', 'success');
            } else if (status.status === 'failed') {
                clearInterval(interval);
                showNotification('Model training failed!', 'error');
            }
            
        } catch (error) {
            clearInterval(interval);
            console.error('Error monitoring training:', error);
        }
    }, 10000); // Check every 10 seconds
}

/**
 * Predict drug response
 */
async function predictDrugResponse() {
    const patientId = document.getElementById('patientId').value;
    const drugId = document.getElementById('drugId').value;
    
    if (!patientId.trim() || !drugId) {
        showNotification('Please enter patient ID and select a drug', 'warning');
        return;
    }
    
    // Get selected omics data types
    const omicsCheckboxes = document.querySelectorAll('input[type="checkbox"][value]');
    const omicsDataTypes = Array.from(omicsCheckboxes)
        .filter(cb => cb.checked)
        .map(cb => cb.value);
    
    if (omicsDataTypes.length === 0) {
        showNotification('Please select at least one omics data type', 'warning');
        return;
    }
    
    try {
        showNotification('Making prediction...', 'info');
        
        const response = await getApi().predictDrugResponse(patientId, drugId, omicsDataTypes);
        
        displayPredictionResults(response);
        showNotification('Prediction completed!', 'success');
        
    } catch (error) {
        console.error('Prediction error:', error);
        showNotification('Prediction failed: ' + error.message, 'error');
    }
}

/**
 * Display prediction results
 */
function displayPredictionResults(prediction) {
    const resultsDiv = document.getElementById('predictionResults');
    const scoreValue = document.getElementById('scoreValue');
    const biomarkersList = document.getElementById('biomarkersList');
    
    // Show results panel
    resultsDiv.style.display = 'block';
    
    // Update score
    const probability = prediction.prediction.predicted_response;
    scoreValue.textContent = (probability * 100).toFixed(1) + '%';
    
    // Update biomarkers
    biomarkersList.innerHTML = '';
    prediction.biomarkers.forEach(biomarker => {
        const biomarkerDiv = document.createElement('div');
        biomarkerDiv.className = 'biomarker-item';
        biomarkerDiv.innerHTML = `
            <span class="biomarker-name">${biomarker.name}</span>
            <span class="biomarker-importance">Importance: ${(biomarker.importance * 100).toFixed(1)}%</span>
        `;
        biomarkersList.appendChild(biomarkerDiv);
    });
    
    // Store prediction for interpretability
    AppState.currentPrediction = prediction;
    
    // Load default interpretability view
    showTab('shap');
}

/**
 * Show interpretability tab
 */
function showTab(tabName) {
    // Update tab buttons
    document.querySelectorAll('.tab-btn').forEach(btn => {
        btn.classList.remove('active');
    });
    document.querySelector(`[onclick="showTab('${tabName}')"]`).classList.add('active');
    
    // Update plot using visualization manager
    const plotContainer = document.getElementById('interpretabilityPlot');
    
    switch (tabName) {
        case 'shap':
            if (window.visualizationManager) {
                window.visualizationManager.createShapPlot('interpretabilityPlot', AppState.currentPrediction);
            } else {
                plotContainer.innerHTML = '<p>SHAP values visualization would be displayed here</p>';
            }
            break;
        case 'attention':
            if (window.visualizationManager) {
                window.visualizationManager.createAttentionHeatmap('interpretabilityPlot', AppState.currentPrediction);
            } else {
                plotContainer.innerHTML = '<p>Attention mechanism visualization would be displayed here</p>';
            }
            break;
        case 'features':
            if (window.visualizationManager) {
                window.visualizationManager.createFeatureImportanceChart('interpretabilityPlot', AppState.currentPrediction);
            } else {
                plotContainer.innerHTML = '<p>Feature importance plot would be displayed here</p>';
            }
            break;
    }
    
    AppState.activeTab = tabName;
}

/**
 * Check API health
 */
async function checkAPIHealth() {
    try {
        const health = await getApi().getHealth();
        console.log('API Health:', health);
        
        if (health.status === 'healthy') {
            showNotification('Platform is ready!', 'success', 2000);
        }
        
    } catch (error) {
        console.warn('API health check failed:', error);
        showNotification('Backend API is not available', 'warning', 3000);
    }
}

/**
 * Show notification
 */
function showNotification(message, type = 'info', duration = 4000) {
    // Create notification element
    const notification = document.createElement('div');
    notification.className = `notification notification-${type}`;
    notification.innerHTML = `
        <div class="notification-content">
            <span>${message}</span>
            <button class="notification-close" onclick="this.parentElement.parentElement.remove()">&times;</button>
        </div>
    `;
    
    // Add styles if not already present
    if (!document.getElementById('notification-styles')) {
        const style = document.createElement('style');
        style.id = 'notification-styles';
        style.textContent = `
            .notification {
                position: fixed;
                top: 100px;
                right: 20px;
                z-index: 3000;
                max-width: 400px;
                padding: 16px;
                border-radius: 8px;
                box-shadow: 0 4px 12px rgba(0, 0, 0, 0.1);
                animation: slideInRight 0.3s ease;
            }
            
            .notification-info { background: #3b82f6; color: white; }
            .notification-success { 
                background: #10b981; 
                color: white; 
                border: 2px solid #059669;
                font-weight: bold;
            }
            .notification-warning { background: #f59e0b; color: white; }
            .notification-error { background: #ef4444; color: white; }
            
            .notification-content {
                display: flex;
                justify-content: space-between;
                align-items: center;
                gap: 12px;
            }
            
            .notification-close {
                background: none;
                border: none;
                color: inherit;
                font-size: 18px;
                cursor: pointer;
                opacity: 0.8;
            }
            
            .notification-close:hover { opacity: 1; }
            
            @keyframes slideInRight {
                from { transform: translateX(100%); opacity: 0; }
                to { transform: translateX(0); opacity: 1; }
            }
        `;
        document.head.appendChild(style);
    }
    
    // Add to page
    document.body.appendChild(notification);
    
    // Auto remove
    setTimeout(() => {
        notification.remove();
    }, duration);
}

/**
 * Throttle function for performance
 */
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
    }
}

/**
 * Utility function for element selection
 */
function $(selector) {
    return document.querySelector(selector);
}

function $$(selector) {
    return document.querySelectorAll(selector);
}

// Export for use in other modules
window.AppState = AppState;
window.showUploadModal = showUploadModal;
window.hideUploadModal = hideUploadModal;
window.uploadFiles = uploadFiles;
window.trainModel = trainModel;
window.predictDrugResponse = predictDrugResponse;
window.showTab = showTab;
