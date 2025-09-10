/**
 * API Client for Multi-Omics Pharmacogenomics Platform
 */

class ApiClient {
    constructor() {
        this.baseURL = 'http://localhost:8000/api/v1';
        this.timeout = 30000; // 30 seconds
    }

    /**
     * Make HTTP request with error handling
     */
    async makeRequest(method, endpoint, data = null, isFormData = false) {
        const url = `${this.baseURL}${endpoint}`;
        
        const options = {
            method: method,
            headers: {
                'Accept': 'application/json'
            }
        };

        // Add content type for JSON requests
        if (!isFormData && data) {
            options.headers['Content-Type'] = 'application/json';
            options.body = JSON.stringify(data);
        } else if (isFormData && data) {
            options.body = data;
        }

        try {
            const response = await fetch(url, options);
            
            if (!response.ok) {
                const errorData = await response.json().catch(() => ({}));
                throw new Error(errorData.detail || `HTTP ${response.status}: ${response.statusText}`);
            }

            return await response.json();
        } catch (error) {
            if (error.name === 'TypeError' && error.message.includes('fetch')) {
                throw new Error('Network error: Unable to connect to the server');
            }
            throw error;
        }
    }

    // Health and Status
    async getHealth() {
        return await this.makeRequest('GET', '/health');
    }

    async getStatus() {
        return await this.makeRequest('GET', '/');
    }

    // Multi-Omics Data Endpoints
    async uploadOmicsData(patientId, dataType, formData) {
        const endpoint = `/omics/upload?patient_id=${encodeURIComponent(patientId)}&data_type=${encodeURIComponent(dataType)}`;
        return await this.makeRequest('POST', endpoint, formData, true);
    }

    async getAvailableDatasets() {
        return await this.makeRequest('GET', '/omics/datasets');
    }

    async getPatientData(patientId) {
        return await this.makeRequest('GET', `/omics/patients/${encodeURIComponent(patientId)}/data`);
    }

    // Model Management Endpoints
    async getModelArchitectures() {
        return await this.makeRequest('GET', '/models/architectures');
    }

    async trainModel(modelType, dataTypes, hyperparameters, cvFolds = 5) {
        const data = {
            model_type: modelType,
            data_types: dataTypes,
            hyperparameters: hyperparameters,
            cross_validation_folds: cvFolds
        };
        return await this.makeRequest('POST', '/models/train', data);
    }

    async getTrainingStatus(jobId) {
        return await this.makeRequest('GET', `/models/training/${encodeURIComponent(jobId)}/status`);
    }

    // Analysis and Prediction Endpoints
    async predictDrugResponse(patientId, drugId, omicsDataTypes, modelType = 'multi_omics_fusion') {
        const data = {
            patient_id: patientId,
            drug_id: drugId,
            omics_data_types: omicsDataTypes,
            model_type: modelType
        };
        return await this.makeRequest('POST', '/analysis/predict', data);
    }

    async explainPrediction(patientId, drugId, predictionId, explanationMethod = 'shap') {
        const data = {
            patient_id: patientId,
            drug_id: drugId,
            prediction_id: predictionId,
            explanation_method: explanationMethod
        };
        return await this.makeRequest('POST', '/analysis/explain', data);
    }

    async discoverBiomarkers(drugClass = null, omicsType = null, significanceThreshold = 0.05) {
        const params = new URLSearchParams();
        if (drugClass) params.append('drug_class', drugClass);
        if (omicsType) params.append('omics_type', omicsType);
        params.append('significance_threshold', significanceThreshold);
        
        return await this.makeRequest('GET', `/analysis/biomarkers?${params.toString()}`);
    }

    // Batch Operations
    async uploadMultipleFiles(files, patientId, dataType) {
        const results = [];
        
        for (let i = 0; i < files.length; i++) {
            const file = files[i];
            const formData = new FormData();
            formData.append('file', file);
            
            try {
                const result = await this.uploadOmicsData(patientId, dataType, formData);
                results.push({ file: file.name, success: true, result });
            } catch (error) {
                results.push({ file: file.name, success: false, error: error.message });
            }
        }
        
        return results;
    }

    // Data Processing Utilities
    async processGenomicsData(patientId, options = {}) {
        const data = {
            patient_id: patientId,
            processing_options: {
                quality_threshold: options.qualityThreshold || 30,
                reference_genome: options.referenceGenome || 'GRCh38',
                filter_variants: options.filterVariants || true,
                ...options
            }
        };
        return await this.makeRequest('POST', '/omics/process/genomics', data);
    }

    async processTranscriptomicsData(patientId, options = {}) {
        const data = {
            patient_id: patientId,
            processing_options: {
                normalization_method: options.normalizationMethod || 'TPM',
                filter_low_expression: options.filterLowExpression || true,
                log_transform: options.logTransform || true,
                ...options
            }
        };
        return await this.makeRequest('POST', '/omics/process/transcriptomics', data);
    }

    async processProteomicsData(patientId, options = {}) {
        const data = {
            patient_id: patientId,
            processing_options: {
                normalization_method: options.normalizationMethod || 'median',
                missing_value_imputation: options.missingValueImputation || 'knn',
                batch_correction: options.batchCorrection || false,
                ...options
            }
        };
        return await this.makeRequest('POST', '/omics/process/proteomics', data);
    }

    // Model Performance and Metrics
    async getModelPerformance(modelId) {
        return await this.makeRequest('GET', `/models/${encodeURIComponent(modelId)}/performance`);
    }

    async compareModels(modelIds) {
        const data = { model_ids: modelIds };
        return await this.makeRequest('POST', '/models/compare', data);
    }

    // Visualization Data
    async getVisualizationData(dataType, patientId, options = {}) {
        const params = new URLSearchParams({
            data_type: dataType,
            patient_id: patientId,
            ...options
        });
        return await this.makeRequest('GET', `/visualization/data?${params.toString()}`);
    }

    async getShapValues(predictionId) {
        return await this.makeRequest('GET', `/visualization/shap/${encodeURIComponent(predictionId)}`);
    }

    async getAttentionWeights(predictionId) {
        return await this.makeRequest('GET', `/visualization/attention/${encodeURIComponent(predictionId)}`);
    }

    // Database Operations
    async searchPatients(query, filters = {}) {
        const data = { query, filters };
        return await this.makeRequest('POST', '/database/patients/search', data);
    }

    async getPatientHistory(patientId) {
        return await this.makeRequest('GET', `/database/patients/${encodeURIComponent(patientId)}/history`);
    }

    // System Administration
    async getSystemStats() {
        return await this.makeRequest('GET', '/admin/stats');
    }

    async getResourceUsage() {
        return await this.makeRequest('GET', '/admin/resources');
    }

    // Error Handling Utilities
    handleApiError(error) {
        console.error('API Error:', error);
        
        let userMessage = 'An unexpected error occurred';
        
        if (error.message.includes('Network error')) {
            userMessage = 'Unable to connect to the server. Please check your internet connection.';
        } else if (error.message.includes('HTTP 400')) {
            userMessage = 'Invalid request. Please check your input data.';
        } else if (error.message.includes('HTTP 401')) {
            userMessage = 'Authentication required. Please log in.';
        } else if (error.message.includes('HTTP 403')) {
            userMessage = 'Access denied. You don\'t have permission for this operation.';
        } else if (error.message.includes('HTTP 404')) {
            userMessage = 'Resource not found. The requested data may not exist.';
        } else if (error.message.includes('HTTP 500')) {
            userMessage = 'Server error. Please try again later or contact support.';
        } else if (error.message.includes('timeout')) {
            userMessage = 'Request timed out. The operation may take longer than expected.';
        }
        
        return { error: error.message, userMessage };
    }

    // Request Interceptors
    async authenticatedRequest(method, endpoint, data = null) {
        // Add authentication token if available
        const token = localStorage.getItem('auth_token');
        if (token) {
            const headers = { 'Authorization': `Bearer ${token}` };
            return await this.makeRequest(method, endpoint, data, false, headers);
        }
        return await this.makeRequest(method, endpoint, data);
    }

    // File Upload Progress
    async uploadWithProgress(file, endpoint, progressCallback) {
        const formData = new FormData();
        formData.append('file', file);

        return new Promise((resolve, reject) => {
            const xhr = new XMLHttpRequest();
            
            xhr.upload.addEventListener('progress', (e) => {
                if (e.lengthComputable && progressCallback) {
                    const progress = (e.loaded / e.total) * 100;
                    progressCallback(progress);
                }
            });

            xhr.addEventListener('load', () => {
                if (xhr.status >= 200 && xhr.status < 300) {
                    resolve(JSON.parse(xhr.responseText));
                } else {
                    reject(new Error(`HTTP ${xhr.status}: ${xhr.statusText}`));
                }
            });

            xhr.addEventListener('error', () => {
                reject(new Error('Network error occurred'));
            });

            xhr.open('POST', `${this.baseURL}${endpoint}`);
            xhr.setRequestHeader('Accept', 'application/json');
            xhr.send(formData);
        });
    }
}

// Create global instance
const apiClient = new ApiClient();

// Export for use in other modules
window.ApiClient = apiClient;
