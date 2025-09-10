-- Multi-Omics Pharmacogenomics Platform - Database Initialization
-- PostgreSQL database schema and initial data

-- Create database extensions
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS "pg_trgm";

-- Users and authentication
CREATE TABLE IF NOT EXISTS users (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    username VARCHAR(50) UNIQUE NOT NULL,
    email VARCHAR(100) UNIQUE NOT NULL,
    password_hash VARCHAR(255) NOT NULL,
    full_name VARCHAR(100),
    institution VARCHAR(100),
    role VARCHAR(20) DEFAULT 'researcher',
    is_active BOOLEAN DEFAULT true,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Patient/Sample data
CREATE TABLE IF NOT EXISTS patients (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    patient_id VARCHAR(50) UNIQUE NOT NULL,
    age INTEGER,
    gender VARCHAR(10),
    ethnicity VARCHAR(50),
    disease_status VARCHAR(100),
    metadata JSONB,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Multi-omics data files
CREATE TABLE IF NOT EXISTS omics_data (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    patient_id UUID REFERENCES patients(id),
    data_type VARCHAR(20) NOT NULL, -- genomics, transcriptomics, proteomics
    file_path VARCHAR(500) NOT NULL,
    file_format VARCHAR(10) NOT NULL, -- vcf, bam, csv, etc.
    file_size BIGINT,
    checksum VARCHAR(64),
    processing_status VARCHAR(20) DEFAULT 'uploaded', -- uploaded, processing, processed, failed
    quality_metrics JSONB,
    metadata JSONB,
    uploaded_by UUID REFERENCES users(id),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Drug information
CREATE TABLE IF NOT EXISTS drugs (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    drug_id VARCHAR(50) UNIQUE NOT NULL,
    drug_name VARCHAR(100) NOT NULL,
    drug_class VARCHAR(100),
    mechanism_of_action TEXT,
    target_pathway VARCHAR(200),
    metadata JSONB,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Drug response data
CREATE TABLE IF NOT EXISTS drug_responses (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    patient_id UUID REFERENCES patients(id),
    drug_id UUID REFERENCES drugs(id),
    response_value FLOAT NOT NULL,
    response_type VARCHAR(50) NOT NULL, -- IC50, AUC, efficacy_score, etc.
    measurement_unit VARCHAR(20),
    experimental_conditions JSONB,
    metadata JSONB,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Machine learning models
CREATE TABLE IF NOT EXISTS ml_models (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    model_name VARCHAR(100) NOT NULL,
    model_version VARCHAR(20) NOT NULL,
    model_type VARCHAR(50) NOT NULL, -- multi_branch_fusion, genomics_cnn, etc.
    architecture_config JSONB,
    hyperparameters JSONB,
    training_data_info JSONB,
    performance_metrics JSONB,
    model_path VARCHAR(500),
    is_active BOOLEAN DEFAULT false,
    trained_by UUID REFERENCES users(id),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(model_name, model_version)
);

-- Prediction results
CREATE TABLE IF NOT EXISTS predictions (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    prediction_id VARCHAR(100) UNIQUE NOT NULL,
    patient_id UUID REFERENCES patients(id),
    drug_id UUID REFERENCES drugs(id),
    model_id UUID REFERENCES ml_models(id),
    predicted_response FLOAT NOT NULL,
    confidence_score FLOAT,
    prediction_intervals JSONB, -- confidence intervals
    feature_importance JSONB,
    shap_values JSONB,
    biomarkers JSONB,
    prediction_status VARCHAR(20) DEFAULT 'completed',
    requested_by UUID REFERENCES users(id),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Analysis jobs
CREATE TABLE IF NOT EXISTS analysis_jobs (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    job_id VARCHAR(100) UNIQUE NOT NULL,
    job_type VARCHAR(50) NOT NULL, -- data_processing, model_training, prediction
    job_status VARCHAR(20) DEFAULT 'queued', -- queued, running, completed, failed
    input_parameters JSONB,
    output_data JSONB,
    error_message TEXT,
    progress_percentage INTEGER DEFAULT 0,
    started_at TIMESTAMP,
    completed_at TIMESTAMP,
    requested_by UUID REFERENCES users(id),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- System logs
CREATE TABLE IF NOT EXISTS system_logs (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    log_level VARCHAR(10) NOT NULL,
    logger_name VARCHAR(100),
    message TEXT NOT NULL,
    exception_info TEXT,
    user_id UUID REFERENCES users(id),
    request_id VARCHAR(100),
    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Performance metrics
CREATE TABLE IF NOT EXISTS performance_metrics (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    metric_name VARCHAR(100) NOT NULL,
    metric_value FLOAT NOT NULL,
    metric_unit VARCHAR(50),
    tags JSONB,
    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Create indexes for better performance
CREATE INDEX IF NOT EXISTS idx_omics_data_patient_type ON omics_data(patient_id, data_type);
CREATE INDEX IF NOT EXISTS idx_drug_responses_patient ON drug_responses(patient_id);
CREATE INDEX IF NOT EXISTS idx_predictions_patient_drug ON predictions(patient_id, drug_id);
CREATE INDEX IF NOT EXISTS idx_analysis_jobs_status ON analysis_jobs(job_status);
CREATE INDEX IF NOT EXISTS idx_system_logs_timestamp ON system_logs(timestamp);
CREATE INDEX IF NOT EXISTS idx_performance_metrics_name_timestamp ON performance_metrics(metric_name, timestamp);

-- Insert some initial data
INSERT INTO drugs (drug_id, drug_name, drug_class, mechanism_of_action) VALUES
('erlotinib', 'Erlotinib', 'EGFR Inhibitor', 'Selective inhibition of EGFR tyrosine kinase'),
('gefitinib', 'Gefitinib', 'EGFR Inhibitor', 'Reversible EGFR tyrosine kinase inhibitor'),
('imatinib', 'Imatinib', 'BCR-ABL Inhibitor', 'Selective inhibition of BCR-ABL kinase'),
('trastuzumab', 'Trastuzumab', 'HER2 Inhibitor', 'Monoclonal antibody targeting HER2'),
('paclitaxel', 'Paclitaxel', 'Taxane', 'Microtubule stabilization agent')
ON CONFLICT (drug_id) DO NOTHING;

-- Create functions for updating timestamps
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = CURRENT_TIMESTAMP;
    RETURN NEW;
END;
$$ language 'plpgsql';

-- Create triggers for automatic timestamp updates
CREATE TRIGGER update_users_updated_at BEFORE UPDATE ON users FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();
CREATE TRIGGER update_patients_updated_at BEFORE UPDATE ON patients FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();
CREATE TRIGGER update_omics_data_updated_at BEFORE UPDATE ON omics_data FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

-- Grant permissions (adjust as needed for your deployment)
-- GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA public TO multiomics_user;
-- GRANT ALL PRIVILEGES ON ALL SEQUENCES IN SCHEMA public TO multiomics_user;
