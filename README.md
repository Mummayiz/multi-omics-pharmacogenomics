# Multi-Omics Pharmacogenomics Platform

A comprehensive platform for analyzing multi-omics data and predicting drug responses using machine learning. This platform integrates genomics, transcriptomics, proteomics, and drug response data to provide personalized medicine insights.

## 🚀 Features

### Data Processing
- **Flexible Data Upload**: Supports CSV files with any number of columns (3-20+)
- **Multi-Omics Integration**: Genomics, Transcriptomics, Proteomics, Drug Response
- **Automatic Data Validation**: Robust error handling and data type conversion
- **Real-time Processing**: Background data processing with progress tracking

### Machine Learning
- **Multi-Omics Fusion Models**: Combines different omics data types
- **Drug Response Prediction**: Predicts patient response to specific drugs
- **Biomarker Discovery**: Identifies key biomarkers and pathways
- **Model Interpretability**: SHAP values and feature importance analysis

### User Interface
- **Modern Web Interface**: Clean, responsive design
- **Interactive Visualizations**: Charts and graphs for data exploration
- **Real-time Updates**: Live progress tracking and notifications
- **Individual Upload Buttons**: Pre-selected data type uploads

## 🏗️ Architecture

### Backend (FastAPI)
- **API Layer**: RESTful endpoints for all operations
- **Data Processing**: Flexible pipeline for different omics data types
- **Machine Learning**: Lightweight ML models using scikit-learn
- **Database**: SQLite for data persistence
- **Background Tasks**: Asynchronous data processing

### Frontend (Vanilla HTML/CSS/JS)
- **Responsive Design**: Works on desktop and mobile
- **Interactive Charts**: Chart.js and Plotly.js visualizations
- **Real-time Communication**: AJAX calls to backend API
- **User-friendly Interface**: Intuitive upload and analysis workflows

## 📊 Supported Data Types

### Genomics Data
- **VCF Files**: Variant call format files
- **CSV Format**: Patient variants with columns like:
  - `patient_id`, `gene_id`, `chromosome`, `position`
  - `ref_allele`, `alt_allele`, `genotype`
  - `allele_frequency`, `quality_score`, `read_depth`
  - `consequence`, `impact`, `clinical_significance`

### Transcriptomics Data
- **Gene Expression**: RNA-seq data
- **CSV Format**: Genes as rows, samples as columns
- **Normalization**: TPM, RPKM, log2 transformation
- **Quality Control**: Low expression filtering

### Proteomics Data
- **Protein Abundance**: Mass spectrometry data
- **CSV Format**: Proteins as rows, samples as columns
- **Missing Value Imputation**: KNN imputation
- **Normalization**: Median centering, quantile normalization

### Drug Response Data
- **Clinical Data**: Patient drug responses
- **CSV Format**: Patient ID, drug name, response score
- **Response Prediction**: Binary and continuous outcomes

## 🛠️ Installation

### Prerequisites
- Python 3.8+
- pip (Python package manager)

### Quick Start

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/multi-omics-pharmacogenomics-platform.git
   cd multi-omics-pharmacogenomics-platform
   ```

2. **Install dependencies**
   ```bash
   pip install -r backend/requirements.txt
   ```

3. **Start the backend**
   ```bash
   cd backend
   python main.py
   ```

4. **Start the frontend** (in a new terminal)
   ```bash
   cd frontend
   python -m http.server 3000
   ```

5. **Open in browser**
   ```
   http://localhost:3000
   ```

## 📁 Project Structure

```
multi-omics-pharmacogenomics-platform/
├── backend/                    # FastAPI backend
│   ├── api/                   # API routes
│   ├── data_processing/       # Data processing pipeline
│   ├── database/              # Database layer
│   ├── models/                # ML models
│   ├── utils/                 # Utilities
│   └── main.py               # Main application
├── frontend/                  # Web frontend
│   ├── css/                  # Stylesheets
│   ├── js/                   # JavaScript
│   └── index.html            # Main page
├── data/                     # Data storage
│   ├── raw/                  # Raw uploaded data
│   └── processed/            # Processed data
├── models/                   # Trained models
│   └── saved/                # Saved model files
├── logs/                     # Application logs
├── scripts/                  # Utility scripts
└── docs/                     # Documentation
```

## 🧪 Sample Data

The repository includes sample datasets for testing:

- `sample_genomics_data.csv` - 15-column genomics data
- `sample_genomics_20cols.csv` - 20-column genomics data
- `sample_transcriptomics_5cols.csv` - 5-column transcriptomics data
- `sample_proteomics_8cols.csv` - 8-column proteomics data
- `sample_drug_response_3cols.csv` - 3-column drug response data

## 🔧 Usage

### 1. Upload Data
- Click "Upload Data" or use individual data type buttons
- Select your CSV file
- Choose data type (genomics, transcriptomics, proteomics, drug_response)
- Enter patient ID
- Click Upload

### 2. Train Models
- Go to "Models" section
- Configure model parameters
- Select data types to include
- Click "Start Training"

### 3. Make Predictions
- Go to "Analysis" section
- Enter patient ID
- Select drug
- Choose omics data types
- Click "Predict Response"

### 4. Explore Results
- View prediction scores and confidence
- Explore biomarker importance
- Analyze SHAP values
- Discover key pathways

## 🧬 API Endpoints

### Data Management
- `POST /api/v1/omics/upload` - Upload omics data
- `GET /api/v1/omics/datasets` - List available datasets
- `GET /api/v1/omics/patients/{patient_id}/data` - Get patient data

### Model Training
- `POST /api/v1/models/train` - Train new models
- `GET /api/v1/models/training/{job_id}/status` - Check training status
- `GET /api/v1/models/architectures` - List available model types

### Analysis
- `POST /api/v1/analysis/predict` - Make drug response predictions
- `POST /api/v1/analysis/explain` - Get prediction explanations
- `GET /api/v1/analysis/biomarkers` - Discover biomarkers

### System
- `GET /api/v1/health` - Health check
- `GET /api/v1/omics/status` - System status

## 🔬 Scientific Background

This platform implements state-of-the-art methods for:

- **Multi-omics Integration**: Combines genomic, transcriptomic, and proteomic data
- **Pharmacogenomics**: Predicts drug responses based on genetic variants
- **Biomarker Discovery**: Identifies key molecular features
- **Pathway Analysis**: Explores biological pathways and mechanisms

## 📈 Performance

- **Flexible Data Handling**: Supports any number of columns (3-20+)
- **Real-time Processing**: Background data processing
- **Scalable Architecture**: Handles large datasets efficiently
- **Error Recovery**: Robust error handling and fallbacks

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 🙏 Acknowledgments

- Built with FastAPI, scikit-learn, and modern web technologies
- Inspired by the latest research in multi-omics and pharmacogenomics
- Designed for both researchers and clinicians

**Made with ❤️ for advancing personalized medicine through multi-omics analysis**
