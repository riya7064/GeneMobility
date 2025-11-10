# 🧬 ARG Mobility Prediction System# 🧬 ARG Mobility Prediction System# 🧬 ARG Mobility Prediction System# 🧬 ARG Mobility Prediction System



**AI-Powered Prediction of Antibiotic Resistance Gene Mobility using ESM Protein Language Models & XGBoost**



[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/)**AI-Powered Prediction of Antibiotic Resistance Gene Mobility using ESM Protein Language Models & XGBoost**

[![PyTorch](https://img.shields.io/badge/PyTorch-2.5.1-red.svg)](https://pytorch.org/)

[![Streamlit](https://img.shields.io/badge/Streamlit-1.51.0-green.svg)](https://streamlit.io/)

[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/)**AI-Powered Prediction of Antibiotic Resistance Gene Mobility using ESM Protein Language Models & XGBoost****AI-Powered Prediction of Antibiotic Resistance Gene Mobility using ESM Protein Language Models & XGBoost**

---

[![PyTorch](https://img.shields.io/badge/PyTorch-2.5.1-red.svg)](https://pytorch.org/)

## 🎯 Overview

[![Streamlit](https://img.shields.io/badge/Streamlit-1.51.0-green.svg)](https://streamlit.io/)

The rise of antimicrobial resistance (AMR) poses a global threat, largely driven by the horizontal gene transfer (HGT) of antibiotic resistance genes (ARGs) between bacteria. This project combines state-of-the-art **ESM-2 protein language models** with **XGBoost machine learning** to predict whether an ARG is mobile (transferable) or non-mobile (chromosomally stable).

[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

### Key Features

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/)[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/)

- 🤖 **ESM-2 (650M parameters)** - Meta's protein language model for sequence embeddings

- 🌲 **XGBoost Classifier** - Gradient boosting with 99% accuracy---

- 🎨 **Interactive Streamlit UI** - Real-time predictions with beautiful visualizations

- 📊 **CARD Database** - 6,053 curated antibiotic resistance genes[![PyTorch](https://img.shields.io/badge/PyTorch-2.5.1-red.svg)](https://pytorch.org/)[![PyTorch](https://img.shields.io/badge/PyTorch-2.5.1-red.svg)](https://pytorch.org/)

- 🔍 **Similarity Search** - Find nearest matching genes in database

- 📈 **Batch Processing** - Analyze multiple sequences at once## 🎯 Overview



---[![Streamlit](https://img.shields.io/badge/Streamlit-1.51.0-green.svg)](https://streamlit.io/)[![Streamlit](https://img.shields.io/badge/Streamlit-1.51.0-green.svg)](https://streamlit.io/)



## 🚀 Quick StartThe rise of antimicrobial resistance (AMR) poses a global threat, largely driven by the horizontal gene transfer (HGT) of antibiotic resistance genes (ARGs) between bacteria. This project combines state-of-the-art **ESM-2 protein language models** with **XGBoost machine learning** to predict whether an ARG is mobile (transferable) or non-mobile (chromosomally stable).



### Prerequisites[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)



- Python 3.10+### Key Features

- CUDA-capable GPU (optional, for faster inference)

- 8GB+ RAM---



### Installation- 🤖 **ESM-2 (650M parameters)** - Meta's protein language model for sequence embeddings



```bash- 🌲 **XGBoost Classifier** - Gradient boosting with 99% accuracy---

# Clone the repository

git clone https://github.com/riya7064/GeneMobility.git- 🎨 **Interactive Streamlit UI** - Real-time predictions with beautiful visualizations

cd GeneMobility

- 📊 **CARD Database** - 6,053 curated antibiotic resistance genes## 🎯 Overview

# Install dependencies

pip install -r requirements.txt- 🔍 **Similarity Search** - Find nearest matching genes in database

```

- 📈 **Batch Processing** - Analyze multiple sequences at once## 🎯 Overview

### Run the Web Application



```bash

# Set environment variable---The rise of antimicrobial resistance (AMR) poses a global threat, largely driven by horizontal gene transfer (HGT) of antibiotic resistance genes (ARGs) between bacteria. This project combines **ESM-2 protein language models** with **XGBoost machine learning** to predict whether an ARG is mobile or non-mobile.

$env:KMP_DUPLICATE_LIB_OK="TRUE"  # Windows PowerShell

# OR

export KMP_DUPLICATE_LIB_OK=TRUE  # Linux/Mac

## 🚀 Quick StartThe rise of antimicrobial resistance (AMR) poses a global threat, largely driven by the horizontal gene transfer (HGT) of antibiotic resistance genes (ARGs) between bacteria. This project combines state-of-the-art **ESM-2 protein language models** with **XGBoost machine learning** to predict whether an ARG is mobile (transferable) or non-mobile (chromosomally stable).

# Launch Streamlit app

streamlit run app.py

```

### Prerequisites### Key Features

Open your browser and navigate to **http://localhost:8501**



---

- Python 3.10+### Key Features

## 📊 Model Architecture

- CUDA-capable GPU (optional, for faster inference)

### ESM-2 Protein Language Model

- 8GB+ RAM- 🤖 **ESM-2 (650M parameters)** - Meta's protein language model

- **Architecture**: Transformer-based (650M parameters)

- **Training Data**: 250M protein sequences

- **Embedding Size**: 1,280 dimensions

- **Layer Used**: Layer 33 (final representation)### Installation- 🌲 **XGBoost Classifier** - 99% accuracy- 🤖 **ESM-2 (650M parameters)** - Meta's protein language model for sequence embeddings



### XGBoost Classifier



- **Algorithm**: Gradient Boosting Decision Trees```bash- 🎨 **Interactive UI** - Real-time predictions- 🌲 **XGBoost Classifier** - Gradient boosting with 99% accuracy

- **Hyperparameters**:

  - `n_estimators`: 200# Clone the repository

  - `max_depth`: 8

  - `learning_rate`: 0.05git clone https://github.com/riya7064/GeneMobility.git- 📊 **6,053 ARGs** from CARD Database- 🎨 **Interactive Streamlit UI** - Real-time predictions with beautiful visualizations

  - `subsample`: 0.7

cd GeneMobility

### Performance Metrics

- 🔍 **Similarity Search** - Find nearest genes- 📊 **CARD Database** - 6,053 curated antibiotic resistance genes

```

Accuracy:   99.01%# Install dependencies

Precision:  100% (Mobile ARGs)

Recall:     99% (Mobile ARGs)pip install -r requirements.txt- 📈 **Batch Processing** - Analyze multiple sequences- 🔍 **Similarity Search** - Find nearest matching genes in database

F1-Score:   0.99

``````



---- 📈 **Batch Processing** - Analyze multiple sequences at once



## 💡 Usage Examples### Run the Web Application



### Single Sequence Prediction---



```python```bash

# Example: 23S rRNA methyltransferase Erm(A)

sequence = "MKQKNPKNTQNFITSKKHVKEILKYTNINKQDKIIEIGSGKGHFTK..."# Set environment variable---



# Prediction: Non-Mobile ARG (39.22% probability)$env:KMP_DUPLICATE_LIB_OK="TRUE"  # Windows PowerShell

```

# OR## 🚀 Quick Start

### Batch Analysis

export KMP_DUPLICATE_LIB_OK=TRUE  # Linux/Mac

Upload a CSV file with columns:

- `sequence`: Protein amino acid sequence## 🚀 Quick Start

- `gene_name` (optional): Gene identifier

# Launch Streamlit app

---

streamlit run app.py### Installation

## 📁 Project Structure

```

```

GeneMobility/### Prerequisites

├── app.py                              # Streamlit web application

├── requirements.txt                    # Python dependenciesOpen your browser and navigate to **http://localhost:8501**

├── Notebooks/                          # Jupyter notebooks

│   ├── ARG_Exploratory_Data_Analysis.ipynb```bash

│   └── ARG_Mobility_Prediction_Model.ipynb

├── Outputs/                            # Model files and results---

│   ├── mobility_predictor_xgb.pkl     # Trained XGBoost model

│   ├── card_embeddings.pkl            # Pre-computed ESM embeddingsgit clone https://github.com/riya7064/GeneMobility.git- Python 3.10+

│   └── ARG_mobility_results.csv       # Prediction results

└── card-data/                          # CARD database files## 📊 Model Architecture

    ├── card.json

    ├── aro_index.tsvcd GeneMobility- CUDA-capable GPU (optional, for faster inference)

    └── protein_fasta_protein_*.fasta

```### ESM-2 Protein Language Model



---- **Architecture**: Transformer-based (650M parameters)pip install -r requirements.txt- 8GB+ RAM



## 🛠️ Technologies- **Training Data**: 250M protein sequences



- **Python 3.10+** - Core programming language- **Embedding Size**: 1,280 dimensions```

- **PyTorch 2.5.1** - Deep learning framework

- **ESM-2** - Protein language model by Meta AI- **Layer Used**: Layer 33 (final representation)

- **XGBoost 3.1.1** - Gradient boosting framework

- **Streamlit 1.51.0** - Web application framework### Installation

- **Plotly** - Interactive visualizations

- **Pandas & NumPy** - Data manipulation### XGBoost Classifier



---- **Algorithm**: Gradient Boosting Decision Trees### Run Application



## 🤝 Contributing- **Hyperparameters**:



Pull requests welcome! Please submit issues for bugs/features.  - `n_estimators`: 200```bash



---  - `max_depth`: 8



## 📄 Citation  - `learning_rate`: 0.05```bash# Clone the repository



```bibtex  - `subsample`: 0.7

@software{arg_mobility_2025,

  title = {ARG Mobility Prediction System},$env:KMP_DUPLICATE_LIB_OK="TRUE"  # Windowsgit clone https://github.com/riya7064/GeneMobility.git

  author = {Riya},

  year = {2025},### Performance Metrics

  url = {https://github.com/riya7064/GeneMobility}

}```streamlit run app.pycd GeneMobility

```

Accuracy:   99.01%

---

Precision:  100% (Mobile ARGs)```

## 📧 Contact

Recall:     99% (Mobile ARGs)

For questions or collaborations, please open an issue on GitHub.

F1-Score:   0.99# Install dependencies

---

```

## 🙏 Acknowledgments

Navigate to **http://localhost:8501**pip install -r requirements.txt

- **CARD Database** - Comprehensive Antibiotic Resistance Database

- **Meta AI** - ESM protein language models---

- **XGBoost Team** - Machine learning framework

- **Streamlit** - Web app framework```



---## 💡 Usage Examples



**⚠️ Disclaimer**: This tool is for research purposes only. Clinical decisions should be made by qualified professionals.---


### Single Sequence Prediction

### Run the Web Application

```python

# Example: 23S rRNA methyltransferase Erm(A)## 📊 Model Performance

sequence = "MKQKNPKNTQNFITSKKHVKEILKYTNINKQDKIIEIGSGKGHFTK..."

```bash

# Prediction: Non-Mobile ARG (39.22% probability)

``````# Set environment variable



### Batch AnalysisAccuracy:   99.01%$env:KMP_DUPLICATE_LIB_OK="TRUE"  # Windows PowerShell



```pythonPrecision:  100%# OR

# Upload CSV with columns: gene_name, sequence

# Download results with mobility predictionsRecall:     99%export KMP_DUPLICATE_LIB_OK=TRUE  # Linux/Mac

```

F1-Score:   0.99

---

```# Launch Streamlit app

## 📁 Project Structure

streamlit run app.py

```

GeneMobility/---```

├── app.py                              # Streamlit web application

├── requirements.txt                    # Python dependencies

├── Notebooks/                          # Jupyter notebooks

│   ├── ARG_Exploratory_Data_Analysis.ipynb## 💡 UsageOpen your browser and navigate to **http://localhost:8501**

│   └── ARG_Mobility_Prediction_Model.ipynb

├── Outputs/                            # Model files and results

│   ├── mobility_predictor_xgb.pkl     # Trained XGBoost model

│   ├── card_embeddings.pkl            # Pre-computed ESM embeddings1. **Single Prediction**: Enter amino acid sequence---

│   └── ARG_mobility_results.csv       # Prediction results

└── card-data/                          # CARD database files2. **Batch Analysis**: Upload CSV with sequences

    ├── card.json

    ├── aro_index.tsv3. **View Results**: Mobility prediction + similarity search## 📊 Model Architecture

    └── protein_fasta_protein_*.fasta

```



------### ESM-2 Protein Language Model



## 🛠️ Technologies- **Architecture**: Transformer-based (650M parameters)



- **Python 3.10+** - Core programming language## 📁 Project Structure- **Training Data**: 250M protein sequences

- **PyTorch 2.5.1** - Deep learning framework

- **ESM-2** - Protein language model by Meta AI- **Embedding Size**: 1,280 dimensions

- **XGBoost 3.1.1** - Gradient boosting framework

- **Streamlit 1.51.0** - Web application framework```- **Layer Used**: Layer 33 (final representation)

- **Plotly** - Interactive visualizations

- **Pandas & NumPy** - Data manipulationGeneMobility/



---├── app.py                    # Streamlit web app### XGBoost Classifier



## 🤝 Contributing├── requirements.txt          # Dependencies- **Algorithm**: Gradient Boosting Decision Trees



Pull requests welcome! Please submit issues for bugs/features.├── Notebooks/               # ML pipeline- **Hyperparameters**:



---├── Outputs/                 # Models & embeddings  - `n_estimators`: 200



## 📄 Citation└── card-data/              # CARD database  - `max_depth`: 8



```bibtex```  - `learning_rate`: 0.05

@software{arg_mobility_2025,

  title = {ARG Mobility Prediction System},  - `subsample`: 0.7

  author = {Riya},

  year = {2025},---

  url = {https://github.com/riya7064/GeneMobility}

}### Performance Metrics

```

## 🛠️ Technologies```

---

Accuracy:   99.01%

## 📧 Contact

- Python 3.10+ | PyTorch 2.5.1 | ESM-2 | XGBoost 3.1.1Precision:  100% (Mobile ARGs)

For questions or collaborations, please open an issue on GitHub.

- Streamlit 1.51.0 | Plotly | Pandas | NumPyRecall:     99% (Mobile ARGs)

---

F1-Score:   0.99

## 🙏 Acknowledgments

---```

- **CARD Database** - Comprehensive Antibiotic Resistance Database

- **Meta AI** - ESM protein language models

- **XGBoost Team** - Machine learning framework

- **Streamlit** - Web app framework## 🤝 Contributing---



---



**⚠️ Disclaimer**: This tool is for research purposes only. Clinical decisions should be made by qualified professionals.Pull requests welcome! Please submit issues for bugs/features.## 💡 Usage Examples




---### Single Sequence Prediction



## 📄 Citation```python

# Example: 23S rRNA methyltransferase Erm(A)

```bibtexsequence = "MKQKNPKNTQNFITSKKHVKEILKYTNINKQDKIIEIGSGKGHFTK..."

@software{arg_mobility_2025,

  title = {ARG Mobility Prediction System},# Prediction: Non-Mobile ARG (39.22% probability)

  author = {Riya},```

  year = {2025},

  url = {https://github.com/riya7064/GeneMobility}### Batch Analysis

}

```Upload a CSV file with columns:

- `sequence`: Protein amino acid sequence

---- `gene_name` (optional): Gene identifier



## 🙏 Acknowledgments---



- **CARD Database** - McMaster University## 📁 Project Structure

- **Meta AI** - ESM-2 model

- **XGBoost Community**```

- **Streamlit**GeneMobility/

├── app.py                          # Streamlit web application

---├── requirements.txt                # Python dependencies

├── README.md                       # Project documentation

<div align="center">├── Notebooks/

│   └── ARG_Mobility_Prediction_Model.ipynb  # ML training pipeline

**🧬 Fighting AMR Through AI-Powered Genomics 🧬**├── Outputs/

│   ├── mobility_predictor_xgb.pkl  # Trained XGBoost model

[GitHub](https://github.com/riya7064/GeneMobility) • [Issues](https://github.com/riya7064/GeneMobility/issues)│   └── card_embeddings.pkl         # Pre-computed ESM embeddings

└── card-data/                      # CARD database files

</div>```


---
| **1. Data Acquisition** | Download latest CARD release (protein sequences, ontology, gene metadata) |
| **2. Data Preprocessing** | Clean dataset, merge JSON/TSV metadata, normalize categorical values, extract features |
| **3. Feature Engineering** | Compute mobility-related properties (gene family, plasmid association, host organisms) |
| **4. Protein Embeddings** | Generate 1280-dimensional embeddings using ESM-2 (650M parameter protein language model) |
| **5. Mobility Labeling** | Heuristic scoring based on sequence length, resistance mechanism, and gene family |
| **6. Machine Learning** | XGBoost classifier trained on ESM embeddings (99% accuracy, 100% precision) |
| **7. Visualization** | Mobility heatmaps, gene embeddings (PCA / UMAP), cluster plots, interactive gauge charts |
| **8. Deployment** | Streamlit web application with real-time prediction and batch analysis |

---

## Models & Technologies Used

### 🧠 Machine Learning Models

#### 1. **ESM-2 (Evolutionary Scale Modeling)**
- **Architecture**: Protein language model with 650M parameters
- **Training Data**: 250M protein sequences from UniRef
- **Embedding Size**: 1280 dimensions
- **Layer Used**: Layer 33 (final representation)
- **Purpose**: Generate contextualized protein sequence embeddings capturing structural and functional information
- **Provider**: Meta AI (Facebook)
- **Installation**: `pip install fair-esm`

#### 2. **XGBoost Classifier**
- **Algorithm**: Gradient Boosted Decision Trees
- **Hyperparameters**:
  - Estimators: 200
  - Max Depth: 8
  - Learning Rate: 0.05
  - Subsample: 0.7
  - Random State: 42
- **Training Dataset**: 
  - Total: 6,053 ARGs
  - Mobile: 4,950 (82%)
  - Non-Mobile: 1,103 (18%)
  - Split: 80% train, 20% test
- **Performance Metrics**:
  - Overall Accuracy: **99%**
  - Precision (Mobile): **100%**
  - Recall (Mobile): **99%**
  - F1-Score: **0.99**
- **Purpose**: Predict mobility potential from ESM embeddings
- **Model File**: `mobility_predictor_xgb.pkl`

### 🛠️ Technology Stack

#### Core Libraries
- **PyTorch 2.5.1+cu121**: Deep learning framework with CUDA support
- **fair-esm 2.0.0**: ESM protein language model implementation
- **XGBoost 3.1.1**: Gradient boosting library
- **scikit-learn**: Machine learning utilities and metrics
- **joblib**: Model serialization

#### Data Processing
- **pandas**: Data manipulation and analysis
- **numpy**: Numerical computing
- **BioPython**: Biological sequence processing

#### Visualization
- **Streamlit 1.51.0**: Web application framework
- **plotly**: Interactive visualizations (gauge charts, histograms)
- **matplotlib & seaborn**: Static plotting

#### Database
- **CARD (Comprehensive Antibiotic Resistance Database)**: 
  - Version: Latest release
  - ARGs Analyzed: 6,053 genes
  - Embeddings File: `card_embeddings.pkl`

### 🚀 Deployment

#### Web Application Features
1. **Single Sequence Prediction**
   - Text area input for amino acid sequences
   - FASTA format support
   - Real-time GPU/CPU detection
   - Mobility probability gauge
   - Nearest similar ARG matching using cosine similarity

2. **Batch Analysis**
   - CSV file upload
   - Progress bar for multiple sequences
   - Results table with predictions
   - Downloadable CSV export
   - Histogram visualization

3. **Model Documentation**
   - Architecture details
   - Performance metrics
   - Prediction workflow
   - CARD database information

#### Hardware Requirements
- **Minimum**: CPU-only (slower predictions)
- **Recommended**: NVIDIA GPU with CUDA support (faster inference)
- **RAM**: 8GB minimum, 16GB recommended
- **Storage**: 5GB for model weights and embeddings

#### Running the Application
```bash
# Activate environment
conda activate gpu-env

# Set environment variable (Windows)
$env:KMP_DUPLICATE_LIB_OK="TRUE"

# Run Streamlit app
streamlit run app.py
```

#### Access URLs
- Local: `http://localhost:8502`
- Network: `http://10.91.247.182:8502`

---

## Architecture Diagram

```
+-------------------------+
| CARD Database           |
| (6,053 ARGs)            |
| - Sequences             |
| - Metadata              |
| - Ontology              |
+----------+--------------+
           |
           v
+-------------------------+
| Data Preprocessing      |
| - JSON/TSV parsing      |
| - FASTA processing      |
| - Feature extraction    |
+----------+--------------+
           |
           v
+-------------------------+
| ESM-2 Model (650M)      |
| - Protein embeddings    |
| - 1280-D vectors        |
| - Layer 33 output       |
+----------+--------------+
           |
           v
+-------------------------+
| Feature Engineering     |
| - Mobility scoring      |
| - Heuristic labeling    |
| - Train/test split      |
+----------+--------------+
           |
           v
+-------------------------+
| XGBoost Classifier      |
| - 200 estimators        |
| - 99% accuracy          |
| - Binary classification |
+----------+--------------+
           |
           v
+-------------------------+
| Streamlit Web App       |
| - Single prediction     |
| - Batch analysis        |
| - Interactive UI        |
+----------+--------------+
           |
           v
+-------------------------+
| Results & Outputs       |
| - Mobility probability  |
| - Nearest ARG match     |
| - Gauge visualizations  |
| - Downloadable reports  |
+-------------------------+
```

---

## Expected Outcomes

### 📊 Model Performance
- **Classification Accuracy**: 99%
- **Precision (Mobile ARGs)**: 100%
- **Recall (Mobile ARGs)**: 99%
- **F1-Score**: 0.99
- **Prediction Speed**: 
  - Single sequence: ~15-20 seconds (GPU)
  - Batch processing: ~0.5-1 minute per sequence

### 🎯 Key Deliverables
1. **Trained ML Model**
   - `mobility_predictor_xgb.pkl` - XGBoost classifier
   - `card_embeddings.pkl` - Pre-computed ESM embeddings database

2. **Interactive Web Application**
   - Real-time ARG mobility prediction
   - Batch sequence analysis
   - Similarity search against CARD database
   - Downloadable prediction reports

3. **Mobility Insights**
   - Ranked list of high-risk mobile ARGs
   - Probability scores for horizontal gene transfer
   - Nearest similar genes with metadata
   - Confidence scores for predictions

4. **Visualizations**
   - Mobility probability gauge charts
   - Batch analysis histograms
   - PCA/UMAP clustering maps (in notebooks)
   - Heatmaps of gene families vs. mobility

### 🔬 Scientific Impact
- **Early Detection**: Identify emerging mobile ARGs before widespread dissemination
- **Risk Assessment**: Quantify HGT potential for any ARG sequence
- **Database Enhancement**: Automated annotation pipeline for new resistance genes
- **Clinical Relevance**: Support antibiotic stewardship and treatment decisions
- **Research Acceleration**: High-throughput screening of metagenomic datasets

---

## Installation & Setup

### Prerequisites
- Python 3.10 or higher
- Conda or pip package manager
- CUDA-compatible GPU (optional, for faster inference)

### Step 1: Clone Repository
```bash
git clone https://github.com/riya7064/GeneMobility.git
cd GeneMobility
```

### Step 2: Create Environment
```bash
# Using conda (recommended)
conda create -n arg-mobility python=3.10
conda activate arg-mobility

# Or using venv
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### Step 3: Install Dependencies
```bash
pip install -r requirements.txt
```

**Key Dependencies:**
```txt
torch>=2.0.0
fair-esm==2.0.0
xgboost>=3.0.0
streamlit>=1.30.0
pandas>=2.0.0
numpy>=1.24.0
scikit-learn>=1.3.0
plotly>=5.17.0
biopython>=1.81
joblib>=1.3.0
```

### Step 4: Download Model Files
Ensure these files are in the project directory:
- `mobility_predictor_xgb.pkl` (XGBoost trained model)
- `card_embeddings.pkl` (Pre-computed CARD embeddings)

### Step 5: Run Application
```bash
# Set environment variable (Windows)
$env:KMP_DUPLICATE_LIB_OK="TRUE"

# Set environment variable (Linux/Mac)
export KMP_DUPLICATE_LIB_OK=TRUE

# Launch Streamlit app
streamlit run app.py
```

Access the app at: `http://localhost:8501`

---

## Usage Guide

### 🔍 Single Sequence Prediction

1. Navigate to the **"Predict Mobility"** tab
2. Choose input method:
   - **Direct Entry**: Paste amino acid sequence
   - **Example Sequence**: Use pre-loaded test sequence
3. Click **"Predict Mobility"** button
4. View results:
   - Mobility classification (Mobile/Non-Mobile)
   - Probability score (0-100%)
   - Confidence level
   - Nearest similar ARG in CARD database
   - Interactive gauge visualization

**Example Input:**
```
MKQKNPKNTQNFITSKKHVKEILKYTNINKQDKIIEIGSGKGHFTK
ELISQIRKNFSKESYDLILVIGSGNGGQTILSLMKFNQKVIIISSPPYI
```

### 📊 Batch Analysis

1. Navigate to the **"Batch Analysis"** tab
2. Prepare CSV file with columns:
   - `sequence`: Protein amino acid sequence (required)
   - `gene_name`: Gene identifier (optional)
3. Upload CSV file
4. Click **"Run Batch Prediction"**
5. View results table with predictions
6. Download results as CSV

**Example CSV Format:**
```csv
gene_name,sequence
ARG001,MKTIIALSYIFCLVFA...
ARG002,MKQKNPKNTQNFITSK...
```

### 📚 Model Information

Navigate to **"About the Model"** tab to view:
- ESM-2 architecture details
- XGBoost hyperparameters
- Performance metrics
- Prediction workflow
- CARD database information

---

## Project Structure

```
GeneMobility/
├── app.py                          # Streamlit web application
├── antibiotic-resistance-gene-mobility-eda.ipynb  # EDA notebook
├── Antibiotic Resistance Gene Mobility.ipynb       # ML training notebook
├── mobility_predictor_xgb.pkl     # Trained XGBoost model
├── card_embeddings.pkl            # Pre-computed ESM embeddings
├── requirements.txt               # Python dependencies
├── .gitignore                     # Git ignore patterns
├── README.md                      # Project documentation
└── card-data/                     # CARD database files
    ├── card.json
    ├── aro_index.tsv
    ├── aro_categories.tsv
    ├── protein_fasta_protein_homolog_model.fasta
    └── ...
```

---

## Conclusion

This project demonstrates how state-of-the-art AI and genomics can address antimicrobial resistance (AMR), one of the most critical global health challenges. By combining:

- **Protein Language Models (ESM-2)**: Capturing complex structural and functional patterns in resistance genes
- **Machine Learning (XGBoost)**: Achieving 99% accuracy in mobility prediction
- **Interactive Deployment (Streamlit)**: Making predictions accessible to researchers worldwide

We provide an automated, scalable solution for:

✅ **Early Detection** - Identifying emerging mobile ARGs before widespread dissemination  
✅ **Risk Quantification** - Assigning mobility probability scores to any ARG sequence  
✅ **Clinical Support** - Informing antibiotic stewardship and treatment strategies  
✅ **Research Acceleration** - Enabling high-throughput screening of metagenomic data  

### Future Enhancements
- 🌐 Integration with real-time genomic surveillance systems
- 🧬 Multi-species transfer prediction models
- 📱 Mobile application for field deployment
- 🔄 Continuous learning from newly discovered ARGs
- 🗺️ Geographic spread visualization and tracking

### Citation
If you use this work, please cite:
```
Antibiotic Resistance Gene Mobility Prediction System
GitHub: https://github.com/riya7064/GeneMobility
```

### Contact & Contributions
- **Repository**: https://github.com/riya7064/GeneMobility
- **Issues**: Report bugs or request features via GitHub Issues
- **Contributions**: Pull requests welcome!

---

## Acknowledgments

- **CARD Database**: Comprehensive Antibiotic Resistance Database team at McMaster University
- **ESM-2**: Meta AI for the protein language model
- **XGBoost**: Distributed Machine Learning Community
- **Streamlit**: For the excellent web framework

---

## License

This project is available for academic and research purposes. Please review the license file for commercial use restrictions.

---

**🧬 Fighting Antimicrobial Resistance Through AI-Powered Genomics**

---

## CARD Database Files Reference

> **Comprehensive Antibiotic Resistance Database (CARD)**  
> A curated resource for antibiotic resistance genes, mutations, and drug mappings.

### ```diff``` Essential Files (Required)

| File                   | Type              | Description                                                      |
| ---------------------- | ----------------- | ---------------------------------------------------------------- |
| `card.json`            | **Main database** | Complete database: genes, mutations, drug mappings, and ontology |
| `aro_index.tsv`        | Mapping table     | Links gene IDs to names and antibiotic classes                  |
| `aro_categories.tsv`   | Mapping table     | Groups genes by resistance categories (β-lactamase, efflux, etc.) |

### ```css``` Protein Sequence Files

| File                                               | Use Case                               |
| -------------------------------------------------- | -------------------------------------- |
| `protein_fasta_protein_homolog_model.fasta`        | ML/deep learning on protein sequences  |
| `protein_fasta_protein_variant_model.fasta`        | Mutation-focused analysis              |
| `protein_fasta_protein_knockout_model.fasta`       | Specific research (small dataset)      |
| `protein_fasta_protein_overexpression_model.fasta` | Specific research (small dataset)      |

### ```yaml``` Nucleotide Sequence Files

| File Pattern            | Use Case                          |
| ----------------------- | --------------------------------- |
| `nucleotide_fasta_*`    | DNA-based models and analysis     |

*Use only if working with DNA sequences; otherwise skip.*

### ```json``` Mutation & Mapping Files

| File                        | Type          | Use Case                                      |
| --------------------------- | ------------- | --------------------------------------------- |
| `snps.txt`                  | Mutation data | Mutation → resistance prediction projects     |
| `shortname_antibiotics.tsv` | Lookup table  | Convert long drug names to short names        |
| `shortname_pathogens.tsv`   | Lookup table  | Pathogen name conversions                     |

---

## Getting Started

1. **Start with the essentials**: `card.json`, `aro_index.tsv`, and `aro_categories.tsv`
2. **Add protein sequences** if building ML models on sequences
3. **Include mutation files** if predicting resistance from genetic variants
4. **Use lookup tables** for data normalization and standardization

---

## File Overview by Use Case

### For Machine Learning Projects
- `card.json`
- `aro_index.tsv`
- `aro_categories.tsv`
- `protein_fasta_protein_homolog_model.fasta`

### For Mutation Analysis
- `card.json`
- `aro_index.tsv`
- `protein_fasta_protein_variant_model.fasta`
- `snps.txt`

### For Basic Exploration
- `card.json`
- `aro_index.tsv`
- `aro_categories.tsv`

---

## Additional Resources

- `CARD-Download-README.txt` - Official download documentation
- `PMID.tsv` - PubMed reference mappings
