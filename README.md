# 🧬 ARG Mobility Prediction System

**AI-Powered Prediction of Antibiotic Resistance Gene Mobility using ESM Protein Language Models & XGBoost**

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.5.1-red.svg)](https://pytorch.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.51.0-green.svg)](https://streamlit.io/)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

---

## 🎯 Overview

The rise of antimicrobial resistance (AMR) poses a global threat, largely driven by the horizontal gene transfer (HGT) of antibiotic resistance genes (ARGs) between bacteria. This project combines state-of-the-art **ESM-2 protein language models** with **XGBoost machine learning** to predict whether an ARG is mobile (transferable) or non-mobile (chromosomally stable).

### Key Features

- 🤖 **ESM-2 (650M parameters)** - Meta's protein language model for sequence embeddings
- 🌲 **XGBoost Classifier** - Gradient boosting with 99% accuracy
- 🎨 **Interactive Streamlit UI** - Real-time predictions with beautiful visualizations
- 📊 **CARD Database** - 6,053 curated antibiotic resistance genes
- 🔍 **Similarity Search** - Find nearest matching genes in database
- 📈 **Batch Processing** - Analyze multiple sequences at once

---

## 🚀 Quick Start

### Prerequisites

- Python 3.10+
- CUDA-capable GPU (optional, for faster inference)
- 8GB+ RAM

### Installation

```bash
# Clone the repository
git clone https://github.com/riya7064/GeneMobility.git
cd GeneMobility

# Install dependencies
pip install -r requirements.txt
```

### Run the Web Application

```bash
# Set environment variable
$env:KMP_DUPLICATE_LIB_OK="TRUE"  # Windows PowerShell
# OR
export KMP_DUPLICATE_LIB_OK=TRUE  # Linux/Mac

# Launch Streamlit app
streamlit run app.py
```

Open your browser and navigate to **http://localhost:8501**

---

## 📊 Model Architecture

### ESM-2 Protein Language Model

- **Architecture**: Transformer-based (650M parameters)
- **Training Data**: 250M protein sequences
- **Embedding Size**: 1,280 dimensions
- **Layer Used**: Layer 33 (final representation)

### XGBoost Classifier

- **Algorithm**: Gradient Boosting Decision Trees
- **Hyperparameters**:
  - `n_estimators`: 200
  - `max_depth`: 8
  - `learning_rate`: 0.05
  - `subsample`: 0.7

### Performance Metrics

```
Accuracy:  99.01%
Precision: 100% (Mobile ARGs)
Recall:    99% (Mobile ARGs)
F1-Score:  0.99
```

---

## 💡 Usage Examples

### Single Sequence Prediction

```python
# Example: 23S rRNA methyltransferase Erm(A)
sequence = "MKQKNPKNTQNFITSKKHVKEILKYTNINKQDKIIEIGSGKGHFTK..."

# Prediction: Non-Mobile ARG (39.22% probability)
```

### Batch Analysis

Upload a CSV file with columns:
- `sequence`: Protein amino acid sequence
- `gene_name` (optional): Gene identifier

---

## 📁 Project Structure

```
GeneMobility/
├── app.py                          # Streamlit web application
├── requirements.txt                # Python dependencies
├── Notebooks/                      # Jupyter notebooks
│   ├── ARG_Exploratory_Data_Analysis.ipynb
│   └── ARG_Mobility_Prediction_Model.ipynb
├── Outputs/                        # Model files and results
│   ├── mobility_predictor_xgb.pkl  # Trained XGBoost model
│   ├── card_embeddings.pkl         # Pre-computed ESM embeddings
│   └── ARG_mobility_results.csv    # Prediction results
└── card-data/                      # CARD database files
    ├── card.json
    ├── aro_index.tsv
    └── protein_fasta_protein_*.fasta
```

---

## 🛠️ Technologies

- **Python 3.10+** - Core programming language
- **PyTorch 2.5.1** - Deep learning framework
- **ESM-2** - Protein language model by Meta AI
- **XGBoost 3.1.1** - Gradient boosting framework
- **Streamlit 1.51.0** - Web application framework
- **Plotly** - Interactive visualizations
- **Pandas & NumPy** - Data manipulation

---

## 🤝 Contributing

Pull requests welcome! Please submit issues for bugs/features.

---

## 📄 Citation

```bibtex
@software{arg_mobility_2025,
  title = {ARG Mobility Prediction System},
  author = {Riya},
  year = {2025},
  url = {https://github.com/riya7064/GeneMobility}
}
```

---

## 📧 Contact

For questions or collaborations, please open an issue on GitHub.

---

## 🙏 Acknowledgments

- **CARD Database** - Comprehensive Antibiotic Resistance Database
- **Meta AI** - ESM protein language models
- **XGBoost Team** - Machine learning framework
- **Streamlit** - Web app framework

---

**⚠️ Disclaimer**: This tool is for research purposes only. Clinical decisions should be made by qualified professionals.

---

## 📚 ML Pipeline Overview

| Step | Description |
|------|-------------|
| **1. Data Acquisition** | Download latest CARD release (protein sequences, ontology, gene metadata) |
| **2. Data Preprocessing** | Clean dataset, merge JSON/TSV metadata, normalize categorical values, extract features |
| **3. Feature Engineering** | Compute mobility-related properties (gene family, plasmid association, host organisms) |
| **4. Protein Embeddings** | Generate 1280-dimensional embeddings using ESM-2 (650M parameter protein language model) |
| **5. Mobility Labeling** | Heuristic scoring based on sequence length, resistance mechanism, and gene family |
| **6. Machine Learning** | XGBoost classifier trained on ESM embeddings (99% accuracy, 100% precision) |
| **7. Visualization** | Mobility heatmaps, gene embeddings (PCA / UMAP), cluster plots, interactive gauge charts |
| **8. Deployment** | Streamlit web application with real-time prediction and batch analysis |

---

## 🎯 Expected Outcomes

### Model Performance
- **Classification Accuracy**: 99%
- **Precision (Mobile ARGs)**: 100%
- **Recall (Mobile ARGs)**: 99%
- **F1-Score**: 0.99
- **Prediction Speed**: ~15-20 seconds per sequence (GPU)

### Key Deliverables

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
   - PCA/UMAP clustering maps
   - Heatmaps of gene families vs. mobility

---

## 🔬 Scientific Impact

- **Early Detection**: Identify emerging mobile ARGs before widespread dissemination
- **Risk Assessment**: Quantify HGT potential for any ARG sequence
- **Database Enhancement**: Automated annotation pipeline for new resistance genes
- **Clinical Relevance**: Support antibiotic stewardship and treatment decisions
- **Research Acceleration**: High-throughput screening of metagenomic datasets

---

## 📖 CARD Database Files Reference

> **Comprehensive Antibiotic Resistance Database (CARD)**  
> A curated resource for antibiotic resistance genes, mutations, and drug mappings.

### Essential Files (Required)

| File | Type | Description |
|------|------|-------------|
| `card.json` | Main database | Complete database: genes, mutations, drug mappings, and ontology |
| `aro_index.tsv` | Mapping table | Links gene IDs to names and antibiotic classes |
| `aro_categories.tsv` | Mapping table | Groups genes by resistance categories (β-lactamase, efflux, etc.) |

### Protein Sequence Files

| File | Use Case |
|------|----------|
| `protein_fasta_protein_homolog_model.fasta` | ML/deep learning on protein sequences |
| `protein_fasta_protein_variant_model.fasta` | Mutation-focused analysis |
| `protein_fasta_protein_knockout_model.fasta` | Specific research (small dataset) |
| `protein_fasta_protein_overexpression_model.fasta` | Specific research (small dataset) |

---

## 🏗️ Architecture Diagram

```
+-------------------------+
|    CARD Database        |
|     (6,053 ARGs)        |
|   - Sequences           |
|   - Metadata            |
|   - Ontology            |
+----------+--------------+
           |
           v
+-------------------------+
|  Data Preprocessing     |
|  - JSON/TSV parsing     |
|  - FASTA processing     |
|  - Feature extraction   |
+----------+--------------+
           |
           v
+-------------------------+
|   ESM-2 Model (650M)    |
|  - Protein embeddings   |
|  - 1280-D vectors       |
|  - Layer 33 output      |
+----------+--------------+
           |
           v
+-------------------------+
|  Feature Engineering    |
|  - Mobility scoring     |
|  - Heuristic labeling   |
|  - Train/test split     |
+----------+--------------+
           |
           v
+-------------------------+
|   XGBoost Classifier    |
|  - 200 estimators       |
|  - 99% accuracy         |
|  - Binary classification|
+----------+--------------+
           |
           v
+-------------------------+
|   Streamlit Web App     |
|  - Single prediction    |
|  - Batch analysis       |
|  - Interactive UI       |
+----------+--------------+
           |
           v
+-------------------------+
|   Results & Outputs     |
|  - Mobility probability |
|  - Nearest ARG match    |
|  - Gauge visualizations |
|  - Downloadable reports |
+-------------------------+
```

---

## 🚦 Usage Guide

### Single Sequence Prediction

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

### Batch Analysis

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

---

## 🔮 Future Enhancements

- 🌐 Integration with real-time genomic surveillance systems
- 🧬 Multi-species transfer prediction models
- 📱 Mobile application for field deployment
- 🔄 Continuous learning from newly discovered ARGs
- 🗺️ Geographic spread visualization and tracking

---

**🧬 Fighting AMR Through AI-Powered Genomics 🧬**

[GitHub](https://github.com/riya7064/GeneMobility) • [Issues](https://github.com/riya7064/GeneMobility/issues)