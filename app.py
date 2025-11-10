import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import torch
import esm
import joblib
from sklearn.metrics.pairwise import cosine_similarity
import warnings

warnings.filterwarnings('ignore')

# ===========================
# PAGE CONFIGURATION
# ===========================
st.set_page_config(
    page_title="ARG Mobility Predictor",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ===========================
# CUSTOM CSS STYLING
# ===========================
st.markdown("""
    <style>
    .prediction-box {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 15px;
        color: white;
        text-align: center;
        box-shadow: 0 10px 30px rgba(0,0,0,0.2);
        margin: 2rem 0;
    }
    
    .non-mobile-box {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        padding: 2rem;
        border-radius: 15px;
        color: white;
        text-align: center;
        box-shadow: 0 10px 30px rgba(0,0,0,0.2);
        margin: 2rem 0;
    }
    
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.3);
        border-left: 4px solid #ffd700;
        color: white;
    }
    
    .info-box {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 4px solid #ffd700;
        margin: 1rem 0;
        color: white;
    }
    
    .stButton>button {
        background: linear-gradient(120deg, #1e3c72 0%, #2a5298 100%);
        color: white;
        font-weight: 600;
        border: none;
        border-radius: 8px;
        padding: 0.75rem 2rem;
        font-size: 1.1rem;
        transition: transform 0.2s;
    }
    
    .stButton>button:hover {
        transform: scale(1.05);
    }
    </style>
""", unsafe_allow_html=True)

# ===========================
# HEADER SECTION
# ===========================
st.markdown('<div class="prediction-box"><h1>🧬 ARG Mobility Prediction System</h1><p>AI-Powered Prediction of Antibiotic Resistance Gene Mobility<br>Using ESM Protein Language Models & XGBoost Machine Learning</p></div>', unsafe_allow_html=True)

# ===========================
# SIDEBAR INFORMATION
# ===========================
with st.sidebar:
    st.image("https://img.icons8.com/color/96/000000/dna.png", width=80)
    
    st.markdown("### 📊 Model Information")
    st.markdown("""
    **Technology Stack:**
    - 🧠 ESM-2 (650M params)
    - 🌲 XGBoost Classifier
    - 🎯 99% Accuracy
    - 📦 CARD Database
    
    **Prediction Classes:**
    - ✅ Mobile ARG
    - ❌ Non-Mobile ARG
    """)
    
    st.markdown("---")
    st.markdown("### 🔬 About")
    st.markdown("""
    This system predicts the mobility potential
    of antibiotic resistance genes using
    state-of-the-art protein language models.
    
    **Developed by:**
    Genomics & AI Lab
    """)

# ===========================
# MODEL LOADING FUNCTIONS
# ===========================
@st.cache_resource
def load_models():
    """Load ML model and ESM model (cached for performance)"""
    model = joblib.load(r"C:\Users\riyar\Desktop\Torah\Outputs\mobility_predictor_xgb.pkl")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    esm_model, alphabet = esm.pretrained.esm2_t33_650M_UR50D()
    esm_model = esm_model.to(device)
    esm_model.eval()
    return model, esm_model, alphabet, device

@st.cache_data
def load_embeddings():
    """Load CARD embeddings database"""
    return pd.read_pickle(r"C:\Users\riyar\Desktop\Torah\Outputs\card_embeddings.pkl")

# ===========================
# PREDICTION FUNCTION
# ===========================
def predict_mobility(sequence, model, esm_model, alphabet, device, df_emb):
    """Predict ARG mobility from protein sequence"""
    batch_converter = alphabet.get_batch_converter()
    batch_labels, batch_strs, batch_tokens = batch_converter([("query", sequence)])
    batch_tokens = batch_tokens.to(device)
    
    with torch.no_grad():
        results = esm_model(batch_tokens, repr_layers=[33])
        token_reps = results["representations"][33]
        embedding = token_reps.mean(1).cpu().numpy()
    
    # Ensure embedding is 2D for XGBoost
    if embedding.ndim == 1:
        embedding = embedding.reshape(1, -1)
    
    prob = model.predict_proba(embedding)[0][1]
    label = "Mobile ARG" if prob > 0.5 else "Non-Mobile ARG"
    
    sims = cosine_similarity(embedding, np.vstack(df_emb["embedding"].values))
    nearest_idx = np.argmax(sims)
    nearest_gene = df_emb.iloc[nearest_idx]
    similarity_score = sims[0][nearest_idx]
    
    return label, prob, nearest_gene, similarity_score

# ===========================
# MAIN APPLICATION TABS
# ===========================
tab1, tab2, tab3 = st.tabs(["🔍 Predict Mobility", "📊 Batch Analysis", "📚 About the Model"])

# ===========================
# TAB 1: SINGLE PREDICTION
# ===========================
with tab1:
    st.markdown('<div class="info-box"><h3>Enter Protein Sequence for Mobility Prediction</h3></div>', unsafe_allow_html=True)
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("""
        📝 **Instructions:**
        
        Enter your protein sequence in FASTA format or as a raw amino acid sequence.
        The system will analyze the sequence and predict its mobility potential.
        """)
        
        input_method = st.radio("Choose Input Method:", ["Direct Sequence Entry", "Example Sequence"])
        
        if input_method == "Direct Sequence Entry":
            sequence_input = st.text_area(
                "Protein Sequence:",
                height=200,
                placeholder="Enter amino acid sequence (e.g., MKQKNPKNTQNFITSKKHVK...)",
                help="Paste your protein sequence here. FASTA headers will be automatically removed."
            )
        else:
            st.info("📌 Using example sequence: 23S rRNA methyltransferase Erm(A)")
            sequence_input = """MKQKNPKNTQNFITSKKHVKEILKYTNINKQDKIIEIGSGKGHFTK 
ELISQIRKNFSKESYDLILVIGSGNGGQTILSLMKFNQKVIIISSPPYI
KMLVKAINSTYGLSKQILENLGIKHISFSVKKGNIDAFYTSFISHHDIK
LVNKGTLIEKLKEGITSSDYFKNEDSAIYMDLAKPGGIPMVGISSKYIK
EYQKMGYDIAMTVLEMAHHWCSLKEAEIAVVTDSHIKIIDNRTPK"""
        
        predict_btn = st.button("🚀 Predict Mobility", use_container_width=True)
    
    with col2:
        st.markdown("### ⚙️ Model Status")
        device_type = "🎮 GPU (CUDA)" if torch.cuda.is_available() else "💻 CPU"
        st.success(f"Device: {device_type}")
        
        st.markdown("### 📈 Model Metrics")
        metrics_col1, metrics_col2 = st.columns(2)
        with metrics_col1:
            st.metric("Accuracy", "99%", delta="High")
        with metrics_col2:
            st.metric("Precision", "100%", delta="Perfect")
    
    if predict_btn:
        if not sequence_input or sequence_input.strip() == "":
            st.error("⚠️ Please enter a protein sequence!")
        else:
            sequence = sequence_input.replace("\n", "").replace(" ", "")
            if sequence.startswith(">"):
                sequence = "".join(sequence.split("\n")[1:])
            sequence = sequence.upper()
            
            valid_aa = set("ACDEFGHIKLMNPQRSTVWY")
            if not all(aa in valid_aa for aa in sequence):
                st.error("⚠️ Invalid amino acid sequence! Please check your input.")
            else:
                with st.spinner("🔬 Analyzing sequence with ESM model..."):
                    try:
                        model, esm_model, alphabet, device = load_models()
                        df_emb = load_embeddings()
                        
                        label, prob, nearest_gene, similarity = predict_mobility(
                            sequence, model, esm_model, alphabet, device, df_emb
                        )
                        
                        st.markdown("---")
                        st.markdown('<div class="info-box"><h3>🎯 Prediction Results</h3></div>', unsafe_allow_html=True)
                        
                        if label == "Mobile ARG":
                            st.markdown(f"""
                            <div class="prediction-box">
                                <h2>✅ {label}</h2>
                                <h3>{prob*100:.2f}% Mobility Probability</h3>
                                <p>This gene is predicted to have HIGH mobility potential<br>
                                and may be transferable via horizontal gene transfer.</p>
                            </div>
                            """, unsafe_allow_html=True)
                        else:
                            st.markdown(f"""
                            <div class="non-mobile-box">
                                <h2>❌ {label}</h2>
                                <h3>{prob*100:.2f}% Mobility Probability</h3>
                                <p>This gene is predicted to have LOW mobility potential<br>
                                and is likely chromosomally stable.</p>
                            </div>
                            """, unsafe_allow_html=True)
                        
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("Sequence Length", f"{len(sequence)} aa")
                        with col2:
                            st.metric("Mobility Score", f"{prob:.4f}")
                        with col3:
                            st.metric("Confidence", f"{max(prob, 1-prob)*100:.1f}%")
                        
                        st.markdown("### 📊 Mobility Probability Distribution")
                        fig = go.Figure(go.Indicator(
                            mode="gauge+number+delta",
                            value=prob * 100,
                            domain={'x': [0, 1], 'y': [0, 1]},
                            title={'text': "Mobility Probability", 'font': {'size': 24}},
                            delta={'reference': 50, 'increasing': {'color': "red"}},
                            gauge={
                                'axis': {'range': [None, 100], 'tickwidth': 1, 'tickcolor': "darkblue"},
                                'bar': {'color': "darkblue"},
                                'bgcolor': "white",
                                'borderwidth': 2,
                                'bordercolor': "gray",
                                'steps': [
                                    {'range': [0, 50], 'color': '#f5576c'},
                                    {'range': [50, 100], 'color': '#667eea'}
                                ],
                                'threshold': {
                                    'line': {'color': "red", 'width': 4},
                                    'thickness': 0.75,
                                    'value': 50
                                }
                            }
                        ))
                        fig.update_layout(height=300, paper_bgcolor="white", font={'color': "darkblue", 'family': "Arial"})
                        st.plotly_chart(fig, use_container_width=True)
                        
                        st.markdown("### 🔎 Most Similar ARG in CARD Database")
                        st.markdown(f"""
                        <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                                    padding: 1.5rem; 
                                    border-radius: 10px; 
                                    border-left: 4px solid #ffd700;
                                    color: white;
                                    margin: 1rem 0;
                                    box-shadow: 0 4px 6px rgba(0,0,0,0.3);">
                            <div style="font-size: 1.1rem; line-height: 1.8;">
                                🧬 <b>Gene Name:</b> {nearest_gene['ARO Name']}<br>
                                👨‍👩‍👧‍👦 <b>Gene Family:</b> {nearest_gene['AMR Gene Family']}<br>
                                💊 <b>Drug Class:</b> {nearest_gene['Drug Class']}<br>
                                ⚙️ <b>Resistance Mechanism:</b> {nearest_gene['Resistance Mechanism']}<br>
                                📏 <b>Similarity Score:</b> <span style="color: #ffd700; font-weight: bold;">{similarity*100:.2f}%</span>
                            </div>
                        </div>
                        """, unsafe_allow_html=True)
                        
                    except Exception as e:
                        st.error(f"❌ Error during prediction: {str(e)}")
                        st.info("Please ensure model files are in the same directory.")

# ===========================
# TAB 2: BATCH ANALYSIS
# ===========================
with tab2:
    st.markdown('<div class="info-box"><h3>📂 Upload Multiple Sequences for Batch Analysis</h3></div>', unsafe_allow_html=True)
    st.markdown("""
    **Upload a CSV file with columns:**
    - `sequence`: Protein sequence
    - `gene_name` (optional): Gene identifier
    """)
    
    uploaded_file = st.file_uploader("Choose CSV file", type=['csv'])
    
    if uploaded_file is not None:
        df_batch = pd.read_csv(uploaded_file)
        st.success(f"✅ Loaded {len(df_batch)} sequences")
        
        if st.button("🚀 Run Batch Prediction"):
            with st.spinner("Processing all sequences..."):
                try:
                    model, esm_model, alphabet, device = load_models()
                    df_emb = load_embeddings()
                    results = []
                    progress_bar = st.progress(0)
                    
                    for idx, row in df_batch.iterrows():
                        seq = row['sequence']
                        label, prob, nearest_gene, similarity = predict_mobility(
                            seq, model, esm_model, alphabet, device, df_emb
                        )
                        results.append({
                            'Gene': row.get('gene_name', f'Seq_{idx}'),
                            'Length': len(seq),
                            'Prediction': label,
                            'Probability': prob,
                            'Nearest_Match': nearest_gene['ARO Name'],
                            'Similarity': similarity
                        })
                        progress_bar.progress((idx + 1) / len(df_batch))
                    
                    df_results = pd.DataFrame(results)
                    
                    st.markdown("### 📊 Batch Prediction Results")
                    st.dataframe(df_results, use_container_width=True)
                    
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Total Sequences", len(df_results))
                    with col2:
                        mobile_count = len(df_results[df_results['Prediction'] == 'Mobile ARG'])
                        st.metric("Mobile ARGs", mobile_count)
                    with col3:
                        st.metric("Non-Mobile ARGs", len(df_results) - mobile_count)
                    
                    fig = px.histogram(df_results, x='Probability', nbins=20,
                                     title='Mobility Probability Distribution', color='Prediction')
                    st.plotly_chart(fig, use_container_width=True)
                    
                    csv = df_results.to_csv(index=False)
                    st.download_button(
                        label="📥 Download Results as CSV",
                        data=csv,
                        file_name="arg_mobility_predictions.csv",
                        mime="text/csv"
                    )
                except Exception as e:
                    st.error(f"Error: {str(e)}")

# ===========================
# TAB 3: ABOUT THE MODEL
# ===========================
with tab3:
    st.markdown('<div class="info-box"><h3>🧠 Model Architecture & Methodology</h3></div>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        ### 🔬 ESM-2 Protein Language Model
        **Model Details:**
        - Architecture: ESM-2 (650M parameters)
        - Training Data: 250M protein sequences
        - Embedding Size: 1280 dimensions
        - Layer: 33 (final representation layer)
        
        **Features:**
        - Captures protein structure & function
        - No manual feature engineering required
        - State-of-the-art performance
        """)
        
        st.markdown("""
        ### 🌲 XGBoost Classifier
        **Hyperparameters:**
        - Estimators: 200
        - Max Depth: 8
        - Learning Rate: 0.05
        - Subsample: 0.7
        
        **Training Dataset:**
        - Total: 6,053 ARGs
        - Mobile: 4,950 (82%)
        - Non-Mobile: 1,103 (18%)
        """)
    
    with col2:
        st.markdown("""
        ### 📊 Performance Metrics
        **Classification Report:**
        """)
        metrics_df = pd.DataFrame({
            'Class': ['Non-Mobile', 'Mobile', 'Overall'],
            'Precision': [0.96, 1.00, 0.99],
            'Recall': [0.99, 0.99, 0.99],
            'F1-Score': [0.98, 0.99, 0.99]
        })
        st.dataframe(metrics_df, use_container_width=True)
        
        st.markdown("""
        ### 🎯 Prediction Workflow
        1. **Input**: Protein amino acid sequence
        2. **Embedding**: ESM-2 generates 1280-D vector
        3. **Classification**: XGBoost predicts mobility
        4. **Similarity**: Cosine similarity with CARD database
        5. **Output**: Mobility prediction + nearest match
        """)
        
        st.markdown("""
        ### 📚 Data Source
        **CARD Database:**
        - Comprehensive Antibiotic Resistance Database
        - 6,000+ curated ARGs
        - Metadata: Gene family, drug class, mechanism
        - URL: https://card.mcmaster.ca/
        """)

# ===========================
# FOOTER
# ===========================
st.markdown("---")
st.markdown("""
<div style="text-align: center; padding: 2rem;">
    <h4>🧬 ARG Mobility Prediction System v1.0 | Powered by ESM-2 & XGBoost</h4>
    <p>📧 Contact: genomics-lab@research.edu | 🌐 GitHub: GeneMobility</p>
</div>
""", unsafe_allow_html=True)