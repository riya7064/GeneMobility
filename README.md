# Data-Driven Analysis of Antibiotic Resistance Gene Mobility for Predicting Horizontal Gene Transfer Risk

---

## Abstract

The rise of antimicrobial resistance (AMR) poses a global threat, largely driven by the horizontal gene transfer (HGT) of antibiotic resistance genes (ARGs) between bacteria. Mobile genetic elements such as plasmids, integrons, and transposons enable the transfer of ARGs across environmental, clinical, and agricultural ecosystems, leading to treatment-resistant infections.

This project performs a data-driven computational analysis of antibiotic resistance genes using genomic and metadata obtained from the Comprehensive Antibiotic Resistance Database (CARD). Protein sequences, gene families, resistance mechanisms, and ontology metadata are processed to quantify attributes linked to mobility. Machine learning is applied to cluster and classify genes based on mobility potential, identifying high-risk ARGs capable of transferring across bacterial species.

The project outputs a ranked list of "high-risk mobile genes" and visual mobility maps, contributing to global AMR surveillance and drug discovery research.

---

## Problem Statement

Antibiotic resistance is escalating worldwide due to the spread of resistance genes between bacteria, not just the evolution of resistance within a single organism. However:

- There is no simple pipeline to analyze mobility potential of resistance genes
- ARG datasets are fragmented across different databases
- Predicting whether a gene is mobile or non-mobile requires integrating sequence data, ontology, and metadata

**Key Question:** Which antibiotic resistance genes demonstrate the highest mobility potential and thus pose the greatest risk for horizontal gene transfer?

---

## Objective

To create a data-driven system that identifies and predicts the mobility potential of antibiotic resistance genes using genomic features and machine learning.

### Specific Objectives

1. Collect and preprocess genomic ARG data from CARD
2. Extract biological and metadata features such as:
   - Resistance class
   - Host organism
   - Protein sequence
   - Mechanism of action
3. Apply clustering to discover patterns of mobility
4. Classify genes into:
   - High-mobility (HGT-prone)
   - Low-mobility (genomically stable)
5. Visualize and interpret the biological significance of high-risk mobile genes

---

## Methodology

| Step | Description |
|------|-------------|
| **1. Data Acquisition** | Download latest CARD release (protein sequences, ontology, gene metadata) |
| **2. Data Preprocessing** | Clean dataset, merge JSON/TSV metadata, normalize categorical values, extract features |
| **3. Feature Engineering** | Compute mobility-related properties (gene family, plasmid association, host organisms) |
| **4. Machine Learning / Analytics** | - Unsupervised: K-Means / Hierarchical clustering<br>- Optional supervised: XGBoost classifier |
| **5. Visualization** | Mobility heatmaps, gene embeddings (PCA / UMAP), cluster plots |
| **6. Interpretation & Insights** | Identify high-risk ARGs and evaluate biological patterns |

---

## Architecture Diagram

```
+---------------------+
| CARD Database       |
| (Sequences + Meta)  |
+----------+----------+
           |
           v
+---------------------+
| Data Preprocessing  |
| (JSON/TSV + FASTA)  |
+----------+----------+
           |
           v
+-----------------------------+
| Feature Engineering         |
| - Ontology features         |
| - Sequence embeddings       |
| - Gene mobility features    |
+----------+------------------+
           |
           v
+-----------------------------+
| Machine Learning            |
| - Clustering (K-Means)      |
| - Classification (XGBoost)  |
+----------+------------------+
           |
           v
+-----------------------------+
| Results & Visualization     |
| - High-risk mobile ARG list |
| - Mobility heatmaps         |
| - ARG clustering map        |
+-----------------------------+
```

---

## Expected Outcomes

- Ranked list of genes likely to undergo horizontal transfer
- Mobility visualization of bacterial species vs ARGs
- Clustering map grouping genes by mobility pattern
- Insight into which mechanisms or antibiotic classes are most mobile

---

## Conclusion

This project demonstrates how data science and genomics can be used to address one of the most critical global health challenges: antimicrobial resistance. By analyzing the genomic features and mobility characteristics of resistance genes, we can:

- Detect emerging high-risk genes before they become widespread
- Support policymakers and researchers in AMR surveillance
- Accelerate antibiotic stewardship and drug development programs

The integration of biological data with machine learning provides an automated and scalable solution for monitoring the mobility of antibiotic resistance genes.

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
