# Computational discovery of novel antibacterial candidates against *Staphylococcus aureus*: integrating ML-QSAR models and molecular *docking* on the marine CMNPD chemical library

Official code repository and computational workflows developed for the **Master's Thesis (TFM)** in Bioinformatics (2025-2026 academic year).

| | |
|---|---|
| **Author** | Jaime Peláez Sánchez |
| **Thesis Advisor** | Jose María Guijarro Blanco |
| **Degree** | Master's Degree in Bioinformatics |
| **Institution** | Universidad Internacional de Valencia (VIU) |
| **License** | [MIT](#7-license-and-citation) |

---

## Table of contents

1. [Background and rationale](#1-background-and-rationale)
2. [Methodological workflow (pipeline overview)](#2-methodological-workflow-pipeline-overview)
3. [Repository structure](#3-repository-structure)
4. [Installation and requirements](#4-installation-and-requirements)
5. [Step-by-step execution guide](#5-step-by-step-execution-guide)
6. [Lead compound: enisorine E](#6-lead-compound-enisorine-e)
7. [License and citation](#7-license-and-citation)

---

## 1. Background and rationale

The rapid spread of antimicrobial resistance among ESKAPE pathogens — with particular virulence in nosocomial and community-acquired methicillin-resistant *Staphylococcus aureus* (MRSA) strains — poses a critical threat to public health. Given the high attrition rates, long timelines, and prohibitive costs associated with conventional empirical biological screening, cheminformatics emerges as a rational and scalable alternative.

This project implements a hybrid computational workflow (*ligand-based* and *structure-based*) designed to explore the marine chemical library **CMNPD** (*Comprehensive Marine Natural Products Database*). The goal is to identify and prioritize selective molecular inhibitors of the enoyl-ACP reductase I enzyme (**saFabI**), a key regulatory node in bacterial fatty acid biosynthesis (FAS-II pathway).

---

## 2. Methodological workflow (pipeline overview)

The workflow is structured in a modular, automated, and sequential manner across six Python scripts:

```text
                          [ChEMBL data]
                                │
                                ▼
        01_data_curation.py ─────────────────► Curation, outlier removal,
                                │                and chemical standardization (RDKit)
                                ▼
    02_feature_extraction.py ─────────────────► NCATS physicochemical descriptors
                                │                and MACCS Keys + filtering
                                ▼
        03_train_qsar.py ─────────────────────► Training, optimization
                                │                (GridSearchCV), and validation (Random Forest)
                                ▼
  04_virtual_screening.py  ◄──────────────────  [CMNPD chemical library (.sdf)]
                                │                Massive screening and strict consensus
                                │                (pChEMBL ≥ 6.0)
                                ▼
05_druggability_and_scaffolds.py ─────────────► Ro5 filter (flexible Lipinski) and
                                │                structural diversity (Bemis-Murcko)
                                ▼
    06_prepare_ligands_3d.py ─────────────────► 3D generation (ETKDG) and energy
                                                 minimization (MMFF94) for docking
```

---

## 3. Repository structure

```text
saFabI-ml-qsar-docking/
├── data/
│   ├── raw/                                # Original, untouched data
│   │   ├── IC50_FabI.csv                   # Raw bioactivity records downloaded from ChEMBL
│   │   └── cmnpd-07-2026.sdf               # CMNPD conformational database
│   └── processed/                          # Curated intermediate data and matrices
│       ├── chembl_curado.csv
│       ├── fisicoquimicos_filtrados.csv
│       ├── maccs_filtrados.csv
│       ├── candidatos_cmnpd_fisicoquimicos.csv
│       └── candidatos_cmnpd_maccs.csv
├── models/                                 # Serialized Machine Learning models
│   ├── rf_fisicoquimico.joblib
│   └── rf_maccs.joblib
├── results/                                # Final tables, 3D structures, and plots
│   ├── figures/                            # Analytical diagrams and plots (300 DPI)
│   ├── candidatos_consenso_top.csv         # Molecules with consensus pChEMBL ≥ 6.0
│   ├── CMNPD_Lipinski_Flexible.csv         # Molecules passing the Ro5 screen
│   ├── CMNPD_Representantes_Elite.csv      # Leading molecules per structural family
│   └── Top_*_Pred_*.sdf                    # Minimized 3D conformers
├── src/                                    # Modular source code (.py)
│   ├── 01_data_curation.py
│   ├── 02_feature_extraction.py
│   ├── 03_train_qsar.py
│   ├── 04_virtual_screening.py
│   ├── 05_druggability_and_scaffolds.py
│   └── 06_prepare_ligands_3d.py
├── environment.yml                         # Reproducible dependency specification
├── .gitignore                              # Git exclusion rules
└── README.md                               # Technical documentation
```

> **Note:** file and folder names inside `data/`, `models/`, and `results/` are kept in Spanish to exactly match the actual filenames produced by the scripts in this repository.

---

## 4. Installation and requirements

It is recommended to create an isolated virtual environment with `conda` or `mamba` on Linux, macOS, or Windows (WSL2):

```bash
# 1. Clone the repository
git clone https://github.com/your-username/saFabI-ml-qsar-docking.git
cd saFabI-ml-qsar-docking

# 2. Build and activate the working environment
conda env create -f environment.yml
conda activate safabi-env
```

Main packages and libraries used:

| Package | Minimum version | Role in the pipeline |
|---|---|---|
| Python | 3.10 | Runtime environment |
| RDKit | 2023.09 | Chemical standardization, molecular fingerprints, and 3D modeling |
| scikit-learn | 1.3 | *Random Forest* regression models and cross-validation |
| pandas / NumPy | — | Data structure handling and numerical computation |
| SciPy / Matplotlib / Seaborn | — | Statistical analysis and figure generation |
| joblib | — | Parallel computation and estimator serialization |

---

## 5. Step-by-step execution guide

The scripts are designed to be run sequentially from the project's root directory.

### Step 1 — Chemical curation and standardization

```bash
python src/01_data_curation.py
```

- **Input:** `data/raw/IC50_FabI.csv`
- **Methodology:** filters bioactivities against *S. aureus* with an exact quantitative relationship (`=`). Applies RDKit's chemical standardization protocol (general cleanup, salt removal via `FragmentParent`, formal charge neutralization via `Uncharger`, and tautomer canonicalization). Applies a duplicate-handling procedure, discarding outlier measurements via the interquartile range (1.5 × IQR) before computing the mean pChEMBL affinity.
- **Output:** `data/processed/chembl_curado.csv` and descriptive plots in `results/figures/`.

### Step 2 — Molecular feature extraction and selection

```bash
python src/02_feature_extraction.py
```

- **Input:** `data/processed/chembl_curado.csv`
- **Methodology:** computes, in parallel, 78 physicochemical descriptors from the NCATS subset and 167 binary MACCS Keys structural fingerprints. Applies a two-stage dimensionality reduction: a low-variance filter (threshold < 0.1) and a Pearson collinearity filter (r > 0.80).
- **Output:** `data/processed/fisicoquimicos_filtrados.csv` (34 descriptors) and `data/processed/maccs_filtrados.csv` (49 keys).

### Step 3 — Training, optimization, and validation of the ML-QSAR models

```bash
python src/03_train_qsar.py
```

- **Input:** processed matrices in `data/processed/`
- **Methodology:** reproducible train/test split (80/20, `random_state=42`). Hyperparameter optimization via 5-fold cross-validation (`GridSearchCV`) for two independent *Random Forest* estimators. Evaluation of both individual predictions and the consensus combination against the test set, computing R², RMSE, MAE, and Pearson's correlation coefficient (r).
- **Output:** serialized models in `models/` and true-vs-predicted scatter plots in `results/figures/`.

### Step 4 — Large-scale virtual screening of CMNPD and consensus rule

```bash
python src/04_virtual_screening.py
```

- **Input:** `data/raw/cmnpd-07-2026.sdf`
- **Methodology:** parses the molecular structures of the CMNPD marine chemical library, extracts the same descriptors optimized in Step 2, and computes affinities against saFabI using the trained estimators. Applies a strict consensus rule: only molecules with a simultaneous predicted pChEMBL ≥ 6.0 (IC₅₀ ≤ 1 μM) in both models are selected.
- **Output:** `results/candidatos_consenso_top.csv` and an inter-model correlation plot in `results/figures/`.

### Step 5 — Oral druggability and structural diversity assessment

```bash
python src/05_druggability_and_scaffolds.py
```

- **Input:** `results/candidatos_consenso_top.csv`
- **Methodology:** evaluates the physicochemical parameters of Lipinski's Rule of Five (Ro5) with a flexible criterion adapted to the structural complexity of marine natural products (≤ 1 violation allowed). Extracts molecular scaffolds via Bemis-Murcko decomposition and prioritizes the compound with the highest predicted activity within each unique structural family (best-in-class).
- **Output:** `results/CMNPD_Representantes_Elite.csv` and `results/CMNPD_Lipinski_Flexible.csv`.

### Step 6 — 3D conformational modeling and energy minimization

```bash
python src/06_prepare_ligands_3d.py
```

- **Input:** `results/CMNPD_Representantes_Elite.csv`
- **Methodology:** selects the top 5 elite representatives, performs protonation and explicit hydrogen addition, generates reproducible three-dimensional conformers via the ETKDG algorithm, and minimizes steric strain under the MMFF94 force field. Exports each compound as an individual SDF file, ready for molecular docking simulation.
- **Output:** three-dimensional structures `results/Top_*_Pred_*.sdf` and a technical profile of the lead candidate.

---

## 6. Lead compound: enisorine E

The methodological pipeline identified **enisorine E** (Top 3, identifier CMNPD29465) as the priority candidate:

| Property | Value |
|---|---|
| **Canonical SMILES** | `CCC(=O)N(C)CCc1cc(Br)c(OCCCNC(=O)Cc2ccc(O)cc2)c(I)c1` |
| **Predicted biological potency (consensus)** | pChEMBL = 6.52 (IC₅₀ ≈ 303 nM) |
| **Natural source** | Marine sponge *Iotrochota* cf. *iota* |

**Pharmacological and mechanistic relevance:** enisorine E represents a halogenated chemical scaffold with no previously reported biological assays against the bacterial fatty acid biosynthesis pathway (FAS-II). In molecular docking simulations against the saFabI catalytic site (PDB ID: [4BNF](https://www.rcsb.org/structure/4BNF)), its bromo- and iodo-substituents together with the 4-hydroxyphenyl group establish a complementary network of hydrogen bonds and hydrophobic contacts that circumvents the mutation sites typically associated with known resistance mechanisms.

---

## 7. License and citation

This project was developed exclusively for academic and scientific research purposes within the framework of Universidad Internacional de Valencia (VIU). The source code is distributed under the **MIT License**.

If you find this repository useful, you may cite it as:

```
Peláez Sánchez J. Computational discovery of novel antibacterial candidates against
Staphylococcus aureus: integrating ML-QSAR models and molecular docking on the marine
CMNPD chemical library [Master's Thesis]. Universidad Internacional de Valencia (VIU); 2026.
```
