# What Models Learn from Missing Data: Imputation as Feature Engineering  
**Author:** Rebecca Whitworth, PhD  
**License:** MIT  
**Version:** November 2025 (v1.0)  

---

## Overview  
This repository contains code and figures accompanying the working paper  
**“What Models Learn from Missing Data: Imputation as Feature Engineering.”**

Standard model validation focuses on learner architecture but ignores imputation strategy. This paper shows that two models with identical features and test-set performance can exhibit completely different decision surfaces simply because they filled missing data differently.

Using the 2022 HMDA dataset, the paper examines how different missing-data mechanisms—Missing Completely at Random (MCAR), Missing at Random (MAR), and Missing Not at Random (MNAR)—interact with common imputation strategies (mean, linear, tree-based). Using the 2022 HMDA dataset, the experiments reconstruct missing information under controlled conditions and measure how model behavior (AUC, R², curvature) deforms as a function of the missingness pattern.

Figure 2 from the paper shows the degredation in AUC from three different learners across the four different missingness regimes.

![Figure 2: AUC Degradation with Linear Imputation](figures/Figure%202%20AUC%20Degredation%20with%20Linear%20Impution.png)

---
## Key Finding

Imputation is not preprocessing—it's feature engineering. When a variable carries unique signal (like debt-to-income ratio in credit models), every imputation method imposes a different geometry on the learner's decision surface:

- **Mean imputation** collapses variation into artificial clusters
- **Linear imputation** flattens curvature and erases local structure  
- **Tree imputation** creates synthetic discontinuities by learning the replacement rule

For model risk: two models trained on "identical" data with different imputations are **not equivalent**. Changes to imputation should trigger re-validation.

---
## For Practitioners

If you're validating credit models, this research suggests:

1. **Document your imputation strategy** in model risk filings—it's not a preprocessing detail
2. **Test sensitivity** to alternative imputation methods during validation  
3. **Monitor missingness patterns** over time—shifts in who has missing data change model behavior even if parameters stay fixed

See the paper's "Practical Implications" section (p. 14) for details.

---
## Repository structure
```
missingness-imputation/
├── Missingness_Imputation.py       # Main experiment script (all Monte Carlo runs)
├── results/                        # Generated CSV outputs (tables, metrics)
├── figures/                        # Auto-generated AUC and mechanism panels
├── environment.yml                 # Conda environment for reproducibility
├── requirements.txt                # pip dependencies
├── README.md
└── LICENSE
```

---

## Environment
Experiments were run on macOS (Apple Silicon) under Python 3.11.  

**Core dependencies**
- numpy 1.26  
- pandas 2.2  
- scikit-learn 1.5  
- matplotlib 3.9  
- seaborn 0.13  

**Recreate with Conda**
```bash
conda env create -f environment.yml
conda activate Missingness_Imputation
```
**or with pip**
```
python -m venv venv
source venv/bin/activate  # or venv\Scripts\activate on Windows
pip install -r requirements.txt
```

---
## Usage
To reproduce all experiments and figures:
```
python hmda_missingness.py
```
Results are saved automatically to:
```
results/monte_outputs.csv
results/table1_fullsample.csv
figures/panel_*.png
```
Each figure corresponds to an imputation method and missingness mechanism as shown in the paper (Income MCAR | DTI MCAR | DTI MAR | DTI MNAR).

---
## Notes for Reproducibility
- Random seed fixed at `SEED = 4242`.  
- Each experiment repeats across 25 Monte Carlo draws of 100 K observations sampled from 5.77 M HMDA records.  
- Figures and metrics are deterministic given identical code and data.  

--
## Data Access

This project uses the 2022 HMDA Public Loan Application Register (LAR),
available from the CFPB public data portal:

https://ffiec.cfpb.gov/data-publication/2022

Download the pipe-delimited file `2022_public_lar_pipe.txt` and place it in:
```
data/2022_public_lar_pipe.txt
```
---
## Citation

**Paper:** [What Models Learn from Missing Data: Imputation as Feature Engineering](https://ssrn.com/abstract=TBD)  
**Code:** [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.XXXXX.svg)](https://doi.org/10.5281/zenodo.XXXXX)

If you use this code or reference its results, please cite:
```
@software{whitworth2025missingness,
  author    = {Rebecca Whitworth},
  title     = {What Models Learn from Missing Data: Imputation as Feature Engineering},
  month     = nov,
  year      = 2025,
  publisher = {SSRN},
  version   = {v1.0},
  url       = {https://ssrn.com/abstract=TBD}
}
```
---
## Contact

Rebecca Whitworth
Economist | Model Risk | Credit Markets 
rebeccawhitworth [at] gmail.com  
[github.com/r-whitworth](https://github.com/r-whitworth)  
[linkedin.com/in/rwhitworth](https://linkedin.com/in/rwhitworth)
