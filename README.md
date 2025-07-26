# Credit Default Risk – Model Robustness Evaluation

**Collaborators:** Yves Assali & Lucas Doan  
**Program:** McGill University, Master of Management in Analytics  
**Industry Partner:** Synechron  
**Date:** July 2025  

## Overview

This project investigates the stability of machine learning models for credit default prediction under real-world uncertainty and adversarial threats. We compare a TabNet deep-learning model and an XGBoost tree-based model, identify their weaknesses on the minority (defaulter) class, and apply a suite of robustness techniques to measure and improve performance.

## Objectives

- **Assess baseline performance** of TabNet and XGBoost on an imbalanced credit dataset  
- **Quantify sensitivity** to input variability, feature shifts, and adversarial attacks  
- **Develop a reusable toolkit** to automate robustness evaluation in any ML pipeline  

## Key Contributions

- **End-to-end pipeline** for credit default classification  
- **Robustness techniques** implemented:  
  - SMOTE oversampling  
  - Gaussian noise injection  
  - Covariate perturbations  
  - Adversarial attacks (L₀, L₂, FGSM, loss-based)  
- **Modular Python toolkit** (`model_robustness_eval.py`) for rapid integration  
- **Comparative analysis** of AUC, recall, and F1 degradation under stress  

## Methodology

1. **Data Preparation:**  
   • Home Credit dataset (≈300 K records, 100+ features)  
   • ID removal, missing-value imputation, one-hot encoding, outlier filtering  
2. **Baseline Training:**  
   • TabNet: 92% accuracy, AUC 0.77, recall 2% on defaulters  
   • XGBoost: strong majority-class metrics, recall 5%  
3. **Robustness Evaluation:**  
   • Apply noise, perturbations, and adversarial scenarios  
   • Record performance drops and recovery after defense techniques  

## Results Highlights

| Model    | AUC (Base) | AUC (Under L₀) | Recall (Base) | Recall (Improved) |
|----------|------------|----------------|---------------|-------------------|
| TabNet   | 0.77       | 0.66           | 2%            | 66%               |
| XGBoost  | 0.75       | 0.63           | 5%            | 22%               |

- **TabNet + SMOTE + noise** yielded the greatest recall improvement.  
- **L₀ loss-based attacks** were the most challenging for both models.  

## Deliverables

- **Jupyter Notebook** (`Deep_Learning_Model__V5.ipynb`)  
- **Python toolkit** (`model_robustness_eval.py`)  
- **Executive summary**, presentation slides, and full documentation  

## Repository Structure

├── Deep_Learning_Model__V5.ipynb
├── model_robustness_eval.py
├── Executive_Summary.pdf
├── Model_Robustness_Presentation.pdf
├── Model_Robustness_Documentation.pdf


## Team & Acknowledgements

**Yves Assali** & **Lucas Doan**  
McGill Desautels Faculty of Management  
Special thanks to Synechron and McGill MMA faculty for guidance.


