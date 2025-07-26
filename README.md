# Credit Default Risk – Model Robustness Evaluation

This repository contains the complete implementation of our research and applied analytics project on **model robustness evaluation** in the context of **credit default risk prediction**. Developed as part of the McGill University Master of Management in Analytics program, in collaboration with **Synechron**, the project explores how machine learning models behave under real-world uncertainty and adversarial threats.

## Project Objective

To assess and improve the robustness of credit risk models by simulating data variability and adversarial conditions. Our goal was to identify weaknesses in model performance—particularly on the minority class (defaulters)—and evaluate how various defense techniques affect metrics such as AUC, recall, and F1-score under stress.

## Key Contributions

- End-to-end credit default classification pipeline using **TabNet** and **XGBoost**
- Integration of **robustness evaluation techniques**:
  - Gaussian noise injection  
  - Covariate perturbations  
  - Adversarial attacks: L₀, L₂, FGSM, and loss-based  
- Development of a **modular Python toolkit** for robustness testing
- Comparative analysis of model sensitivity, class recall, and AUC degradation
- Full documentation and one-page executive summary for stakeholders

## Repository Structure

```
.
├── Deep_Learning_Model__V5.ipynb         # Core notebook: training, testing, and robustness evaluations
├── model_robustness_eval.py              # Python module: reusable robustness evaluation functions
├── Executive_Summary.pdf                 # Visual one-pager summarizing key findings and deliverables
├── Model_Robustness_Presentation.pdf     # Final presentation (client-ready)
├── Model_Robustness_Documentation.pdf    # Full documentation of techniques and results
```

## Models Implemented

### 1. TabNet (Deep Learning)
- Initial accuracy: 92%, AUC: 0.77  
- Weak Class 1 recall (2%) improved to 66% using SMOTE + noise injection  
- Showed stronger resilience under L₀ loss-based attacks

### 2. XGBoost (Tree-based)
- Solid baseline metrics on majority class  
- Class 1 recall improved marginally (5% → 22%) with SMOTE  
- More sensitive to adversarial and covariate attacks compared to TabNet

## Techniques Implemented

| Technique               | Purpose                                |
|-------------------------|----------------------------------------|
| SMOTE                   | Balance class distribution             |
| Gaussian Noise          | Simulate real-world feature variability|
| Covariate Perturbation  | Test model response to feature shift   |
| L₀ / L₂ Attacks         | Sparse or bounded adversarial tests    |
| FGSM                    | Gradient-based attack simulation       |
| Loss-based Attack       | Targeted model destabilization         |

## How to Use

### 1. Clone the Repository

```bash
git clone https://github.com/yvesassa/model-robustness-credit-risk.git
cd model-robustness-credit-risk
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

If `requirements.txt` is missing, manually install:

```bash
pip install pandas numpy matplotlib scikit-learn xgboost torch pytorch-tabnet
```

### 3. Run Notebook

Use Jupyter, VS Code, or Google Colab to open and run:

```bash
Deep_Learning_Model__V5.ipynb
```

### 4. Reuse the Evaluation Module

Import and call functions from `model_robustness_eval.py` in your own scripts:

```python
from model_robustness_eval import evaluate_adversarial_attack
```

All functions are documented and ready for direct use.

## Results Summary

| Model     | AUC (Baseline) | AUC (L₀ Attack) | Recall (Baseline) | Recall (Improved) |
|-----------|----------------|-----------------|--------------------|-------------------|
| TabNet    | 0.77           | 0.66            | 2%                 | 66%               |
| XGBoost   | 0.75           | 0.63            | 5%                 | 22%               |

## Deliverables

- Robustness evaluation module for external use  
- Executive summary for decision-makers  
- Final presentation (shared with client)  
- All code and documentation in one structured repo  

## Team

**Yves Assali**  
**Lucas Doan**  
McGill University – Desautels Faculty of Management  
Master of Management in Analytics


## Acknowledgements

Special thanks to **Synechron** for their industry guidance and to the McGill MMA faculty for academic mentorship.  
We also acknowledge open-source tools such as TabNet, XGBoost, PyTorch, and scikit-learn that made this project possible.
