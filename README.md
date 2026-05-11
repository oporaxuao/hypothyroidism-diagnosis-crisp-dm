# 🦋 Hypothyroidism Diagnosis with Machine Learning

Binary classification model trained on 3,772 patient records to predict hypothyroidism risk from hormonal measurements and clinical history — built with a relentless focus on Recall to ensure no sick patient goes undiagnosed.

> *"Every year, thousands of patients live with chronic fatigue, unexplained weight gain, and depression — not knowing the cause may be a butterfly-shaped gland in their neck."*

---

## 🎯 Business Objective

Hypothyroidism is one of the most underdiagnosed conditions worldwide. The thyroid, when it fails, silently disrupts metabolism, mood, and cardiac function — and diagnosis often comes too late. This project was built around one core question: **can a machine learn to recognize the signs that clinical screening sometimes misses?**

In a conventional clinical setting, a physician analyzes exams, history, and symptoms sequentially. With hundreds of patients, this process is slow and prone to human error. The most critical risk is not diagnosing a healthy person as sick — that generates an extra exam, nothing more. The real risk is the opposite: **letting a sick patient leave the consultation without a diagnosis**.

In machine learning, this is a False Negative. Minimizing this number drove every technical decision in this project.

---

## 📂 Dataset

| Attribute | Detail |
|---|---|
| Records | 3,772 patients |
| Features | 29 variables (hormonal, demographic, clinical) |
| Target variable | `binaryClass`: P (positive) or N (negative) |
| Class imbalance | 92% negative / 8% positive |

The imbalance itself defined the entire strategy: a model that always predicted "healthy" would achieve 92% accuracy — and be completely useless. This drove every subsequent technical decision: metric choice, imputation strategy, and model parameters.

---

## 🗂️ Methodology — CRISP-DM

### 1. Business Understanding
Before opening any file, it was essential to understand the clinical problem. **TSH (Thyroid-Stimulating Hormone)** is the primary marker of hypothyroidism: when the thyroid fails, the pituitary gland elevates TSH trying to stimulate it. This medical knowledge was validated by the model itself — TSH emerged as the most important feature in SHAP analysis, confirming the machine learned biology, not noise.

### 2. Exploratory Data Analysis (EDA)
The data arrived dirty. The `?` character masked missing values across entire columns — TBG, for example, had over 95% missing values. Key findings:
- Hormonal distributions clearly distinct between positive and negative patients, especially in TSH
- Extreme hormonal outliers that — contrary to classical statistics — were exactly the most valuable signal. A TSH of 500 is not noise; it is the diagnosis.
- Demographic profile consistent with literature: disease more prevalent in women and patients over 50

### 3. Data Preparation
Three technical decisions stand out:

**Hormonal outliers were kept.** Removing a TSH of 500 would erase the clearest evidence of severe hypothyroidism. In clinical data, the outlier is often the diagnosis.

**Post-split imputation.** The median used to fill missing values was calculated exclusively on the training set and applied to the test set — never the reverse. Doing the opposite causes data leakage: the model would "see" test information before it should, artificially inflating metrics.

**Columns with over 70% missing values were removed.** Imputing the majority of a column's values is not filling gaps — it is inventing data.

### 4. Modeling

Four models were compared in 5-fold stratified cross-validation:

| Model | Recall | F1-Score | AUC-ROC |
|---|---|---|---|
| Logistic Regression | baseline | baseline | baseline |
| Random Forest | — | — | — |
| XGBoost | — | — | — |
| **LightGBM (tuned)** | **best** | **best** | **best** |

`StratifiedKFold` ensured each fold maintained the original 92/8 class ratio. `RandomizedSearchCV` with 40 iterations found the best LightGBM hyperparameters in a fraction of the time a full `GridSearchCV` would require.

### 5. Evaluation
The champion model was evaluated on the test set — data that never influenced any training or selection decision.

SHAP analysis closed the loop: beyond good numbers, the model learned biologically correct relationships. TSH at the top of global importance. T3 and TT4 right below. The model is not a black box — it is an auditable system.

---

## 📊 Results

| Metric | Value | Clinical meaning |
|---|---|---|
| Recall | ≥ 95% | Of every 100 sick patients, the model detects ~95 |
| F1-Score | ≥ 92% | Strong balance between precision and sensitivity |
| AUC-ROC | ≥ 98% | Class separation capability near the theoretical ideal |
| Precision | ≥ 90% | Of every 100 alerts, ~90 are real cases |

---

## 🔍 Top 5 Most Decisive Features (SHAP)

1. **TSH** — Elevated level is the primary indicator. The pituitary screams when the thyroid goes silent.
2. **T3 / TT4** — Thyroid hormones produced directly by the gland.
3. **FTI** — Free Thyroxine Index, a derived measure of high clinical relevance.
4. **Age** — Risk increases significantly after age 50.
5. **On Thyroxine** — Patients already on hormone replacement therapy present distinct physiological patterns.

---

## ⚠️ Limitations & Next Steps

This model is not production-ready — it is a robust proof of concept. Before any real clinical implementation:

- **External validation:** test on data from other hospitals to assess generalization
- **Probability calibration:** ensure "70% risk" actually means 70%
- **Data drift monitoring:** hormonal distributions change with populations and equipment
- **Regulatory approval:** any clinical decision support system requires validation by competent authorities

---

## 🛠️ Tech Stack

| Category | Tools |
|---|---|
| Language | Python 3.x |
| Data Manipulation | Pandas, NumPy |
| Visualization | Matplotlib, Seaborn |
| Machine Learning | Scikit-learn, XGBoost, LightGBM |
| Explainability | SHAP |
| Environment | Jupyter Notebook |

---

## ▶️ How to Run

```bash
# Clone the repository
git clone https://github.com/oporaxuao/hypothyroidism-diagnosis-catboost.git
cd hypothyroidism-diagnosis-catboost

# Install dependencies
pip install pandas numpy matplotlib seaborn scikit-learn xgboost lightgbm shap jupyter

# Launch the notebook
jupyter notebook prevencao_hypothyroid.ipynb
```

Run cells in sequence. The notebook is self-contained — each step generates the variables needed for the next.

---

## 👤 Author

**João Alfredo de Sousa Siqueira**
[![LinkedIn](https://img.shields.io/badge/LinkedIn-oporaxuao-blue)](https://linkedin.com/in/oporaxuao)
[![GitHub](https://img.shields.io/badge/GitHub-oporaxuao-black)](https://github.com/oporaxuao)
