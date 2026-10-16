# 🏦 FinTech Credit Risk & Exploratory Data Analysis (EDA)

**Diagnosing Loan Defaults through Machine Learning, Statistical Analysis & Synthetic Data Balancing (SMOTE)**

## 📌 Executive Summary

This project tackles a critical business challenge for a FinTech lending firm: diagnosing the root cause of loan defaults and evaluating automated underwriting models. By conducting an end-to-end Exploratory Data Analysis (EDA) on **37,408 historical loan applications**, this study uncovered an **11.1% portfolio default rate** (`4,154` bad loans) driven by a severe **34.2% deficit in borrower cash liquidity**.

To evaluate whether current application attributes could reliably predict future defaults, three distinct machine learning strategies were tested across an untouched test benchmark of **7,431 loans**: a balanced **Decision Tree**, an unbalanced **Random Forest**, and a **SMOTE-balanced Random Forest**. Ultimately, the modeling pipeline exposed a hard **48% Recall ceiling**, providing mathematical proof of data collection gaps and driving an actionable 3-step executive roadmap for the VP of Risk.

---

## 📊 Key Portfolio & Financial Findings (`N = 37,408`)

| Metric / Discovery | Healthy Borrowers (`0`) | Defaulters (`1`) | Executive Business Impact |
| :--- | :--- | :--- | :--- |
| **Portfolio Volume** | `33,254` (`88.9%`) | `4,154` (`11.1%`) | Significant class imbalance that biases standard unweighted ML models toward the majority class. |
| **Average Loan Size (`INVESTMENT_TOTAL`)** | `$6,183,669.59` | `$6,367,281.59` | Defaulters and healthy borrowers request nearly identical loan sizes. |
| **Average Scheduled Payment (`DUE_PAYMENT`)** | `$376,573.18` | `$369,796.20` | Scheduled installment obligations do not linearly separate good and bad loans. |
| **Average Cash Reserves (`ACCCURRENTBALANCE`)** | **`$1,220,866.51`** | **`$802,761.17`** | **The Smoking Gun:** Defaulters carry **34.2% less cash liquidity** in their current accounts. |
| **Extreme "Whale" Outliers (`> $100M`)** | `257` Anomalies Removed (Peaking at `$633M`) | Filtered via a `$100M` ceiling to leave **37,151 clean records** for stable algorithmic training. |

---

## 🛠️ Project Architecture & Methodology

### Phase 1: Data Wrangling & Feature Engineering
* **Legacy Ingestion:** Extracted `37,408` records across `11` columns from legacy `.xls` files using `pandas`, `xlrd`, and `openpyxl`.
* **Statistical Imputation:** Rescued `945` incomplete records without dropping valuable historical default signals—imputing `838` missing `INSTALL_SIZE` records with the statistical median, `103` missing `CLIENT_TYPE` records with the mode (`Semi-urban`), and `6` missing demographic records (`INF_GENDER`, `INF_MARITAL_STATUS`, `COMPENSATION_CHARGED`) into an `'Unknown'` bucket.
* **Binary Target Engineering:** Mapped legacy text-based loan quality grades (`G` = `0` | `B`, `DF`, `SS` = `1`) into a strict binary target variable (`IS_DEFAULT`).

### Phase 2: Exploratory Data Analysis (EDA) & Outlier Sanitization
* **Univariate & Bivariate Grouping:** Quantified the `88.9%` vs. `11.1%` class split and uncovered the `$1.22M` vs. `$802.7K` cash liquidity gap using `pandas` aggregations.
* **Anomaly Filtering:** Plotted `INVESTMENT_TOTAL` against `DUE_PAYMENT` in `Seaborn` to isolate extreme corporate anomalies reaching `$633M`, applying a `$100M` ceiling filter (`257` rows removed) to retain **37,151 clean retail/commercial loans**.

### Phase 3: Baseline Machine Learning & The Precision/Recall Trap
* **Pipeline Setup:** Applied One-Hot Encoding (`pd.get_dummies`) to categorical features and split the clean dataset into an `80%` Training Set (`29,720` loans) and a `20%` Untouched Reality Test Set (`7,431` loans containing `6,684` good loans and `747` true defaults).
* **The "Paranoid" Decision Tree:** Trained a `DecisionTreeClassifier(class_weight='balanced')` to penalize missed defaults. While it caught **48% of defaults (`360 / 747`)**, it falsely accused **444 creditworthy customers** (`45%` Precision).

### Phase 4: Ensemble Modeling (Random Forest) & SMOTE Balancing
* **Attempt 1 — Unbalanced Random Forest (`n_estimators=100`):** Achieved **91% overall accuracy** and **61% Precision** (falsely flagging only `164` good clients), but favored the majority class—catching only **35% of defaults (`258 / 747`)** and missing `489` bad loans.
* **Attempt 2 — Synthetic Balancing via `SMOTE`:** Applied SMOTE exclusively to the training data, analyzing the `3,367` training defaults to synthetically generate **22,986 new default profiles** (`26,353` Good vs. `26,353` Synthetic Defaults). Retraining the Random Forest restored Recall to **48% (`361 / 747` caught)**, but dropped Precision to **38%** (`601` false positives).

---

## 🤖 Head-to-Head Model Performance Matrix (`Test Set = 7,431 Loans`)

| Model Architecture | Overall Accuracy | Default Precision (`Class 1`) | Defaults Caught (`Recall`) | Good Clients Falsely Flagged (`FP`) | Defaults Missed (`FN`) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **1. Baseline Decision Tree** (`class_weight='balanced'`) | `89.0%` | `45.0%` | **48.0%** (`360 / 747`) | `444` | `387` |
| **2. Unbalanced Random Forest** *(Before SMOTE)* | **`91.0%`** | **`61.0%`** | `35.0%` (`258 / 747`) | **`164`** | `489` |
| **3. SMOTE Random Forest** *(50/50 Synthetic Training)* | `87.0%` | `38.0%` | **48.0%** (`361 / 747`) | `601` | **`386`** |

---

## 🎯 Strategic Recommendations for the VP of Risk

Why did Recall hit a hard ceiling at **48%** across both balanced models? Because loan size, current account balance, and basic demographics alone cannot fully capture human repayment behavior. Instead of deploying a brittle auto-rejection engine, the business should execute a 3-step roadmap:

1. **Deploy the Unbalanced Random Forest as a Tier-1 Triage Flag:** Leverage Model 2's **61% Precision** to route high-risk applications to manual underwriting without alienating hundreds of good customers (disturbing only `2.4%` of healthy applicants).
2. **Enforce Minimum Liquidity Buffers:** Because defaulters hold **34.2% less cash** in their current accounts (`$802.7K` vs. `$1.22M`), institute mandatory minimum cash-to-loan ratios prior to disbursement.
3. **Overhaul Application Data Ingestion:** Upgrade the origination pipeline to capture **Verified Debt-to-Income (DTI) ratios**, **external credit bureau scores**, and **employment tenure** to break past the 48% detection ceiling.

---

## 📂 Repository Structure

```text
├── FinTech_Credit_Risk_EDA.ipynb   # Complete Jupyter Notebook (Code, Visuals & Outputs)
├── PDF.pdf                         # 3-Page Executive Briefing & Management Presentation
├── README.md                       # Project Documentation & Executive Summary
└── train-FIN_ANA_DATA.xls          # Raw Historical Training Dataset (37,408 Records)
```

## 🚀 How to Run Locally

```bash
# 1. Clone the repository
git clone https://github.com/dehnis07/fintech-credit-risk-eda.git
cd fintech-credit-risk-eda

# 2. Install required Python libraries
pip install pandas numpy matplotlib seaborn scikit-learn imbalanced-learn xlrd openpyxl

# 3. Open FinTech_Credit_Risk_EDA.ipynb in VS Code or Jupyter Lab and click "Run All"
```