# Personal Loan Acceptance Prediction

## Customer Analytics and Machine Learning for Targeted Banking Campaigns

This project analyzes customer demographic, financial, and banking-behavior data to predict whether a customer is likely to **accept a personal loan offer**.

The project is presented from a **business analysis and decision-support perspective**. The objective is not to determine whether a customer should be approved for credit. Instead, the analysis focuses on identifying customers who are more likely to respond positively to a personal-loan marketing campaign so that a bank could prioritize outreach, reduce broad untargeted marketing, and design more relevant customer offers.

The workflow combines data preparation, exploratory analysis, feature selection, machine learning, hyperparameter tuning, cross-validation, and statistical model comparison.

---

## Executive Summary

The raw dataset contains **5,000 customer records and 14 variables** covering customer demographics, income, banking relationships, credit-card behavior, mortgage information, and previous personal-loan acceptance.

During data preparation:

- column names were standardized for readability;
- `CreditCard_Spending` was converted from string values such as `1/60` to numeric values such as `1.60`;
- **52 records with negative professional experience** were removed;
- missing values and duplicate records were checked;
- the processed dataset contains **4,948 customer records**;
- `Customer_ID` was excluded from predictive modeling because it is an identifier rather than a behavioral predictor.

The processed data contains **480 loan acceptances**, representing approximately **9.7%** of customers. This class imbalance is important because the business problem is fundamentally about identifying a relatively small group of likely campaign responders.

Feature selection using **SelectKBest with the ANOVA F-test (`f_classif`)** identified `Annual_Income`, `CreditCard_Spending`, and `CD_Account` as the strongest individual predictors in the current analysis.

Five traditional machine-learning classifiers were compared using stratified 5-fold cross-validation and ROC-AUC. **Random Forest achieved the strongest cross-validated performance with a mean ROC-AUC of 0.9977** and a test ROC-AUC of **0.9984**.

A TensorFlow/Keras neural network was also explored through sequential hyperparameter tuning. Its validation AUC was consistently high, although exact neural-network tuning results can vary between runs because training is stochastic and the notebook tunes one hyperparameter at a time rather than evaluating every combination jointly.

---

## Business Problem

A bank may have thousands or millions of customers who could potentially receive a personal-loan offer. Sending the same campaign to every customer can create several business problems:

- unnecessary marketing cost;
- low campaign response rates;
- irrelevant offers for customers with little interest;
- inefficient use of sales and relationship-management resources;
- customer fatigue from excessive or poorly targeted communication.

A predictive customer-propensity model can help shift the process from **mass marketing** toward **data-driven campaign prioritization**.

The practical question addressed by this project is:

> **Which existing customers are more likely to accept a personal loan offer, and which customer characteristics are most useful for identifying them?**

---

## Business Analysis Questions

The project is designed to support questions such as:

1. What proportion of customers accepted the previous personal-loan offer?
2. How do customer income, spending, education, mortgage, and banking relationships differ between accepters and non-accepters?
3. Which variables have the strongest statistical relationship with personal-loan acceptance?
4. Can customer information be used to rank or classify likely campaign responders accurately?
5. Which predictive model provides the strongest discrimination between likely accepters and non-accepters?
6. What trade-off exists between predictive performance and model interpretability?
7. How could model scores be incorporated into a targeted marketing workflow?
8. What limitations must be addressed before using this type of model in a real banking environment?

---

## Dataset

**Dataset:** Bank Personal Loan Modelling  
**Source:** Kaggle  
**Source reference:** https://www.kaggle.com/datasets/ahmadrafiee/bank-personal-loan

### Raw Dataset Summary

| Item | Value |
|---|---:|
| Customer records | 5,000 |
| Original variables | 14 |
| Personal-loan acceptances | 480 |
| Personal-loan non-acceptances | 4,520 |
| Records with negative experience | 52 |
| Missing values identified | 0 |
| Duplicate rows identified | 0 |

### Processed Dataset Summary

| Item | Value |
|---|---:|
| Customer records | 4,948 |
| Variables before removing identifier | 14 |
| Personal-loan acceptances | 480 |
| Personal-loan non-acceptances | 4,468 |
| Acceptance rate | 9.7% |
| Missing values | 0 |
| Duplicate rows | 0 |

### Main Variables

| Variable | Business Meaning |
|---|---|
| `Age` | Customer age |
| `Experience` | Years of professional experience |
| `Annual_Income` | Customer annual income measure |
| `ZIP_Code` | Customer residential ZIP code |
| `Family_Size` | Number of people in the customer's family |
| `CreditCard_Spending` | Average credit-card spending measure |
| `Education_Level` | Education category |
| `Mortgage` | Mortgage value |
| `Personal_Loan` | Target: whether the customer accepted the personal-loan offer |
| `Securities_Account` | Whether the customer has a securities account |
| `CD_Account` | Whether the customer has a certificate-of-deposit account |
| `Online_Banking` | Whether the customer uses online banking |
| `Uses_CreditCard` | Whether the customer uses the bank's credit card |

`Customer_ID` is retained in the processed Phase 1 dataset for traceability but is removed before predictive modeling.

---

## Project Structure

```text
Personal-Loan-Acceptance-Prediction-Machine-Learning/
│
├── README.md
├── requirements.txt
├── .gitignore
│
├── data/
│   ├── raw/
│   │   └── Bank_Personal_Loan_Modelling.csv
│   │
│   └── processed/
│       └── Phase1_cleaned.csv
│
├── notebooks/
│   ├── Phase1.ipynb
│   └── Phase2.ipynb
│
└── private/
    ├── Phase1.html
    └── Phase2.html
```

The `private/` directory is excluded from Git and is intended for local HTML exports or other files that should not be published in the repository.

---

## Analytical Workflow

```text
Raw Customer Dataset
        |
        v
Data Validation and Cleaning
        |
        v
Column Standardization
        |
        v
Exploratory Data Analysis
        |
        v
Processed Customer Dataset
        |
        v
Remove Customer Identifier
        |
        v
ANOVA SelectKBest Feature Selection
        |
        v
Train/Test Split + Model-Specific Scaling
        |
        v
Hyperparameter Tuning
        |
        v
Model Evaluation
        |
        v
5-Fold Cross-Validation
        |
        v
Statistical Model Comparison
        |
        v
Business Interpretation
```

---

## Phase 1 — Data Preparation and Exploratory Analysis

`notebooks/Phase1.ipynb` prepares the raw banking dataset for predictive analysis.

### Data Preparation

The notebook performs the following steps:

- imports the raw CSV dataset;
- reviews dataset structure and feature types;
- standardizes column names;
- converts `CreditCard_Spending` from slash-formatted strings to numeric values;
- checks unique values and summary statistics;
- removes observations where `Experience < 0`;
- checks missing values;
- checks duplicate observations;
- performs exploratory visualizations;
- exports the cleaned dataset for Phase 2.

The resulting processed dataset contains **4,948 records**.

### Exploratory Analysis

Phase 1 investigates:

- annual-income distribution;
- education-level distribution;
- credit-card spending;
- personal-loan acceptance distribution;
- income versus loan acceptance;
- education versus loan acceptance;
- credit-card spending versus loan acceptance;
- mortgage versus loan acceptance;
- income and spending interactions;
- income, education, and loan-status relationships.

---

## Business-Oriented Exploratory Findings

The following observations are descriptive relationships in the processed dataset and should not be interpreted as causal effects.

### Loan acceptance is a minority outcome

Only around **9.7%** of the processed customer records represent customers who accepted the personal-loan offer.

From a campaign perspective, this means a useful model must identify a relatively small responder segment rather than simply predict the majority class.

### Income is strongly associated with acceptance

Average `Annual_Income` in the processed dataset is approximately:

- **144.75** for customers who accepted the personal loan;
- **66.19** for customers who did not.

This large separation is consistent with `Annual_Income` being the highest-ranked feature in the ANOVA feature-selection stage.

### Credit-card spending is higher among accepters

Average `CreditCard_Spending` is approximately:

- **3.91** for accepters;
- **1.72** for non-accepters.

This indicates that customer spending behavior may provide useful information for campaign segmentation.

### CD-account relationship is a strong signal

Within the processed dataset:

- customers with a `CD_Account` have an acceptance rate of approximately **46.4%**;
- customers without a `CD_Account` have an acceptance rate of approximately **7.3%**.

This does not establish causation, but it suggests that the customer's existing relationship with the bank may be valuable when prioritizing personal-loan campaigns.

### Education and mortgage characteristics also differ

Customers at education levels 2 and 3 show higher observed acceptance rates than education level 1, and customers who accepted the offer also have a higher average mortgage value in this dataset.

These findings suggest that campaign propensity is associated with a combination of **financial capacity, spending behavior, household characteristics, and existing banking relationships**, rather than one variable alone.

---

## Phase 2 — Predictive Modeling

`notebooks/Phase2.ipynb` uses the cleaned Phase 1 dataset to build and compare predictive models.

### Feature Selection

The notebook applies:

- `SelectKBest`
- ANOVA F-test using `f_classif`
- `k = 10`

The current feature-ranking output is:

| Rank | Feature | F-Score |
|---:|---|---:|
| 1 | `Annual_Income` | 1686.21 |
| 2 | `CreditCard_Spending` | 781.50 |
| 3 | `CD_Account` | 547.79 |
| 4 | `Mortgage` | 102.25 |
| 5 | `Education_Level` | 96.50 |
| 6 | `Family_Size` | 19.76 |
| 7 | `Securities_Account` | 2.44 |
| 8 | `Age` | 1.00 |
| 9 | `Experience` | 0.99 |
| 10 | `Online_Banking` | 0.19 |

The selected features are then used consistently across the predictive models.

---

## Models Evaluated

The project evaluates:

- Logistic Regression
- Decision Tree
- Random Forest
- Support Vector Machine
- k-Nearest Neighbors
- Feedforward Neural Network using TensorFlow/Keras

### Preprocessing by Model

`StandardScaler` is applied to models that are sensitive to feature magnitude:

- Logistic Regression
- Support Vector Machine
- k-Nearest Neighbors
- Neural Network

Tree-based models are trained without feature scaling:

- Decision Tree
- Random Forest

---

## Hyperparameter Tuning

### Random Forest

Grid search evaluates combinations of:

- `n_estimators`
- `max_depth`
- `min_samples_split`
- `min_samples_leaf`
- `max_features`

Best parameters in the current run:

```text
n_estimators = 150
max_depth = 10
min_samples_split = 2
min_samples_leaf = 1
max_features = sqrt
```

### Logistic Regression

The project tunes:

- `penalty`
- regularization strength `C`

Current best parameters:

```text
penalty = l2
C = 0.1
solver = liblinear
```

### Decision Tree

The project tunes:

- `criterion`
- `max_depth`
- `min_samples_split`
- `min_samples_leaf`

Current best parameters:

```text
criterion = entropy
max_depth = 3
min_samples_split = 2
min_samples_leaf = 1
```

### Support Vector Machine

The project tunes:

- kernel type;
- `C`;
- `gamma`.

Current best parameters:

```text
kernel = rbf
C = 10
gamma = scale
```

### k-Nearest Neighbors

The project tunes:

- number of neighbors;
- weighting method;
- distance metric.

Current best parameters:

```text
n_neighbors = 20
weights = distance
metric = manhattan
```

### Neural Network

The TensorFlow/Keras feedforward network explores:

- number of hidden layers;
- neurons per layer;
- learning rate;
- dropout rate;
- batch size.

Each setting is evaluated using validation AUC after 30 epochs.

The neural-network tuning is **sequential and stochastic**, so exact validation AUC values can change between runs. It should therefore be treated as exploratory model tuning rather than a deterministic exhaustive grid search.

---

## Model Performance

### Test-Set ROC-AUC

| Model | Test ROC-AUC |
|---|---:|
| **Random Forest** | **0.9984** |
| Decision Tree | 0.9963 |
| Support Vector Machine | 0.9926 |
| k-Nearest Neighbors | 0.9789 |
| Logistic Regression | 0.9583 |

### Stratified 5-Fold Cross-Validation

| Model | Mean ROC-AUC |
|---|---:|
| **Random Forest** | **0.9977** |
| Decision Tree | 0.9931 |
| Support Vector Machine | 0.9850 |
| k-Nearest Neighbors | 0.9713 |
| Logistic Regression | 0.9559 |

Among the five models included in the cross-validation comparison, **Random Forest provides the strongest overall discrimination**.

The notebook also performs paired t-tests using fold-level AUC scores. Random Forest shows statistically significant differences versus Logistic Regression, Decision Tree, SVM, and kNN at the 5% significance level in the current analysis.

The neural network is evaluated separately through validation AUC and is not included in that paired cross-validation t-test table.

---

## Business Interpretation

The project indicates that a customer's likelihood of accepting a personal-loan campaign is not random. In this dataset, acceptance is strongly associated with a combination of:

- income;
- credit-card spending;
- certificate-of-deposit ownership;
- mortgage characteristics;
- education;
- family size;
- broader banking relationships.

From a campaign-management perspective, the model could be used as a **propensity-ranking tool**.

Rather than contacting all customers equally, a bank could:

1. score eligible customers using the trained model;
2. rank customers by predicted probability of acceptance;
3. prioritize the highest-propensity segment;
4. tailor campaign messaging using customer characteristics;
5. test different score thresholds against campaign cost and conversion value;
6. compare model-targeted campaigns with existing targeting rules through controlled experiments.

The model should support marketing prioritization rather than replace credit-policy, affordability, compliance, or responsible-lending assessments.

---

## Business Value

A well-validated propensity model could potentially support:

### More Efficient Campaign Targeting

Marketing resources can be concentrated on customers with stronger predicted response probabilities.

### Reduced Contact Waste

Lower-propensity customers can receive fewer irrelevant loan offers, reducing unnecessary campaign volume.

### Customer Segmentation

The feature analysis provides a basis for understanding which financial and relationship characteristics distinguish higher-response customer groups.

### Sales Prioritization

Relationship managers or outbound teams could use propensity scores to prioritize follow-up activity.

### Data-Driven Campaign Testing

Predicted probabilities can be converted into score bands and tested against actual campaign conversion rates.

### Model-Based Decision Support

The project demonstrates how descriptive customer analysis and predictive modeling can be combined instead of relying only on intuition or broad demographic rules.

---

## Recommended Use in a Real Business Workflow

A production implementation should not immediately send offers based only on the model output.

A stronger process would be:

```text
Eligible Customer Population
          |
          v
Policy / Compliance Filters
          |
          v
Propensity Model
          |
          v
Predicted Acceptance Probability
          |
          v
Customer Segmentation / Score Bands
          |
          v
Campaign Treatment
          |
          v
Observed Conversion
          |
          v
Performance Monitoring and Retraining
```

The decision threshold should be selected using **business costs and benefits**, not only ROC-AUC.

For example, the bank could compare:

- campaign cost per contacted customer;
- expected value of a successful loan conversion;
- conversion rate by model-score band;
- false-positive contact cost;
- customer contact limits;
- incremental uplift relative to existing targeting.

---

## Limitations

The findings should be interpreted as a portfolio machine-learning analysis rather than a deployment-ready banking decision system.

Key limitations include:

- The processed dataset contains only **4,948 records**.
- The positive class represents only about **9.7%** of observations.
- No explicit class-balancing technique such as SMOTE or class weighting is implemented.
- The evaluation uses random train/test splitting and stratified cross-validation rather than a true out-of-time validation sample.
- The dataset may not represent the demographic, behavioral, regulatory, or economic characteristics of a current real-world banking portfolio.
- Feature selection is univariate and does not by itself capture all interactions between variables.
- The neural-network tuning procedure changes one hyperparameter at a time rather than evaluating all combinations.
- Neural-network results can vary across runs unless deterministic random seeds and operations are fully controlled.
- The model comparison focuses heavily on ROC-AUC and does not optimize a real campaign profit or cost function.
- Probability calibration is not evaluated.
- No dedicated explainability framework such as SHAP is implemented.
- Fairness and bias across customer groups are not formally evaluated.
- Observed relationships are associative and should not be interpreted as causal.
- A campaign-propensity model should not be used as a substitute for lending eligibility, affordability, risk, or compliance decisions.

---

## Future Improvements

Potential extensions include:

- adding class weighting or resampling experiments;
- evaluating precision-recall AUC because the target is imbalanced;
- selecting campaign thresholds using expected business value;
- adding probability calibration;
- adding SHAP-based model explainability;
- comparing gradient boosting, XGBoost, LightGBM, or CatBoost;
- applying systematic neural-network tuning;
- adding repeated or nested cross-validation;
- testing the model on an out-of-time or external dataset;
- evaluating model fairness across customer segments;
- adding a customer propensity scoring output;
- creating low-, medium-, and high-propensity campaign segments;
- building an interactive Power BI or Streamlit dashboard for campaign analysis;
- monitoring model drift and campaign conversion after deployment;
- comparing model targeting against a random or rule-based campaign baseline.

---

## Technology Stack

- Python
- Jupyter Notebook
- pandas
- Matplotlib
- Seaborn
- scikit-learn
- SciPy
- TensorFlow / Keras
- Tabulate

---

## Reproducibility

The current notebooks were executed using **Python 3.13.15**.

Install the required libraries using:

```powershell
python -m pip install -r requirements.txt
```

### Recommended Virtual Environment

From the project root:

```powershell
py -3.13 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

In VS Code, select the interpreter/kernel from:

```text
.venv
```

---

## Run Order

Run the notebooks in this order.

### Step 1 — Data Preparation

Open:

```text
notebooks/Phase1.ipynb
```

Expected input:

```text
data/raw/Bank_Personal_Loan_Modelling.csv
```

Expected output:

```text
data/processed/Phase1_cleaned.csv
```

### Step 2 — Predictive Modeling

Open:

```text
notebooks/Phase2.ipynb
```

Expected input:

```text
data/processed/Phase1_cleaned.csv
```

Phase 2 performs feature selection, model training, hyperparameter tuning, evaluation, cross-validation, statistical comparison, and business interpretation.

---

## Required Relative Paths

When the notebook working directory is `notebooks/`, use:

### Phase 1 raw-data input

```python
df_name = "../data/raw/Bank_Personal_Loan_Modelling.csv"
```

### Phase 1 processed-data output

```python
bank_personal_loan_dataset.to_csv(
    "../data/processed/Phase1_cleaned.csv",
    index=False
)
```

### Phase 2 processed-data input

```python
df_phase1 = "../data/processed/Phase1_cleaned.csv"
```

Using relative paths keeps the repository portable and avoids computer-specific directory references.

---

## Repository Purpose

This repository demonstrates an end-to-end customer analytics workflow that connects:

**business problem definition → data preparation → exploratory analysis → feature selection → predictive modeling → model validation → stakeholder interpretation**

The key business takeaway is that predictive modeling can help transform a broad personal-loan campaign into a more focused, measurable, and testable customer-targeting strategy. The Random Forest model provides the strongest predictive performance in the current cross-validated comparison, while the feature analysis highlights customer income, spending behavior, and existing banking relationships as important signals for campaign response.
