# 🌾 Crop Yield Prediction

![Python](https://img.shields.io/badge/Python-3.x-blue?logo=python)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-ML-orange?logo=scikit-learn)
![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-F37626?logo=jupyter)
![Status](https://img.shields.io/badge/Status-Complete-brightgreen)



Predict crop yield using environmental and agricultural features — temperature, precipitation, pesticide use, crop type, and region — across 10 different crops worldwide.

---

## 📋 Table of Contents

- [Project Overview](#project-overview)
- [Dataset](#dataset)
- [Exploratory Data Analysis](#exploratory-data-analysis)
- [Models & Results](#models--results)
- [Best Model — Gradient Boosting](#best-model--gradient-boosting)
- [Feature Importance](#feature-importance)
- [Installation](#installation)
- [Usage](#usage)
- [References](#references)

---

## Project Overview

The goal of this project is to build and train a regression model to predict crop yield based on a variety of factors including environmental, soil, and pesticide data. The insights can help reduce the land area needed for farming, reduce waste, and promote a more sustainable farming economy.

**6 regression models** were trained and evaluated:

| # | Model | Approach |
|---|-------|----------|
| 1 | Linear Regression | Baseline — polynomial features + interaction terms |
| 2 | Decision Tree | Tree-based, tuned via GridSearchCV |
| 3 | Random Forest | Ensemble of decision trees |
| 4 | Gradient Boosting | Sequential boosting (best model) |
| 5 | K-Nearest Neighbors | Instance-based learning |
| 6 | Neural Network (MLP) | 250-unit hidden layer |

---

## Dataset

Four CSV files merged into a single DataFrame:

| File | Description | Key Columns |
|------|-------------|-------------|
| `yield.csv` | Crop yield per country & year | `Area`, `Item`, `Year`, `Value` |
| `temp.csv` | Average annual temperature | `Area`, `Year`, `avg_temp` |
| `rainfall.csv` | Average annual precipitation | `Area`, `Year`, `avg_precipitation` |
| `pesticides.csv` | Pesticide usage in tonnes | `Area`, `Year`, `Pesticides` |

**After merging and cleaning:**

- Countries with fewer than 100 records were removed to reduce outlier skew
- Final dataset: **28,000+ rows** across 10 crop types
- Target variable: `Yield` (hg/ha)

**Crops covered:** Cassava · Maize · Plantains · Potatoes · Rice · Sorghum · Soybeans · Sweet Potatoes · Wheat · Yams

---

## Exploratory Data Analysis

### Key Findings

- **Average yield:** 77,047 hg/ha
- **Average temperature:** 20.5 °C
- **Average pesticide use:** 37,069 tonnes

### Correlations

The correlation matrix revealed:
- Weak negative correlation between pesticide use and average precipitation
- Weak negative correlation between crop type (`Item`) and yield
- No strong linear relationship between any single feature and yield — this ruled out simple linear models early

### Temperature Range by Crop

| Crop | Climate Type | Notes |
|------|-------------|-------|
| Maize, Wheat, Rice, Potatoes | Wide range | Grow in varied temperatures |
| Soybeans, Sorghum | Wide range | Adaptable crops |
| Cassava, Yams, Plantains, Sweet Potatoes | Narrow, warm | Tropical/warm climate only |

### Pesticide vs Yield

Higher pesticide use does **not** consistently lead to higher yield. Most high yields occur at lower pesticide usage levels, with only a few outliers at higher usage.

---

## Models & Results

All models were trained with:
- **Train/Test split:** 80% / 20% (`random_state=42`)
- **Preprocessing:** `OneHotEncoder` for categorical features (`Item`, `Area`), `StandardScaler` for numerical features
- **Hyperparameter tuning:** `GridSearchCV` with 5-fold cross-validation

### Performance Comparison

| Model | Train R² | Test R² | Train MAE | Test MAE | Train RMSE | Test RMSE |
|-------|:--------:|:-------:|----------:|---------:|-----------:|----------:|
| **Gradient Boosting** ⭐ | **0.999** | **0.978** | **940** | 19,008 | **2,099** | **11,893** |
| Random Forest | 0.996 | 0.978 | 1,847 | **4,618** | 4,805 | 12,088 |
| Decision Tree | 0.996 | 0.971 | 1,511 | 4,869 | 5,175 | 13,837 |
| K-Nearest Neighbors | 0.999 | 0.555 | 144 | 28,741 | 1,615 | 54,524 |
| Neural Network (MLP) | 0.771 | 0.762 | 25,961 | 26,147 | 39,647 | 39,810 |
| Linear Regression | 0.705 | 0.697 | 27,566 | 27,570 | 45,028 | 44,948 |

> **R²** (higher = better) &nbsp;|&nbsp; **MAE / RMSE** (lower = better)

### Model Verdicts

| Model | Verdict | Reason |
|-------|---------|--------|
| Gradient Boosting | ✅ Best overall | Lowest RMSE on test data, good generalization |
| Random Forest | ✅ Runner-up | Same R², but lowest test MAE (4,618) |
| Decision Tree | ✅ Good | Solid R², slightly more overfit than ensembles |
| KNN | ❌ Severely overfit | Train R² = 0.999 → Test R² = 0.555 |
| Neural Network | ⚠️ Underfit | Consistent but low accuracy (~0.76) |
| Linear Regression | ⚠️ Weak | Crop-yield relationship is non-linear |

---

## Best Model — Gradient Boosting

**Optimal hyperparameters** (found via GridSearchCV):

```python
GradientBoostingRegressor(
    n_estimators=300,
    learning_rate=0.2,
    max_depth=10,
    random_state=42
)
```

**Final scores:**

```
Training R²  :  0.9990
Testing R²   :  0.9780
Training RMSE:  2,099
Testing RMSE :  11,893
```

**Cross-validation** (5-fold on training set) confirmed stable performance with low variance across folds.

**Learning curve** analysis showed the model benefits from more data — the gap between training and CV scores suggests slight overfitting that could be reduced with a larger dataset.

---

## Feature Importance

Two methods were used to assess feature importance on the Gradient Boosting model:

### 1. Built-in Feature Importance
Ranks features by how much they reduce impurity across all trees.
- **Top features:** Crop type (`Item`), Region (`Area`), then climatic factors

### 2. Permutation Importance
Shuffles each feature and measures performance drop — more reliable on correlated features.
- **Top features:** Crop type and Region dominate, confirming climatic variables have secondary influence

> **Insight:** Crop species and geographic region are the biggest predictors of yield. Climatic factors (temperature, precipitation, pesticides) matter but crops show high tolerance to environmental variation.

---

## Installation

```bash
# Clone the repository
git clone https://github.com/your-username/crop-yield-prediction.git
cd crop-yield-prediction

# Install dependencies
pip install numpy pandas matplotlib seaborn scikit-learn plotly statsmodels
```

**Required data files** (place in `Data/` folder):

```
Data/
├── yield.csv
├── temp.csv
├── rainfall.csv
└── pesticides.csv
```

---

## Usage

Open and run the notebook:

```bash
jupyter notebook crop_yield.ipynb
```

Run all cells in order. The GridSearch cell is commented out by default (very slow) — best parameters are already applied in the model cells.

---

## References

1. [FAO — FAOSTAT Data](https://www.fao.org/faostat/en/) (2023)
2. [Our World in Data — Crop Yields](https://ourworldindata.org/crop-yields) (2022)
3. [OECD — Crop Production](https://data.oecd.org/agroutput/crop-production.html) (2023)
4. [Geopard — Predicting crop yield with remote sensing](https://geopard.tech/blog/predicting-crop-yield-with-remote-sensing-data/) (2023)
5. [USDA — Food Security Status 2022](https://www.ers.usda.gov/topics/food-nutrition-assistance/food-security-in-the-u-s/) (2023)

---

*Built with Python · scikit-learn · pandas · matplotlib · seaborn · plotly*
