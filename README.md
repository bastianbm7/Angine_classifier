# Exercise-Induced Angina Classification

Compares five classical ML classifiers (Logistic Regression, Naive Bayes, SVC, LDA, QDA) to predict exercise-induced angina in heart disease patients, using standard classification metrics plus ROC, precision-recall and learning curves to judge model quality.

## What it does

- **Data cleaning**: loads the heart disease dataset, renames columns, replaces invalid `?` entries with NaN, drops sparse/unused columns (`slope`, `ca`, `thal`), and casts numeric columns.
- **Preprocessing**: splits numeric (age, resting blood pressure, cholesterol, max heart rate, ST depression) and categorical (sex, chest pain type, fasting blood sugar, resting ECG, diagnosis) features, applying `StandardScaler` and `OneHotEncoder` via a `ColumnTransformer`.
- **Model comparison** (`Angine_prediction.ipynb`): trains Logistic Regression, Gaussian Naive Bayes, SVC, LDA and QDA on the same 80/20 split, then reports accuracy, precision, recall, F1 and ROC-AUC for each, plus confusion matrices, learning curves, ROC curves and precision-recall curves (interactive Plotly charts).
- **Static export** (`Angine_prediction.html`): rendered copy of the notebook with all outputs, viewable without running anything.

## Results

Logistic Regression and LDA come out on top overall (~0.79 accuracy, ~0.80 ROC-AUC), while Naive Bayes and QDA reach perfect recall (1.0) but at the cost of far more false positives.

## Tech stack

Python, pandas, scikit-learn, matplotlib, seaborn, Plotly.

## Project structure

```
Angine_prediction.ipynb   # Full pipeline: cleaning, preprocessing, 5 classifiers, metrics & plots
Angine_prediction.html    # Rendered notebook export
data.csv                  # Heart disease dataset
```

## How to run

```bash
pip install pandas matplotlib seaborn numpy plotly scikit-learn scipy
```

Open `Angine_prediction.ipynb` and run top to bottom — the notebook also pulls `data.csv` directly from this repo's raw GitHub URL, so it runs standalone even without the local file.
