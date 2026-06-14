# Fashion Forward Forecasting - Recommendation Prediction Pipeline

A machine learning pipeline that predicts whether a customer recommends a clothing item based on their review. The pipeline combines numerical, categorical, and free-text data into a single scikit-learn workflow, using spaCy-based NLP processing and a tuned Random Forest classifier.

## What It Does

Given a customer review record - including the reviewer's age, the product's department/division/class, the review text, and engagement metrics - the pipeline predicts a binary outcome: whether the customer recommends the item (1) or not (0).

This is a common task for e-commerce platforms that want to surface "recommended" badges, flag potentially negative reviews for follow-up, or understand which product attributes and language patterns correlate with customer satisfaction.

## Pipeline Architecture

The pipeline is built entirely with scikit-learn's `Pipeline` and `ColumnTransformer`, so preprocessing and modeling run as one fitted object:

- **Numerical features** (Age, Positive Feedback Count): missing value imputation (mean) followed by Min-Max scaling
- **Categorical features** (Clothing ID, Division Name, Department Name, Class Name): missing value imputation (most frequent) followed by one-hot encoding
- **Text feature** (Review Text): custom spaCy-based lemmatizer that removes stop words, followed by TF-IDF vectorization (500 features)
- **Custom punctuation features**: hand-written transformers count exclamation marks, question marks, and commas in each review, since punctuation patterns can carry sentiment signal
- **Model**: Random Forest Classifier

All of the text-cleaning and feature-engineering steps are implemented as custom scikit-learn transformers (`SpacyLemmatizer`, `CharacterCounter`, `TextSelector`), so the entire pipeline - from raw CSV columns to prediction - can be saved, reloaded, and applied to new data with a single `.predict()` call.

## Model Tuning

The baseline Random Forest achieved 85.1% test accuracy. The pipeline was then tuned with `RandomizedSearchCV` (5-fold cross-validation) across:

- Number of trees (`n_estimators`)
- Tree depth (`max_depth`)
- Feature sampling strategy (`max_features`)
- TF-IDF vocabulary size

The best configuration found 150 estimators, no max depth limit beyond 20, and `max_features='sqrt'`, yielding a final test accuracy around 84%.

## Tech Stack

- Python
- pandas / numpy
- scikit-learn (Pipeline, ColumnTransformer, RandomForestClassifier, RandomizedSearchCV)
- spaCy (text lemmatization and stop-word removal)
- matplotlib (data exploration and evaluation visualizations)

## Project Structure

```
starter/
├── starter.ipynb       # Full pipeline: EDA, preprocessing, training, tuning, evaluation
├── README.md
└── data/
    └── reviews.csv      # Women's clothing e-commerce review dataset
```

## Getting Started

1. Install dependencies:
   ```
   pip install -r requirements.txt
   python -m spacy download en_core_web_sm
   ```
2. Launch the notebook:
   ```
   jupyter notebook starter/starter.ipynb
   ```
3. Run all cells. Training the full pipeline (including spaCy lemmatization) takes a couple of minutes; the hyperparameter search takes longer due to cross-validation across multiple model configurations.

## Dataset

The dataset is an anonymized set of women's clothing e-commerce reviews containing 8 features (numerical, categorical, and text) plus a binary target column, `Recommended IND`, indicating whether the reviewer recommends the product.
