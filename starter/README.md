# Fashion Forward Forecasting — Pipeline Project

## Project Summary

This project builds a machine learning pipeline to predict whether a customer
would recommend a women's clothing product based on their review. The model
is built for StyleSense, an online retailer with a backlog of reviews missing
recommendation data.

The pipeline handles numerical, categorical, and text data in a single unified
workflow — from raw data to prediction — using scikit-learn and spaCy.

## Results

- Baseline model accuracy: 85.1%
- Best model accuracy (after fine-tuning): 84.0%
- Cross-validation accuracy: 83.5% (5-fold)

## Files

| File | Description |
|------|-------------|
| `starter/starter.ipynb` | Main Jupyter Notebook with all code and analysis |
| `starter/data/reviews.csv` | Women's clothing e-commerce reviews dataset (18,442 rows) |
| `requirements.txt` | Python package dependencies |

## Pipeline Structure

The pipeline is built with scikit-learn and processes three types of data:

- **Numerical** (`Age`, `Positive Feedback Count`): imputed with mean, scaled with MinMaxScaler
- **Categorical** (`Clothing ID`, `Division Name`, `Department Name`, `Class Name`): imputed with most frequent value, one-hot encoded
- **Text** (`Review Text`): lemmatized with spaCy, vectorized with TF-IDF; punctuation counts (exclamations, questions, commas) added as extra features

A `RandomForestClassifier` is used as the final model. Hyperparameters were
tuned using `RandomizedSearchCV` with 5-fold cross-validation.

## How to Run

```bash