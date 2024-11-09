# Kaggle Titanic Survival Prediction Project

This repository contains a comprehensive solution for the Kaggle Titanic Machine Learning competition. The primary objective is to predict passenger survival on the Titanic using a variety of machine learning algorithms. The project utilizes data preprocessing, feature engineering, and model selection to achieve accurate predictions.

## Project Structure

- **`titanic.ipynb`**: Jupyter notebook containing all data analysis, feature engineering, model training, and evaluation steps.
- **`requirements.txt`**: Lists all necessary Python packages to set up the environment.
- **`README.md`**: This file, providing an in-depth overview of the project, including installation and usage instructions.

## Environment Setup

### Step 1: Install Anaconda (Python 3.10)

1. Download and install [Anaconda](https://www.anaconda.com/products/distribution) for Python 3.10.
2. Create a new environment for the project:
   ```bash
   conda create -n titanic_env python=3.10
   ```
3. Activate the environment:
   ```bash
   conda activate titanic_env
   ```

### Step 2: Install Required Libraries

1. Ensure you are in the project directory where `requirements.txt` is located.
2. Install the libraries:
   ```bash
   pip install -r requirements.txt
   ```

### Step 3: Run the Notebook

To explore the analysis and results, open `titanic.ipynb` in Jupyter Notebook or Jupyter Lab:

```bash
jupyter notebook titanic.ipynb
```

## Libraries and Tools Used

- **Numpy** and **Pandas**: For data manipulation and analysis.
- **Seaborn** and **Matplotlib**: For data visualization.
- **Scikit-learn**: For data preprocessing, model training, evaluation, and hyperparameter tuning.
- **XGBoost**: For gradient-boosted decision trees.
- **CatBoost**: For handling categorical data efficiently with gradient boosting.
- **Warnings**: To ignore unnecessary warnings during model training and evaluation.

## Data Preprocessing and Feature Engineering

The notebook preprocesses and engineers features by:

- Handling missing values
- Encoding categorical variables
- Scaling numerical features
- Creating new features based on existing data

## Algorithms Used and Hyperparameter Details

This project explores multiple machine learning algorithms. Each model’s hyperparameters, advantages, and disadvantages are detailed below.

### 1. Logistic Regression

- **Hyperparameters**:
  - `C`: Controls the regularization strength. Higher values reduce regularization.
  - `solver`: The algorithm to use in optimization (`liblinear` or `lbfgs`).
- **Advantages**: Simple, interpretable, and works well with linearly separable data.
- **Disadvantages**: Limited to linear relationships; sensitive to outliers.

### 2. Random Forest Classifier

- **Hyperparameters**:
  - `n_estimators`: Number of trees in the forest.
  - `max_depth`: Maximum depth of each tree.
  - `min_samples_split`: Minimum number of samples to split a node.
  - `min_samples_leaf`: Minimum samples at each leaf node.
- **Advantages**: Handles non-linearity, reduces overfitting, and works well with missing data.
- **Disadvantages**: Slower training and prediction for large datasets; complex to interpret.

### 3. Support Vector Machine (SVM)

- **Hyperparameters**:
  - `C`: Controls the margin hardness.
  - `kernel`: Defines the kernel type (e.g., `linear`, `rbf`).
- **Advantages**: Effective in high-dimensional spaces, robust with outliers.
- **Disadvantages**: Slower with large datasets; performance can degrade with noisy data.

### 4. K-Nearest Neighbors (KNN)

- **Hyperparameters**:
  - `n_neighbors`: Number of neighbors to consider.
  - `weights`: Determines weight function (`uniform` or `distance`).
- **Advantages**: Simple and interpretable, non-parametric.
- **Disadvantages**: Computationally expensive, sensitive to irrelevant features.

### 5. Gradient Boosting Classifier

- **Hyperparameters**:
  - `n_estimators`: Number of boosting rounds.
  - `learning_rate`: Reduces each tree’s contribution.
  - `max_depth`: Depth of each tree.
- **Advantages**: Reduces overfitting, highly accurate with parameter tuning.
- **Disadvantages**: Can be slow to train; sensitive to outliers and noisy data.

### 6. XGBoost

- **Hyperparameters**:
  - `n_estimators`, `learning_rate`, `max_depth`, and `subsample` control the boosting and model structure.
- **Advantages**: Fast and accurate with built-in cross-validation.
- **Disadvantages**: Resource-intensive; requires careful tuning.

### 7. CatBoost Classifier

- **Hyperparameters**:
  - `iterations`, `depth`, `learning_rate`, and `l2_leaf_reg` for tuning boosting and tree depth.
- **Advantages**: Handles categorical data without encoding; efficient with large datasets.
- **Disadvantages**: Computationally demanding; longer training time.

### 8. Artificial Neural Network (MLPClassifier)

- **Hyperparameters**:
  - `hidden_layer_sizes`, `activation`, `solver`, and `alpha`.
- **Advantages**: Handles non-linear relationships, highly customizable.
- **Disadvantages**: Requires tuning; sensitive to data scaling and can overfit.

## Evaluation Metrics

- **Accuracy Score**: Measures overall model accuracy.
- **Confusion Matrix**: Provides insight into true vs. predicted classes.
- **Classification Report**: Summarizes precision, recall, and F1-score.

## Model Comparison and Results

The notebook compares each model’s performance using a blend of accuracy, precision, recall, and F1-score. Hyperparameter tuning (via `GridSearchCV`) is applied to enhance performance. The models are evaluated to find the optimal balance between overfitting and generalization.

## Conclusion

This project demonstrates the effectiveness of machine learning algorithms in predicting Titanic passenger survival. Different models bring unique advantages and are optimized for the task at hand. Please refer to the `titanic.ipynb` notebook for in-depth code, analysis, and model performance.

## Acknowledgements

- Kaggle for the Titanic dataset.
- Open-source contributors for valuable machine learning libraries.
