# Python Yuva Internship

Repository containing assignments, projects, and pipeline code for the Python Yuva Internship.

## Directory Structure

- **`week 1/`**: Titanic Dataset - Data Acquisition, Cleaning & Preprocessing Pipeline
  - `data_cleaning_pipeline.py`: End-to-end data cleaning, EDA, imputation, outlier handling, feature engineering, and encoding.
  - `Week1_Data_Cleaning_Report.docx`: Comprehensive project documentation and cleaning report.
  - `titanic_cleaned.csv`: Cleaned dataset with imputed values and standard formatting.
  - `titanic_model_ready.csv`: Encoded and scaled dataset ready for machine learning modeling.

- **`week 2/`**: Titanic Dataset - Exploratory Data Analysis & Visualization
  - `eda_titanic.py`: Reproducible exploratory data analysis and visualization script.
  - `Week2_EDA_Visualization_Report.docx`: Comprehensive project documentation and analysis report.
  - `titanic_cleaned.csv`: Cleaned dataset used for analysis.
  - `submission_description.txt`: Project overview and submission summary.
  - `figures/`: Report-ready EDA charts and plots.

- **`week 3/`**: Iris Dataset - Unsupervised Learning & Clustering Analysis
  - `clustering_iris.py`: Python script for feature scaling, K-Means clustering, PCA, and hierarchical clustering.
  - `Week3_Unsupervised_Clustering_Report.docx`: Comprehensive project documentation and clustering report.
  - `iris_clustered.csv`: Iris dataset with cluster assignments.
  - `cluster_selection_metrics.csv`: Inertia and silhouette score evaluation metrics for K selection.
  - `submission_description.txt`: Project overview and clustering methodology summary.
  - `figures/`: Report-ready clustering charts, elbow curves, PCA, and dendrograms.

- **`week 4/`**: Breast Cancer Dataset - Supervised Learning & Binary Classification
  - `supervised_learning_breast_cancer.py`: Python script for leakage-safe ML pipeline, cross-validation, GridSearchCV tuning, and evaluation.
  - `Week4_Supervised_Learning_Report.docx`: Comprehensive project documentation and classification report.
  - `breast_cancer_classification_dataset.csv`: Dataset with diagnostic features and target labels.
  - `test_predictions.csv`: Model test set predictions with probabilities and class labels.
  - `submission_description.txt`: Project overview and supervised modeling summary.
  - `figures/`: Report-ready performance charts, ROC curve, confusion matrix, and permutation importances.

- **`week 5/`**: Digits Dataset - Deep Learning & Convolutional Neural Network (CNN)
  - `deep_learning_digits.py`: PyTorch script for CNN architecture, training loop, early stopping, and evaluation.
  - `Week5_Deep_Learning_Report.docx`: Comprehensive project documentation and deep learning report.
  - `digits_dataset.csv`: Tabular copy of the 8x8 grayscale digits dataset.
  - `training_history.csv`: Epoch-by-epoch training and validation loss and accuracy metrics.
  - `evaluation_metrics.csv`: Final test performance metrics.
  - `confusion_matrix.csv`: Test set confusion matrix.
  - `submission_description.txt`: Project overview and deep learning summary.
  - `figures/`: Report-ready loss/accuracy curves, confusion matrix, architecture diagram, and prediction samples.

- **`week 6/`**: Titanic Dataset - Integrative Capstone Project (End-to-End Pipeline)
  - `capstone_pipeline.py`: Comprehensive Python script integrating data cleaning, EDA, supervised learning (Random Forest), and unsupervised learning (K-Means).
  - `Week6_Integrative_Capstone_Report.docx`: Final capstone project report and documentation.
  - `titanic_cleaned.csv`: Cleaned dataset used for the capstone pipeline.
  - `titanic_capstone_enriched.csv`: Enriched dataset with engineered features and cluster assignments.
  - `supervised_metrics.csv`: Performance metrics for the Random Forest model.
  - `supervised_confusion_matrix.csv`: Test confusion matrix for supervised evaluation.
  - `cross_validation_summary.csv`: 5-fold cross-validation performance summary.
  - `feature_importance.csv`: Feature importance scores from Random Forest.
  - `cluster_selection_metrics.csv`: Inertia and silhouette metrics across candidate k values.
  - `cluster_profiles.csv`: Mean feature profiles for passenger segments.
  - `submission_description.txt`: Project overview and capstone summary.
  - `figures/`: Report-ready EDA plots, model performance, feature importance, and cluster PCA charts.





