# Football Prediction & Evaluation Pipeline

This document outlines the technical process for evaluating pre-trained football prediction models (both pre-match and in-play) and for visualizing their outputs. The pipeline consists of several Jupyter notebook files and pre-trained model files.

## 1. Project Overview

The primary goals of the code are:
1.  **Model Evaluation (`evaluation.ipynb`):** To assess the performance of models pre-trained on [football match dataset](https://github.com/xgabora/Club-Football-Match-Data-2000-2025).
2.  **Prediction Visualization (`example.ipynb`):** To generate predictions for a sample match using the in-play model and visualize its output, including a detailed goal probability heatmap.

## 2. Project Structure

A recommended project structure is as follows:

```
football_prediction_suite/
├── evaluation.ipynb 
├── example.ipynb
├── training/
│   └── football_data.csv
├── models/
│   ├── pre_match/ 
│   └── in_play/   
├── example_output/
└── requirements.txt
```

## 3. Initial Setup and Requirements

### 3.1. Python Environment
*   Python 3.8+ is recommended.
*   Using a virtual environment (e.g., `venv`) is highly advised to manage dependencies consistently.
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Linux/macOS
    venv\Scripts\activate     # On Windows
    ```

### 3.2. Required Libraries
Install the necessary Python libraries using the provided `requirements.txt` file:
```bash
pip install -r requirements.txt
```
The `requirements.txt` file should contain at least:
```
scikit-learn
pandas
numpy
tensorflow
keras
xgboost
lightgbm
seaborn
matplotlib
scipy
```
For GPU support with TensorFlow, ensure proper CUDA/cuDNN setup and the corresponding TensorFlow build.

### 3.3. Data and Model Files

*   **Historical Data:**
    *   `training/football_data.csv`: A comprehensive CSV file containing curated historical match data, also [available here](https://github.com/xgabora/Club-Football-Match-Data-2000-2025). This file is the primary input for the `evaluation.ipynb` script and may be referenced by `example.ipynb` if pre-match preprocessor fitting were active (though typically skipped for direct odds-based pre-match probabilities in the example).
*   **Pre-Match Models:**
    *   `models/pre_match/full_features.keras`: Keras neural network model trained with a full pre-match feature set.
    *   `models/pre_match/reduced_features.keras`: Keras neural network model trained with a reduced pre-match feature set - fallback.
*   **In-Play Model Components (in `models/in_play/`):**
This directory contains all the serialized Python objects (pickled files) that constitute the in-play stacked prediction model. Each file plays a specific role:
    *   `base_draw_specialist.pkl`: A base model specifically trained to predict the draw.
    *   `base_hda_classifier.pkl`: A base model trained to predict the Home/Draw/Away (H/D/A) outcome directly, using pre-play guess as well.
    *   `base_lgbm_away.pkl`: A LightGBM base model trained to predict the number of goals scored by the away team.
    *   `base_lgbm_home.pkl`: A LightGBM base model trained to predict the number of goals scored by the home team.
    *   `base_nn_away.pkl`: A Neural Network base model trained to predict away team goals.
    *   `base_nn_home.pkl`: A Neural Network base model trained to predict home team goals.
    *   `base_xgb_away.pkl`: An XGBoost base model trained to predict away team goals.
    *   `base_xgb_home.pkl`: An XGBoost base model trained to predict home team goals.
    *   `cluster_medians.pkl` (Optional): A dictionary containing the median values for cluster-related features. Used for imputing missing cluster feature values.
    *   `draw_calibrator.pkl`: A scikit-learn calibration model used to calibrate the raw probability outputs of draws.
    *   `feature_names.pkl`: A list containing the exact names and order of input features expected by the base models (after feature engineering) and the `scaler.pkl`.
    *   `label_encoder.pkl`: A scikit-learn `LabelEncoder` object fitted on the H/D/A outcome labels. Used to convert text labels to numerical and back.
    *   `meta_feature_names.pkl`: A list specifying the names and order of features expected by the `meta_model.pkl`.
    *   `meta_model.pkl`: The final meta-learner (LightGBM) in the stacked ensemble. It takes the predictions of all base models as input.
    *   `scaler.pkl`: A scikit-learn scaler fitted on the training data features (defined in `feature_names.pkl`).
    *   `target_transformer.pkl`: A scikit-learn transformer fitted on the target goal values during the training of the Neural Network base models.
All specified data and model files must be present in their respective directories, or the paths within the scripts must be updated accordingly.

## 4. Script Descriptions

### 4.1. `evaluation.ipynb` (Model Performance Evaluation)

This script is designed to rigorously evaluate the performance of both pre-match and in-play prediction models using historical data.

#### Workflow:

1.  **Global Setup:**
    *   Imports libraries and configures warnings/display options.
    *   Defines paths to data and model files.
    *   Specifies target column names, mappings, feature sets for different models, and evaluation parameters (e.g., subset size).
    *   Sets random seeds for reproducibility.

2.  **Pre-Match Model Evaluation:**
    *   **Data Loading & Cleaning (`load_and_clean_data`):** Loads selected columns from `training/football_data.csv`, parses dates, handles invalid odds, and filters for valid target outcomes.
    *   **Feature Engineering (`engineer_features`):** Creates time-based features (Year, Month, DayOfWeek) and differential features (EloDiff, FormDiffs) from the cleaned data.
    *   **Preprocessor Fitting (`fit_preprocessors`):** Fits `SimpleImputer`, `StandardScaler`, `OneHotEncoder`, and `LabelEncoder` on the dataset. This simulates fitting preprocessors on a training set. Stores fitted preprocessors and the columns they were fitted on.
    *   **Evaluation Subset Preparation:** Selects the most recent N matches for evaluation, applying further filtering based on data requirements for the full Keras model. Extracts true target labels.
    *   **Keras Data Transformation (`transform_data_for_keras`):** Transforms the evaluation data subset using the fitted preprocessors into the multi-input list format expected by the Keras models.
    *   **Model Evaluation Function (`evaluate_model`):**
        *   Loads a specified Keras model (`.keras` file).
        *   Prepares input data using `transform_data_for_keras`.
        *   Generates predictions and calculates accuracy and macro F1-score.
        *   Includes error handling and memory management (clearing Keras session).
    *   **Execution:** Runs `evaluate_model` for both the "full_features" and "reduced_features" Keras models.

3.  **In-Play Model Evaluation:**
    *   **Component Loading (`load_pipeline_prerequisites_from_inplay_dir`):** Loads all `.pkl` components of the in-play stacked model from the `models/in_play/` directory.
    *   **Data Loading & Full Feature Engineering (`load_and_engineer_data_like_training`):** Loads `training/football_data.csv` and applies the exact feature engineering pipeline used during the training of the in-play model.
    *   **Evaluation Subset Preparation & Scaling:**
        *   Aligns the fully engineered data with the features expected by the in-play model (from `feature_names.pkl`).
        *   Selects the most recent N matches for evaluation from both scaled and original engineered data.
        *   Extracts true scores and H/D/A outcomes for the evaluation subset using the in-play `label_encoder.pkl`.
    *   **Full Pipeline Prediction (`predict_with_full_pipeline`):**
        *   Generates predictions using the entire in-play stack: base model predictions, NN goal transformations, draw probability calibration, meta-feature construction including cluster features, and final meta-model prediction.
    *   **Metrics Calculation (`calculate_and_format_metrics_ots_style`):** Calculates a comprehensive suite of metrics: outcome accuracy, exact score accuracy, goal difference accuracy, R², MSE, MAE for goals, Brier score, and ROC AUC for H/D/A probabilities.
    *   **Execution:** Runs the in-play prediction and metrics calculation pipeline.

#### Key Considerations for `evaluation.ipynb`:
*   **Data-Model Consistency:** The historical data in `football_data.csv` and the feature engineering steps within the script must align with how the models were originally trained.
*   **Path Accuracy:** Correct paths to data and all model files are critical.

### 4.2. `example.ipynb` (Prediction Visualization)

This script demonstrates how to use the pre-trained in-play model to make predictions for a new, unseen sample match at half time, and visualizes the predicted goal distribution.

#### Workflow:

1.  **Global Setup:**
    *   Imports necessary libraries.
    *   Defines paths for loading the in-play model components (`INPLAY_LOAD_DIR`) and saving visualizations (`VIZ_OUTPUT_DIR`).
    *   Defines visualization parameters (e.g., `HEATMAP_MAX_GOALS`).
    *   **Sample Match Data Definition:** A small sample match ("Arsenal vs Real Madrid") is defined directly within the script as a Python dictionary, including match details, half-time score, and current statistics, pre-match prediction market odds, Elo, and form. This is converted to a DataFrame.

2.  **Feature Engineering:**
    *   **`inplay_load_pipeline_components(load_dir)`:** Loads all `.pkl` files for the in-play model pipeline.
    *   **`combine_odds(odd1, odd2)`:** Utility to calculate combined odds.
    *   **`inplay_feature_engineering(df_orig, model_expected_feature_names)`:** Replicates the in-play model's training feature engineering process on the sample match data.

3.  **Prediction and Visualization Process:**
    *   **Pre-Match Odds Probabilities:** Calculates and prints 3-way H/D/A probabilities derived directly from the sample match's pre-match odds.
    *   **In-Play Model Prediction (for current/HT state):** Loads the model stack, aligns the data, generates goal predictions and probabilities from all base models, creates meta-features and gets final probabilities for total goals and H/D/A.
    *   **Goal Distribution Heatmap Generation:** Uses average predicted home and away goals as lambdas to create the probability matrix for final scores. Uses seaborn heatmap to create visual representation of the scoreline probabilities, saved into 'VIZ_OUTPUT_DIR' folder.

#### Key Considerations for `example.ipynb`:
*   **In-Play Model Path:** The `INPLAY_LOAD_DIR` must correctly point to the directory containing all in-play model components.
*   **Sample Data Accuracy:** The statistics provided for the sample match (especially those representing the "current" in-play state) should be realistic for a meaningful demonstration.

## 5. Final Thoughts and Considerations

### 5.1. Current State

The scripts successfully asses the data, calculate probabilties and evaluate the guesses. The script works with and is based on this [dataset](https://github.com/xgabora/Club-Football-Match-Data-2000-2025). The process is modular, with each script created from multiple Jupyter code blocks.

### 5.2. Key Considerations

*   **Path Management:** Using absolute paths or constructing paths relative to the script's location (e.g., using `os.path.dirname(os.path.abspath(__file__))`) can make the scripts more portable than relying on the current working directory.
*   **Dependency Management:** Strictly use the `requirements.txt` and a virtual environment to ensure consistent library versions across different setups.
*   **Model Versioning:** If models are updated, ensure that the evaluation scripts are either updated to match any changes in feature sets or component names, or that different versions of the evaluation scripts are maintained for different model versions.
*   **Error Handling:** The scripts include `try-except` blocks but ensure that important error messages are not overly suppressed during development or debugging.
