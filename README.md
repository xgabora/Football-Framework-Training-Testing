# Football Prediction & Evaluation Pipeline

This project encompasses scripts for evaluating pre-trained football prediction models (both pre-match and in-play) and for visualizing their outputs, particularly focusing on goal distribution heatmaps for demonstration.

## 1. Project Overview

The primary goals are:
1.  **Model Evaluation (`evaluation.ipynb`):** To assess the performance of:
    *   Pre-match Keras-based neural network models (with full and reduced feature sets).
    *   A complex in-play stacked ensemble model.
2.  **Prediction Visualization (`example.ipynb`):** To generate predictions for a sample match using the in-play model and visualize its output, including a detailed goal probability heatmap.

## 2. Project Structure

A recommended project structure is as follows:

```
football_prediction_suite/
├── evaluation.ipynb          # Script for evaluating model performance
├── example.ipynb             # Script for demonstrating predictions & visualization
├── training/
│   └── football_data.csv     # Main historical match data CSV (curated)
├── models/
│   ├── pre_match/
│   │   ├── full_features.keras   # Pre-trained Keras model (full feature set)
│   │   └── reduced_features.keras # Pre-trained Keras model (reduced feature set)
│   └── in_play/                # Directory containing all in-play model components
│       ├── scaler.pkl
│       ├── feature_names.pkl
│       ├── meta_model.pkl
│       └── ... (all other .pkl base models and components)
├── example_output/           # Directory for visualizations from example.ipynb
│   └── Arsenal_vs_Real_Madrid_goal_distribution_heatmap.png # Example output
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
pandas
numpy
scikit-learn
tensorflow
xgboost
lightgbm
scipy
matplotlib
seaborn
# openpyxl (if any intermediate excel interactions are planned)
```
For GPU support with TensorFlow, ensure proper CUDA/cuDNN setup and the corresponding TensorFlow build.

### 3.3. Data and Model Files

*   **Historical Data:**
    *   `training/football_data.csv`: A comprehensive CSV file containing curated historical match data. This file is the primary input for the `evaluation.ipynb` script and may be referenced by `example.ipynb` if pre-match preprocessor fitting were active (though typically skipped for direct odds-based pre-match probabilities in the example).
*   **Pre-Match Models:**
    *   `models/pre_match/full_features.keras`: Keras model trained with a full pre-match feature set.
    *   `models/pre_match/reduced_features.keras`: Keras model trained with a reduced pre-match feature set.
*   **In-Play Model Components (in `models/in_play/`):**
    *   A collection of `.pkl` files representing the entire in-play stacked model pipeline. This includes scalers, feature name lists, label encoders, base models (NNs, XGBoost, LightGBM), a draw calibrator, meta-feature definitions, and the final meta-model. Specific filenames are hardcoded within the scripts (e.g., `scaler.pkl`, `meta_model.pkl`).

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
    *   **Preprocessor Fitting (`fit_preprocessors`):** Fits `SimpleImputer`, `StandardScaler`, `OneHotEncoder`, and `LabelEncoder` on the *entire engineered dataset*. This simulates fitting preprocessors on a training set. Stores fitted preprocessors and the columns they were fitted on.
    *   **Evaluation Subset Preparation:** Selects the most recent N matches for evaluation, applying further filtering based on data requirements for the full Keras model. Extracts true target labels.
    *   **Keras Data Transformation (`transform_data_for_keras`):** Transforms the evaluation data subset using the fitted preprocessors into the multi-input list format expected by the Keras models. This includes numerical scaling, one-hot encoding, and label encoding for embedding layers (handling unknown categories).
    *   **Model Evaluation Function (`evaluate_model`):**
        *   Loads a specified Keras model (`.keras` file).
        *   Prepares input data using `transform_data_for_keras`.
        *   Generates predictions and calculates accuracy and macro F1-score.
        *   Includes error handling and memory management (clearing Keras session).
    *   **Execution:** Runs `evaluate_model` for both the "full_features" and "reduced_features" Keras models.

3.  **In-Play Model Evaluation:**
    *   **Component Loading (`load_pipeline_prerequisites_from_inplay_dir`):** Loads all `.pkl` components (scaler, feature lists, base models, meta-model, etc.) of the in-play stacked model from the `models/in_play/` directory.
    *   **Data Loading & Full Feature Engineering (`load_and_engineer_data_like_training`):** Loads `training/football_data.csv` and applies the *exact, extensive feature engineering pipeline* used during the training of the in-play model. This is crucial for consistency.
    *   **Evaluation Subset Preparation & Scaling:**
        *   Aligns the fully engineered data with the features expected by the in-play model (from `feature_names.pkl`).
        *   Imputes remaining NaNs and scales features using the loaded in-play `scaler.pkl`.
        *   Selects the most recent N matches for evaluation from both scaled and original engineered data.
        *   Extracts true scores and H/D/A outcomes for the evaluation subset using the in-play `label_encoder.pkl`.
    *   **Full Pipeline Prediction (`predict_with_full_pipeline`):**
        *   Generates predictions using the entire in-play stack: base model predictions, NN goal transformations (if applicable), draw probability calibration, meta-feature construction (including cluster features if configured), and final meta-model prediction.
    *   **Metrics Calculation (`calculate_and_format_metrics_ots_style`):** Calculates a comprehensive suite of metrics: outcome accuracy, exact score accuracy, goal difference accuracy, R², MSE, MAE for goals, Brier score, and ROC AUC for H/D/A probabilities.
    *   **Execution:** Runs the in-play prediction and metrics calculation pipeline.

4.  **Summary Output:**
    *   Prints a formatted summary of evaluation metrics to the console for all evaluated models. Pre-match Keras model results are typically presented in a Markdown table. In-play model results include a more detailed list of metrics.

#### Key Considerations for `evaluation.ipynb`:
*   **Data-Model Consistency:** The historical data in `football_data.csv` and the feature engineering steps within the script must perfectly align with how the models were originally trained.
*   **Preprocessor Fitting Strategy:** Pre-match preprocessors are fitted on the entire dataset available to the script before evaluation. This is a common strategy for offline evaluation to ensure consistent transformations.
*   **Path Accuracy:** Correct paths to data and all model files are critical.

### 4.2. `example.ipynb` (Prediction Visualization)

This script demonstrates how to use the pre-trained in-play model to make predictions for a new, sample match (e.g., at half-time) and visualizes the predicted goal distribution.

#### Workflow:

1.  **Global Setup:**
    *   Imports necessary libraries.
    *   Defines paths for loading the in-play model components (`INPLAY_LOAD_DIR`) and saving visualizations (`VIZ_OUTPUT_DIR`).
    *   Defines visualization parameters (e.g., `HEATMAP_MAX_GOALS`).
    *   **Sample Match Data Definition:** A small sample match (e.g., "Arsenal vs Real Madrid") is defined directly within the script as a Python dictionary, including match details, half-time score, and current (or assumed half-time) statistics, pre-match odds, Elo, and form. This is converted to a Pandas DataFrame.

2.  **Helper Functions:**
    *   **`inplay_load_pipeline_components(load_dir)`:** Loads all `.pkl` files for the in-play model pipeline (same as in `evaluation.ipynb`).
    *   **`combine_odds(odd1, odd2)`:** Utility to calculate combined odds.
    *   **`inplay_feature_engineering(df_orig, model_expected_feature_names)`:** Replicates the in-play model's training feature engineering process on the sample match data. Crucially adapted to handle a single new data point (e.g., using default values for missing historical context like form if not provided).
    *   **`generate_bivariate_poisson_matrix(lambda_home, lambda_away, max_goals_axis)`:** Generates a scoreline probability matrix using a bivariate Poisson distribution based on predicted average home and away goals (lambdas).

3.  **Prediction and Visualization Process:**
    *   **Pre-Match Odds Probabilities:** Calculates and prints 3-way H/D/A probabilities derived directly from the sample match's pre-match odds.
    *   **In-Play Model Prediction (for current/HT state):**
        1.  **Load Components:** Loads the in-play model stack.
        2.  **Feature Engineering:** Applies `inplay_feature_engineering` to the sample match DataFrame.
        3.  **Data Alignment & Scaling:** Ensures all features expected by the in-play model are present, imputes NaNs, and scales the features using the loaded in-play scaler.
        4.  **Base Model Predictions:** Generates goal predictions and probabilities from all base models.
        5.  **NN Goal Transformation & Draw Calibration:** Applies these steps if configured.
        6.  **Meta-Feature Construction:** Creates meta-features from base model outputs.
        7.  **Meta-Model Prediction:**
            *   Gets final H/D/A probabilities for the current state of the match (e.g., half-time) from the meta-model. These are printed.
            *   Calculates average predicted *total* home goals (`avg_pred_h_final`) and *total* away goals (`avg_pred_a_final`) from the base regression models. These serve as lambdas for the heatmap.
    *   **Goal Distribution Heatmap Generation:**
        1.  **Poisson Matrix:** Uses `avg_pred_h_final` and `avg_pred_a_final` as lambdas to generate the probability matrix for final scores via `generate_bivariate_poisson_matrix`.
        2.  **Impossible Scores Masking:** Identifies and visually marks scores that are impossible given the provided half-time score (e.g., final score cannot be less than HT score for a team).
        3.  **Annotation & Plotting:** Uses `seaborn.heatmap` to create a visual representation of the scoreline probabilities. The heatmap is configured with a bottom-left origin (0-0 score), appropriate labels, title (including HT score), and probability annotations in each cell.
        4.  **Saving:** Saves the heatmap image (e.g., PNG) to `VIZ_OUTPUT_DIR` and prints a save confirmation.
    *   **Error Handling:** The prediction and visualization steps are within a `try-except` block for robustness.
    *   **Minimized Console Output:** The script is configured to provide minimal console output, focusing on the key probability figures and the heatmap save location.

#### Key Considerations for `example.ipynb`:
*   **In-Play Model Path:** The `INPLAY_LOAD_DIR` must correctly point to the directory containing all in-play model components.
*   **Sample Data Accuracy:** The statistics provided for the sample match (especially those representing the "current" in-play state) should be realistic for a meaningful demonstration.
*   **Feature Engineering for Single Instance:** The `inplay_feature_engineering` function needs to correctly handle a single new data instance, especially for features that normally rely on historical data (like form, which might need to be defaulted or provided directly in the sample).

## 5. General Considerations and Best Practices

*   **Path Management:** Using absolute paths or constructing paths relative to the script's location (e.g., using `os.path.dirname(os.path.abspath(__file__))`) can make the scripts more portable than relying on the current working directory.
*   **Dependency Management:** Strictly use the `requirements.txt` and a virtual environment to ensure consistent library versions across different setups.
*   **Model Versioning:** If models are updated, ensure that the evaluation scripts are either updated to match any changes in feature sets or component names, or that different versions of the evaluation scripts are maintained for different model versions.
*   **Error Handling:** The scripts include `try-except` blocks, which are good for preventing crashes but ensure that important error messages are not overly suppressed during development or debugging.
*   **Modular Functions:** Breaking down complex tasks into smaller, well-defined functions (as done in both scripts) improves readability, maintainability, and testability.
