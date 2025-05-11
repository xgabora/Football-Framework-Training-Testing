For GPU support with TensorFlow, ensure proper CUDA/cuDNN setup and the corresponding TensorFlow build.

### 1.4. Model Files

The evaluation pipeline relies on several pre-trained model files:

**Pre-Match Models:**
*   `models/pre_match/full_features.keras`: A Keras neural network model trained with a comprehensive set of pre-match features.
*   `models/pre_match/reduced_features.keras`: A Keras neural network model trained with a more constrained set of pre-match features.

**In-Play Model Components (located in `models/in_play/`):**
*   `scaler.pkl`: Scikit-learn scaler for input features.
*   `feature_names.pkl`: List of feature names expected by the in-play base models.
*   `target_transformer.pkl` (Optional): Transformer for goal predictions (e.g., `PowerTransformer`).
*   `label_encoder.pkl`: Scikit-learn `LabelEncoder` for Home/Draw/Away (H/D/A) outcomes.
*   `draw_calibrator.pkl`: Calibrator for draw probabilities (e.g., `IsotonicRegression` or `CalibratedClassifierCV`).
*   `meta_model.pkl`: The final meta-model (e.g., XGBoost, LightGBM) that combines base model outputs.
*   `cluster_medians.pkl` (Optional): Medians for cluster-based features if used.
*   `meta_feature_names.pkl`: List of feature names expected by the meta-model.
*   A suite of base model `.pkl` files (e.g., `base_nn_home.pkl`, `base_xgb_away.pkl`, `base_draw_specialist.pkl`, `base_hda_classifier.pkl`).

All specified model files must be present in their respective directories relative to where the script is run, or correct absolute paths must be provided.

## 2. Data Acquisition and Preparation

The pipeline uses a single primary data source for evaluation.

*   **Source:** `training/football_data.csv`. This CSV file contains historical match data, including results, scores, statistics, odds, Elo ratings, form metrics, and potentially cluster features. This is assumed to be the fully curated dataset output by a preceding data processing pipeline (like the one described previously).
*   **Format:** CSV.

## 3. Script Descriptions and Workflow (`evaluation.ipynb`)

The `evaluation.ipynb` script performs the following sequential operations, broadly divided into pre-match and in-play model evaluation sections.

### 3.1. Global Setup and Configuration
*   **Imports:** Loads necessary libraries (Pandas, NumPy, Scikit-learn, TensorFlow, XGBoost, LightGBM, SciPy).
*   **Path Definitions:** Specifies paths to the `DATA_FILE`, pre-match Keras models (`MODEL_FULL_PATH`, `MODEL_REDUCED_PATH`), and the in-play model directory (`FINAL_MODEL_DIR`).
*   **Constants & Mappings:** Defines `TARGET_COL` (e.g., 'FTResult'), `TARGET_MAP` for H/D/A encoding, `CLUSTER_COLS` for in-play model, and evaluation subset parameters (`EVALUATION_SUBSET_SIZE`, `EVALUATION_SUBSET_NAME`).
*   **Feature Set Definitions (Pre-Match):** Lists of raw features to load, and specific numerical and categorical feature sets for the "full" and "reduced" pre-match Keras models.
*   **Seed Initialization:** Sets random seeds for NumPy and TensorFlow for reproducibility.
*   **Warning Filters & Pandas Options:** Configures warning suppression and Pandas display options for cleaner output.

### 3.2. Pre-Match Model Evaluation Workflow

This section focuses on evaluating the Keras-based pre-match prediction models.

#### 3.2.1. Data Loading and Cleaning (`load_and_clean_data`)
*   **Input:** `DATA_FILE`, list of `FEATURES_TO_LOAD_FROM_CSV`, `TARGET_COL`, `ODDS_COLS`.
*   **Purpose:** Loads the specified subset of columns from the main data file.
*   **Methods:**
    *   Reads the CSV.
    *   Converts `MatchDate` to datetime objects, sorts data by date, and resets the index.
    *   Filters for valid target outcomes (H, D, A).
    *   Cleans odds columns: converts to numeric, replaces non-positive odds with a large value (1e6), and drops rows with any remaining NaN odds.
*   **Output:** A Pandas DataFrame (`df_full_dataset`) containing the cleaned raw data for pre-match model evaluation.

#### 3.2.2. Feature Engineering (`engineer_features`)
*   **Input:** The cleaned DataFrame from the previous step.
*   **Purpose:** Creates additional features required by the Keras models.
*   **Methods:**
    *   Extracts `Year`, `Month`, and `DayOfWeek` from `MatchDateTime`.
    *   Handles `HomeElo`, `AwayElo`, `Form3Home/Away`, `Form5Home/Away`: converts to numeric, imputes NaNs with medians (or predefined defaults if the column was entirely missing), and then calculates differential features (`EloDiff`, `Form3Diff`, `Form5Diff`).
*   **Output:** An engineered DataFrame (`df_engineered_full`).

#### 3.2.3. Preprocessor Fitting (`fit_preprocessors`)
*   **Input:** The fully engineered DataFrame (`df_engineered_full`), lists of numerical and categorical features for both full and reduced models.
*   **Purpose:** To fit all necessary preprocessors (imputers, scalers, encoders) on the *entire available dataset* to ensure transformations on the evaluation set are consistent with how they would be in a deployment scenario (where preprocessors are fit on training data).
*   **Methods:**
    *   Identifies available features from the input DataFrame corresponding to the defined feature sets.
    *   Fits `SimpleImputer` (median for numerical, most_frequent for categorical) for each relevant feature set.
    *   Applies imputation to a copy of the data.
    *   Fits `StandardScaler` on the imputed numerical features for both full and reduced models.
    *   Fits `OneHotEncoder` on imputed low-cardinality categorical features.
    *   Fits `LabelEncoder` for high-cardinality categorical features ('HomeTeam', 'AwayTeam', 'Division'), adding an `<UNK>` token to handle unseen categories during transformation. Stores vocabulary sizes.
*   **Output:** A dictionary (`preprocessors`) containing all fitted preprocessor objects and lists of columns they were fitted on.

#### 3.2.4. Evaluation Subset Preparation
*   Takes the latest `EVALUATION_SUBSET_SIZE` matches from `df_engineered_full`.
*   Applies an additional `dropna` based on `FULL_MODEL_RAW_DATA_REQUIREMENTS` to ensure the full Keras model has its minimally required raw inputs for a fair comparison.
*   Extracts the true target values (`y_true_eval`) and maps them to numerical representations using `TARGET_MAP`.

#### 3.2.5. Keras Model Data Transformation (`transform_data_for_keras`)
*   **Input:** A slice of data (e.g., `df_evaluation_subset`), the `preprocessors` dictionary, feature lists specific to the model being evaluated, a flag for model type ('full' or 'reduced'), and a reference to the loaded Keras model (for embedding layer dimensions).
*   **Purpose:** Transforms the input data slice into the multi-input format expected by the Keras models.
*   **Methods:**
    *   Applies the appropriate fitted `SimpleImputer` and `StandardScaler` to numerical features.
    *   Applies fitted `SimpleImputer` and `OneHotEncoder` to low-cardinality categorical features.
    *   Concatenates processed numerical and OHE features to form the `X_main_input`.
    *   For high-cardinality features ('HomeTeam', 'AwayTeam', 'Division'):
        *   Applies the fitted `SimpleImputer`.
        *   Transforms string categories to integer indices using the fitted `LabelEncoder`s, handling unknown values by mapping them to the `<UNK>` token's index.
        *   Caps the encoded indices to be within the vocabulary size of the corresponding embedding layer in the Keras model.
*   **Output:** A list of NumPy arrays, where each array is an input to the Keras model (main input, team embeddings, division embedding).

#### 3.2.6. Model Evaluation Function (`evaluate_model`)
*   **Input:** Model name, path to the `.keras` model file, evaluation DataFrame, true labels, preprocessors dictionary, feature lists, and model type flag.
*   **Purpose:** Loads a Keras model, prepares its input data, makes predictions, and calculates performance metrics.
*   **Methods:**
    *   Loads the Keras model using `tf.keras.models.load_model`.
    *   Calls `transform_data_for_keras` to prepare the input for the loaded model.
    *   Generates predictions (`pred_proba`) and derives class labels (`preds`).
    *   Calculates `accuracy_score` and macro-averaged F1-score using `precision_recall_fscore_support`.
    *   Includes error handling for model loading and prediction steps.
    *   Manages memory by deleting model and input variables and clearing the Keras backend session.
*   **Output:** A dictionary containing 'Accuracy', 'F1 (Macro)', and 'Evaluated Samples'.

#### 3.2.7. Running Pre-Match Evaluations
*   Calls `evaluate_model` for both `MODEL_FULL_PATH` and `MODEL_REDUCED_PATH` using their respective feature configurations.
*   Stores results in `results_summary`.

### 3.3. In-Play Model Evaluation Workflow

This section evaluates the stacked in-play prediction model.

#### 3.3.1. Component Loading (`load_pipeline_prerequisites_from_inplay_dir`)
*   **Input:** Path to the in-play model directory (`FINAL_MODEL_DIR`).
*   **Purpose:** Loads all pre-trained components of the in-play model pipeline.
*   **Methods:**
    *   Iterates through a predefined map of component keys to `.pkl` filenames.
    *   Loads each component using `pickle.load()`.
    *   Handles `FileNotFoundError` for optional components (e.g., `cluster_medians.pkl`) by setting them to `None`.
    *   Returns `None` if essential components are missing.
*   **Output:** A dictionary (`pipeline_components`) containing all loaded in-play model objects.

#### 3.3.2. Data Loading and Feature Engineering (`load_and_engineer_data_like_training`)
*   **Input:** `DATA_FILE` path and `CLUSTER_COLS` list.
*   **Purpose:** To load the entire dataset and apply the same extensive feature engineering pipeline that was used during the training of the in-play model's base and meta-learners. This ensures consistency between training and evaluation.
*   **Methods:**
    *   Reads the CSV data from `DATA_FILE`.
    *   Performs initial cleaning: drops rows with missing essential columns (`FTHome`, `FTAway`, odds, Elo), filters for valid odds, and drops NaNs for specific form/stats columns if present.
    *   Applies a comprehensive feature engineering process:
        *   Date/Time feature creation (`MatchDateTime`, `HTTotalGoals`).
        *   Combined odds (1X, X2, 12).
        *   Elo-based features (Difference, Total, Advantage).
        *   Form metrics and derived form dynamics (Momentum, Older).
        *   Implied probabilities from odds and bookmaker margin.
        *   Detailed statistical features (Shots, Target, Corners - totals, differences, accuracies, dominance indices).
        *   Discipline features (Card points, Foul differences).
        *   Advanced interaction features (Scoring/Defensive Efficiencies/Ratings, Draw Likelihood, Form Efficiencies, Clean Sheet Probabilities, Low Score Indicators, etc.).
    *   Sorts data by `MatchDateTime`.
*   **Output:** A fully engineered DataFrame (`df_full_engineered`).

#### 3.3.3. Evaluation Subset Preparation & Scaling (In-Play)
*   Aligns `df_full_engineered` with features specified in the loaded `feature_names.pkl` (from in-play components), adding missing columns with 0s.
*   Selects the features required for scaling (`X_to_scale`).
*   Imputes any remaining NaNs in `X_to_scale` (e.g., with median or 0).
*   Transforms `X_to_scale` using the loaded in-play `scaler.pkl`.
*   Selects the latest `EVALUATION_SUBSET_SIZE` matches from the scaled data (`X_full_scaled_df`) and the original engineered data (`df_full_engineered`) for evaluation.
*   Extracts true scores (`y_true_scores_eval`) and H/D/A outcomes (`y_true_hda_eval_text`, `y_true_hda_eval_numeric`) for the evaluation subset, using the loaded in-play `label_encoder.pkl`.

#### 3.3.4. Full Pipeline Prediction (`predict_with_full_pipeline`)
*   **Input:** Scaled evaluation features (`X_scaled_subset`), corresponding original data slice (`df_original_subset_for_clusters`), and the `pipeline_components` dictionary.
*   **Purpose:** Generates predictions using the entire in-play stacked model pipeline.
*   **Methods:**
    1.  **Base Model Predictions:** Makes predictions using all loaded base models (NNs, XGBoost, LightGBM for goals; draw specialist and H/D/A classifier for probabilities) on the scaled input features.
    2.  **NN Goal Transformation:** If `target_transformer_nn.pkl` was loaded, inverse-transforms the NN goal predictions.
    3.  **Draw Calibration:** Calibrates raw draw probabilities using `iso_reg_calibrator.pkl`.
    4.  **Meta-Feature Construction:**
        *   Creates a DataFrame of meta-features using outputs from base models (calibrated draw prob, goal predictions, score differences, H/D/A classifier outputs like predicted class, probability margin, entropy).
        *   If `CLUSTER_COLS` are used and present in `df_original_subset_for_clusters`, these are added (imputed with `cluster_medians.pkl` or 0 if needed).
        *   Aligns these meta-features with `meta_feature_names.pkl`, filling missing ones with 0.
    5.  **Meta-Model Prediction:** Feeds the aligned meta-features into the `meta_model_final.pkl` to get final H/D/A outcome predictions and probabilities.
*   **Output:** A dictionary containing final H/D/A predictions/probabilities and intermediate base model goal predictions.

#### 3.3.5. Metrics Calculation (`calculate_and_format_metrics_ots_style`)
*   **Input:** True scores, true H/D/A text labels, the `predictions` dictionary from the previous step, and the `label_encoder` used for evaluation.
*   **Purpose:** Calculates a comprehensive set of metrics for the in-play model.
*   **Methods:**
    *   Calculates `Outcome Accuracy` (H/D/A).
    *   Averages base model goal predictions to get overall predicted home/away scores.
    *   Calculates `Exact Score Acc` and `Exact Goal Diff` accuracy based on rounded average scores.
    *   Calculates R-squared (`R2_Home`, `R2_Away`), Mean Squared Error (`MSE_Home`, `MSE_Away`), and Mean Absolute Error (`MAE_Home`, `MAE_Away`) for goal predictions.
    *   Calculates multi-class `Brier Score` and weighted `ROC AUC (OvR W)` for H/D/A probabilities, ensuring `y_true` is binarized correctly using all known classes from the label encoder. Includes error handling for cases with insufficient classes for ROC AUC.
*   **Output:** A dictionary (`metrics_summary`) containing all calculated metrics.

#### 3.3.6. Running In-Play Evaluation
*   Calls `predict_with_full_pipeline` on the prepared evaluation data.
*   Calls `calculate_and_format_metrics_ots_style` to get the performance metrics.
*   Stores the results in `evaluation_results_dict`.

### 3.4. Final Summary Output
*   Consolidates evaluation results from both pre-match Keras models and the in-play model into Pandas DataFrames.
*   Prints a formatted summary table to the console, showing key metrics (Accuracy, F1, R2, MSE, MAE, ROC AUC, Brier Score) for each evaluated model. The pre-match summary is typically printed in Markdown format for readability.

## 4. Outputs and Considerations

### 4.1. Outputs
*   **Console Output:**
    *   Progress messages during data preparation and model loading.
    *   A summary table for pre-match Keras models showing 'Evaluated Samples', 'Accuracy', and 'F1 (Macro)'.
    *   A detailed metrics breakdown for the in-play model, including outcome accuracy, score prediction accuracies, goal regression metrics (R2, MSE, MAE), ROC AUC, and Brier score.

### 4.2. Key Considerations

*   **Data Consistency (`football_data.csv`):** The evaluation relies on `football_data.csv` being up-to-date and consistent with the data used for training all models. Feature definitions and availability must match.
*   **Model and Component Paths:** Correct paths to all `.keras` models and `.pkl` components are essential. Errors in paths will lead to `FileNotFoundError`.
*   **Feature Engineering Replication:** The feature engineering steps in `engineer_features` (for pre-match) and `load_and_engineer_data_like_training` (for in-play) must precisely mirror the feature engineering used during the respective model training phases. Discrepancies will invalidate the evaluation.
*   **Preprocessor State (Pre-Match):** Pre-match preprocessors (`SimpleImputer`, `StandardScaler`, `OneHotEncoder`, `LabelEncoder`) are fitted on the entire available dataset before being applied to the evaluation subset. This simulates a scenario where preprocessors are fitted on training data.
*   **Evaluation Subset Selection:** The evaluation is performed on the most recent `EVALUATION_SUBSET_SIZE` matches. The logic for selecting this subset (especially the `dropna` for pre-match Keras models) can affect the number of samples and comparability.
*   **Metrics Interpretation:**
    *   Pre-match models are primarily evaluated on classification metrics (Accuracy, F1).
    *   In-play models are evaluated on a broader range, including classification, goal regression, and probabilistic metrics, reflecting their more complex output.
*   **Memory Management:** The script includes `gc.collect()` calls and `tf.keras.backend.clear_session()` to help manage memory, especially when loading and processing multiple large models and datasets, though Python's garbage collection is largely automatic.
*   **Computational Cost:** Evaluating deep learning models (Keras) and complex stacked ensembles can be computationally intensive, especially the feature engineering and prediction steps on larger datasets or evaluation subsets.
