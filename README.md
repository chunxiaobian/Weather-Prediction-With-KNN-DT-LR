# Seattle Weather Classification

[简体中文](README_zh.md) | **English**

A machine learning coursework project built with Python and scikit-learn. It uses precipitation, maximum temperature, minimum temperature, and wind speed to classify weather into five categories: drizzle, fog, rain, snow, and sun. The Jupyter Notebook covers data loading, label encoding, dataset splitting, visualization, class balancing, hyperparameter tuning, evaluation, and sample predictions.

> This README describes the actual code and saved outputs in `Assignment 3 weather classifer.ipynb`. Reported results come from the original notebook and have not been independently reproduced. The introductory Iris dataset description and some comments are leftover template content and do not describe the weather classification task.

## Dataset

The input file is `seattle-weather.csv`. The dataset shown in the notebook contains **1,461 records and 6 columns**, covering January 1, 2012 through December 31, 2015.

| Column | Description | Role |
| --- | --- | --- |
| `date` | Observation date | Removed before modeling |
| `precipitation` | Precipitation | Input feature |
| `temp_max` | Maximum temperature | Input feature |
| `temp_min` | Minimum temperature | Input feature |
| `wind` | Wind speed | Input feature |
| `weather` | Weather category | Prediction target |

`LabelEncoder` encodes the target as `drizzle=0`, `fog=1`, `rain=2`, `snow=3`, and `sun=4`.

The original notebook does not provide a verifiable source or measurement units for the weather dataset. Supply the CSV before running the project. The Iris dataset link in the notebook is unrelated to this project.

## Workflow

1. Load the CSV, encode weather labels, and select four numerical features.
2. Randomly split the data with `random_state=42` into approximately 60% training, 20% validation, and 20% test data: 876, 292, and 293 records, respectively.
3. Reserve two test records for simulated new-sample predictions, leaving 291 records for final testing.
4. Create scatter plots for all six feature pairs in the training set.
5. Apply SMOTE oversampling to the training set only to address class imbalance.
6. Select one model, enumerate its parameter combinations, and choose parameters using validation accuracy.
7. Combine the oversampled training set with the original validation set and refit the final model. For logistic regression, KNN, and SVC, also refit the scaler.
8. Report test accuracy, macro precision, recall, F1, a confusion matrix, and a classification report, then predict the two reserved samples.

The two “future samples” come from the randomly split test set and only demonstrate inference. They do not represent forecasts for future dates. The project classifies weather using same-day observations; it does not implement time series forecasting.

## Available Models

| Selection | Model | Main tuned parameters | Standardization |
| --- | --- | --- | --- |
| 1 | Decision tree | Criterion, depth, minimum split size, minimum leaf size | No |
| 2 | Random forest | Number of trees, criterion, depth, split and leaf limits, feature count | No |
| 3 | Logistic regression | `C`, `solver` | Yes |
| 4 | K-nearest neighbors (KNN) | Neighbor count, weights, distance parameter `p` | Yes |
| 5 | Support vector classifier (SVC) | Kernel, `C`, `gamma` | Yes |

Each run selects one model. The notebook does not contain a complete comparison of all five models, so its saved results do not establish which model performs best overall.

## Saved Results

The saved run uses **KNN**, with the following parameters selected on the validation set:

```python
{"n_neighbors": 1, "weights": "uniform", "p": 1}
```

Here, `p=1` specifies Manhattan distance.

| Metric | Result |
| --- | ---: |
| Validation accuracy | 58.56% |
| Test accuracy | 62.89% |
| Test macro precision | 0.4177 |
| Test macro recall | 0.5113 |
| Test macro F1 | 0.4360 |
| Accuracy on two demonstration samples | 100% (2/2) |

### Test Classification Report

| Class | Precision | Recall | F1 | Support |
| --- | ---: | ---: | ---: | ---: |
| drizzle | 0.08 | 0.17 | 0.11 | 12 |
| fog | 0.19 | 0.28 | 0.23 | 25 |
| rain | 0.85 | 0.81 | 0.83 | 129 |
| snow | 0.27 | 0.75 | 0.40 | 4 |
| sun | 0.70 | 0.55 | 0.62 | 121 |

The model performs relatively well on rain, but poorly on drizzle and fog. Rain and sun dominate the test set, so overall accuracy does not fully describe minority-class performance. There are only four snow test samples, and the two demonstration samples are insufficient to establish reliable generalization.

## Running the Project

### Prepare the Files

Place the following files in the same directory. The README files do not include copies of the dataset or notebook.

```text
weather-classifier/
├── README.md
├── README.en.md
├── Assignment 3 weather classifer.ipynb
└── seattle-weather.csv
```

### Local Jupyter

Install the dependencies used by the project:

```bash
python -m pip install pandas numpy matplotlib scikit-learn imbalanced-learn jupyter
jupyter notebook
```

After opening the notebook:

1. Skip or remove the `google.colab` import and Google Drive mounting cell.
2. Skip the original `cd /content/drive/...` cell. Set the working directory to the CSV location, or update the path passed to `pd.read_csv()`.
3. Run the cells in order from the beginning.
4. Enter a number from `1` to `5` when prompted to select a model.
5. Review the validation parameters, test metrics, confusion matrix, and demonstration predictions.

The installation command does not pin dependency versions. The original notebook does not record a complete environment, and interfaces such as `multi_class="auto"` in the logistic regression branch may be incompatible with the installed version. If parameter errors occur, adapt the code to that version. Compatibility across different dependency versions has not been tested here.

### Google Colab

Upload the notebook, place the CSV in Google Drive, install any missing dependencies, and run the mounting cell. Replace the original `cd` path with the actual data directory, then run the cells in order and select a model.

## Known Limitations and Improvements

- **Documentation:** The title, Iris introduction, and some training/validation/test plot comments do not match the implementation. All six scatter plots actually use the training set.
- **Evaluation design:** The project uses a single random split without stratification or cross-validation. Add stratified cross-validation and use metrics such as macro F1 alongside accuracy for tuning.
- **Temporal generalization:** Forecasting future weather would require a revised prediction target, temporal features, and evaluation based on chronological splits.
- **Preprocessing order:** SMOTE currently operates on unscaled features before standardization for some models. A pipeline can perform scaling and resampling within each training fold, reducing the influence of feature scales on neighbor selection and preventing leakage during cross-validation.
- **Model comparison:** Record all five models under a consistent evaluation protocol before selecting a model. The current results do not demonstrate that KNN outperforms the alternatives.
- **Input handling:** An invalid model number only prints a message rather than prompting again, and non-integer input raises an error. Add input validation.
- **Data quality and reproducibility:** Add missing-value and duplicate checks, dataset provenance, dependency versions, and random-state records.
- **Deployment:** Model persistence, a prediction API, and application deployment are not implemented. Deployment should preserve the model, scaler, and label encoder together.

## Technology Stack

Python, Jupyter Notebook / Google Colab, pandas, NumPy, Matplotlib, scikit-learn, and imbalanced-learn (SMOTE).
