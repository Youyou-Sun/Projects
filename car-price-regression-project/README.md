# Vehicle MSRP Prediction with Linear and Ridge Regression

This project builds an interpretable regression model for estimating a vehicle's manufacturer suggested retail price (MSRP) from technical specifications and market attributes.

## Project overview

The analysis follows a complete train–validate–test workflow:

- clean and audit 11,914 vehicle records;
- explore the skewed MSRP distribution and differences across manufacturers;
- engineer vehicle age, combined fuel economy, MPG spread, and market-segment count;
- preprocess numeric and categorical data through a reusable scikit-learn pipeline;
- compare a median baseline, ordinary linear regression, and Ridge models;
- select the regularization penalty on validation data;
- evaluate the final model once on an untouched test set;
- inspect influential coefficients and the largest prediction errors.

## Why this version is different

This is an independent learning project inspired by Module 2 of [DataTalksClub's Machine Learning Zoomcamp](https://github.com/DataTalksClub/machine-learning-zoomcamp/tree/main/02-regression). The project uses the same public learning dataset, but the narrative and code were written from scratch and reorganized around a scikit-learn pipeline, original feature engineering, model comparison, and error analysis. It is not a copy of the course notebook.

## Results

<!-- RESULTS_START -->
The median baseline produced a validation RMSE of **1.1367** on the log scale. Ordinary linear regression reduced it to **0.3326**, while Ridge with `alpha=1` was effectively tied at **0.3326**. Since regularization did not meaningfully improve validation performance, the ordinary linear model was selected.

On the untouched test set, the final model achieved:

| Metric | Test result |
| --- | ---: |
| RMSE on `log(1 + MSRP)` | 0.3098 |
| MAE in original dollars | $7,664 |
| R² on log scale | 0.919 |
<!-- RESULTS_END -->

Because MSRP has a long right tail, the model is trained and primarily evaluated on `log(1 + MSRP)`. Dollar MAE is included to keep the results easy to interpret.

## Repository structure

```text
.
├── car_price_regression.ipynb  # complete analysis and model development
├── README.md                   # project summary
├── requirements.txt            # Python dependencies
└── .gitignore                  # excludes downloaded data and local files
```

## How to run

1. Clone this repository and move into the project folder.
2. Create and activate a virtual environment.
3. Install the dependencies:

   ```bash
   pip install -r requirements.txt
   ```

4. Open `car_price_regression.ipynb` in Jupyter Notebook, JupyterLab, or VS Code and run all cells.

The notebook downloads the CSV from the source used by the course if `data/data.csv` is not already present.

## Data and attribution

- Dataset: [Car Features and MSRP](https://www.kaggle.com/datasets/CooperUnion/cardataset), originally published on Kaggle by Cooper Union.
- Learning reference: [Machine Learning Zoomcamp — Module 2: Regression](https://github.com/DataTalksClub/machine-learning-zoomcamp/tree/main/02-regression), created by DataTalksClub.

## Limitations

MSRP is not the same as a real transaction price. The dataset also lacks mileage, condition, location, and date-of-sale information, so this model is intended as a portfolio learning project rather than a production pricing tool.
