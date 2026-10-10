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


