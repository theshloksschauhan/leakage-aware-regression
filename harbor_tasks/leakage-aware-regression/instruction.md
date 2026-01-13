You need to create and analyze a synthetic time-series dataset, then train and evaluate a regression model while detecting any data leakage. Follow these steps:

Generate the dataset: Use Python (pandas/numpy) to create a dataset with 100 samples and four columns: x1, x2, leak_feature, and target.

Let target = 2*x1 + 0.5*x2 + noise (where noise is small random Gaussian noise).

Define leak_feature as a nearly perfect predictor of target, e.g. leak_feature = target * 0.75 + small_noise.

Save this dataset to /app/data.csv (including a header row).

Detect data leakage: Load /app/data.csv and compute the Pearson correlation between each feature (x1, x2, leak_feature) and the target.

Any feature with correlation > 0.99 with target should be treated as leaking information.

Write the name(s) of these leaking feature(s) to /app/removed_features.txt, one per line (for example, the file should contain leak_feature).

Prepare training data: Remove the detected leaking feature(s) from the dataset. Use only the remaining feature columns (e.g. x1 and x2) to predict target.

Train a regression model: Using scikit-learn, train a LinearRegression model on the remaining features to predict the target. (For example, use LinearRegression().fit(X, y) with X = [remaining features] and y = target.)

Make predictions: Use your trained model to predict target for all rows. Save the results to /app/predictions.csv with two columns: target (the true value) and prediction (the model's predicted value), including a header row.

Evaluate the model: Calculate the Mean Squared Error (MSE) between the true target and your prediction. Write this numeric MSE value to /app/mse.txt.

Output files:

/app/removed_features.txt: Contains the name(s) of removed feature(s), one per line (e.g. leak_feature).

/app/predictions.csv: Two-column CSV (target,prediction) with header.

/app/mse.txt: A single number (the computed MSE).

Use absolute file paths (e.g. /app/...) and the provided libraries (numpy, pandas, scikit-learn) for data processing and modeling. Follow these instructions exactly to ensure correct output formats and locations.
