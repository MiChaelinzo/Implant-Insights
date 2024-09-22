import mlflow
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, classification_report
from databricks import feature_store

# Load features from the Feature Store 
fs = feature_store.FeatureStoreClient()
features_df = fs.read_table(name="implant_image_features")

# Prepare data for training (assuming 'label' is your target variable)
X = features_df.drop("label")
y = features_df.select("label")

# Split data into training and testing sets 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define a parameter grid for hyperparameter tuning (optional but recommended)
param_grid = { 
    'n_estimators': [100, 200, 300],
    'max_depth': [None, 5, 10],
    'min_samples_split': [2, 5, 10]
}

# Train a Random Forest model with hyperparameter tuning using GridSearchCV
with mlflow.start_run():
    rf = RandomForestClassifier()
    grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, cv=3, n_jobs=-1, verbose=2)
    grid_search.fit(X_train, y_train)

    # Get the best model from the grid search
    best_model = grid_search.best_estimator_

    # Make predictions on the test set
    y_pred = best_model.predict(X_test)

    # Evaluate the model
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)

    # Log metrics, parameters, and the model to MLflow
    mlflow.log_metric("accuracy", accuracy)
    mlflow.log_params(best_model.get_params())
    mlflow.sklearn.log_model(best_model, "random_forest_model")

    # Print the evaluation results
    print(f"Accuracy: {accuracy}")
    print("Classification Report:\n", report)