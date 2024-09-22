import mlflow
from pyspark.sql.functions import *

# Load the best model from MLflow (adjust model name and stage as needed)
model_uri = "models:/your_model_name/Production"
loaded_model = mlflow.pyfunc.spark_udf(spark, model_uri) 

# Load new implant data (assuming it's been ingested into the 'implant_data_bronze' table)
new_data_df = (spark.read.format("delta")
                .table("implant_data_bronze")
                .filter(col("ingestion_timestamp") > last_processed_timestamp)  # Filter for new data only
                .filter(col("data_category") == "medical_image"))  # Or your desired category

# Preprocess the new data (similar to how you did it in data_processing.py)
processed_new_data_df = new_data_df.withColumn("features", process_image_udf(col("content")))

# Perform inference using the loaded model
predictions_df = processed_new_data_df.withColumn("prediction", loaded_model(*feature_columns))

# Save or process the predictions further (e.g., write to a new Delta Lake table)
(predictions_df
  .write
  .format("delta")
  .mode("append")  # Append to existing predictions table
  .saveAsTable("implant_predictions"))

# Update the last processed timestamp (store this somewhere persistent)
last_processed_timestamp = new_data_df.agg(max("ingestion_timestamp")).collect()[0][0]