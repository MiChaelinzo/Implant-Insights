from pyspark.sql.functions import *
from pyspark.sql.types import *
from databricks import feature_store

# Enhanced schema with additional metadata fields 
schema = StructType([
    StructField("filename", StringType(), True),
    StructField("content", BinaryType(), True),
    StructField("metadata", MapType(StringType(), StringType()), True),
    StructField("ingestion_timestamp", TimestampType(), True)  # Capture ingestion time
    StructField("payload", StringType(), True),   # Raw data from implant
    StructField("ingestion_timestamp", TimestampType(), True)

])

# Auto Loader with error handling 
df = (spark.readStream.format("cloudFiles") 
      .option("cloudFiles.format", "binaryFile")
      .schema(schema)
      .load("/mnt/implant-data")
      .withColumn("ingestion_timestamp", current_timestamp()))  

# Extract metadata (assume more fields are present in your actual metadata)
df = (df
      .withColumn("file_type", get_file_type_udf(col("content"))) 
      .withColumn("patient_id", col("metadata")["patient_id"]) 
      .withColumn("implant_id", col("metadata")["implant_id"]) 
      .withColumn("data_category", col("metadata")["data_category"]) ) 
      .withColumn("patient_id", regexp_extract(col("payload"), r"ID:(\w+)", 1))
      .withColumn("heart_rate", regexp_extract(col("payload"), r"HR:(\d+)", 1))
      .withColumn("blood_pressure", regexp_extract(col("payload"), r"BP:(\d+/\d+)", 1))

# Write to Delta Lake with more granular partitioning 
(df.writeStream
  .format("delta")
  .option("checkpointLocation", "/mnt/implant-data/_checkpoint") 
  .partitionBy("file_type", "patient_id", "data_category")  # Partition by category for targeted queries
  .outputMode("append")
  .table("implant_data_bronze"))
