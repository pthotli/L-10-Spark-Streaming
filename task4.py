import os
from pyspark.sql import SparkSession
from pyspark.sql.functions import from_json, col, abs as abs_diff
from pyspark.sql.types import StructType, StructField, StringType, IntegerType, DoubleType

# Import necessary MLlib classes [cite: 29, 30, 32]
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.regression import LinearRegression, LinearRegressionModel

# Create Spark Session
spark = SparkSession.builder.appName("Task4_FarePrediction_Assignment").getOrCreate()
spark.sparkContext.setLogLevel("WARN")

# Define paths for the model and training data [cite: 31]
MODEL_PATH = "models/fare_model"
TRAINING_DATA_PATH = "training-dataset.csv"

# --- PART 1: MODEL TRAINING (Offline) ---
if not os.path.exists(MODEL_PATH):
    print(f"\n[Training Phase] No model found. Training a new model using {TRAINING_DATA_PATH}...")

    # Load the training data from the provided CSV file [cite: 28]
    train_df_raw = spark.read.csv(TRAINING_DATA_PATH, header=True, inferSchema=False)

    # Cast distance_km and fare_amount columns to DoubleType for ML [cite: 23]
    train_df = train_df_raw.withColumn("distance_km", col("distance_km").cast(DoubleType())) \
                           .withColumn("fare_amount", col("fare_amount").cast(DoubleType()))

    # Create a VectorAssembler to combine feature columns [cite: 29]
    assembler = VectorAssembler(inputCols=["distance_km"], outputCol="features")
    train_data_with_features = assembler.transform(train_df)

    # Create and fit a LinearRegression model [cite: 30]
    lr = LinearRegression(featuresCol="features", labelCol="fare_amount")
    model = lr.fit(train_data_with_features)

    # Save the trained model to the specified MODEL_PATH [cite: 31]
    model.write().overwrite().save(MODEL_PATH)
    print(f"[Training Complete] Model saved to -> {MODEL_PATH}")
else:
    print(f"[Model Found] Using existing model from {MODEL_PATH}")


# --- PART 2: STREAMING INFERENCE ---
print("\n[Inference Phase] Starting real-time fare prediction stream...")

# Define the schema for the incoming streaming data [cite: 24]
schema = StructType([
    StructField("trip_id", StringType()),
    StructField("driver_id", IntegerType()),
    StructField("distance_km", DoubleType()),
    StructField("fare_amount", DoubleType()),
    StructField("timestamp", StringType())
])

# Read streaming data from the socket
raw_stream = spark.readStream.format("socket") \
    .option("host", "localhost") \
    .option("port", 9999) \
    .load()

# Parse the incoming JSON data from the stream
parsed_stream = raw_stream.select(from_json(col("value"), schema).alias("data")).select("data.*")

# Load the pre-trained LinearRegressionModel from MODEL_PATH [cite: 32]
model = LinearRegressionModel.load(MODEL_PATH)

# Use a VectorAssembler for the streaming data features [cite: 29]
assembler_inference = VectorAssembler(inputCols=["distance_km"], outputCol="features")
stream_with_features = assembler_inference.transform(parsed_stream)

# Use the loaded model to make predictions on the streaming data [cite: 33]
predictions = model.transform(stream_with_features)

# Calculate the 'deviation' between the actual 'fare_amount' and the 'prediction' [cite: 26, 34]
predictions_with_deviation = predictions.withColumn("deviation", abs_diff(col("fare_amount") - col("prediction")))

# Select the final columns to display in the output [cite: 35]
output_df = predictions_with_deviation.select(
    "trip_id", "driver_id", "distance_km", "fare_amount",
    col("prediction").alias("predicted_fare"), "deviation"
)

# Write the final results to the console [cite: 35]
query = output_df.writeStream \
    .format("console") \
    .outputMode("append") \
    .option("truncate", False) \
    .start()

# Wait for the streaming query to terminate
query.awaitTermination()