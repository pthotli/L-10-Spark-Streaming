
import os
from pyspark.sql import SparkSession
from pyspark.sql.functions import from_json, col, window, hour, minute, avg
from pyspark.sql.types import StructType, StructField, StringType, IntegerType, DoubleType, TimestampType
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.regression import LinearRegression, LinearRegressionModel

# Create Spark Session
spark = SparkSession.builder.appName("Task5_FareTrendPrediction").getOrCreate()
spark.sparkContext.setLogLevel("WARN")

# Define paths
MODEL_PATH = "models/fare_trend_model_v2"
TRAINING_DATA_PATH = "training-dataset.csv"

# --- PART 1: OFFLINE MODEL TRAINING ---
if not os.path.exists(MODEL_PATH):
    print(f"\n[Training Phase] Training new trend model using {TRAINING_DATA_PATH}...")
    
    # Load and cast data
    train_df = spark.read.csv(TRAINING_DATA_PATH, header=True, inferSchema=True)
    train_df = train_df.withColumn("timestamp", col("timestamp").cast(TimestampType())) \
                       .withColumn("fare_amount", col("fare_amount").cast(DoubleType()))

    # 1. Aggregate into 5-minute windows [cite: 45]
    windowed_df = train_df.groupBy(window(col("timestamp"), "5 minutes")).agg(avg("fare_amount").alias("avg_fare"))

    # 2. Feature Engineering: Extract hour and minute from window start [cite: 46]
    feature_df = windowed_df.withColumn("hour_of_day", hour(col("window.start"))) \
                            .withColumn("minute_of_hour", minute(col("window.start")))

    # 3. Assemble features and train model [cite: 47]
    assembler = VectorAssembler(inputCols=["hour_of_day", "minute_of_hour"], outputCol="features")
    final_train_data = assembler.transform(feature_df)
    
    lr = LinearRegression(featuresCol="features", labelCol="avg_fare")
    model = lr.fit(final_train_data)
    model.write().overwrite().save(MODEL_PATH)
    print(f"[Training Complete] Trend model saved to -> {MODEL_PATH}")

# --- PART 2: STREAMING INFERENCE ---
print("\n[Inference Phase] Starting real-time trend prediction...")

schema = StructType([
    StructField("trip_id", StringType()),
    StructField("driver_id", IntegerType()),
    StructField("distance_km", DoubleType()),
    StructField("fare_amount", DoubleType()),
    StructField("timestamp", TimestampType()) # Ensure timestamp type for windowing
])

# Read streaming data from socket [cite: 19, 24]
raw_stream = spark.readStream.format("socket").option("host", "localhost").option("port", 9999).load()
parsed_stream = raw_stream.select(from_json(col("value"), schema).alias("data")).select("data.*")

# 1. Apply same 5-minute windowed aggregation to stream [cite: 48]
streaming_windowed = parsed_stream.groupBy(window(col("timestamp"), "5 minutes")) \
                                  .agg(avg("fare_amount").alias("avg_fare"))

# 2. Apply same feature engineering [cite: 48]
streaming_features = streaming_windowed.withColumn("hour_of_day", hour(col("window.start"))) \
                                       .withColumn("minute_of_hour", minute(col("window.start")))

# 3. Load model and predict [cite: 49]
model = LinearRegressionModel.load(MODEL_PATH)
assembler_inf = VectorAssembler(inputCols=["hour_of_day", "minute_of_hour"], outputCol="features")
inf_features = assembler_inf.transform(streaming_features)
predictions = model.transform(inf_features)

# Select final output columns [cite: 50]
output_df = predictions.select(
    col("window.start").alias("window_start"),
    col("window.end").alias("window_end"),
    "avg_fare",
    col("prediction").alias("predicted_next_avg_fare")
)

# Start the stream [cite: 50]
query = output_df.writeStream.format("console").outputMode("complete").start()
query.awaitTermination()