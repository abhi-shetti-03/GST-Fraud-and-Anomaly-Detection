# ==========================================================
# GST ANOMALY DETECTION : JOB 03
# RAW TO BRONZE
# ==========================================================




# ==========================================================
# STEP 1 : IMPORT REQUIRED LIBRARIES
# ==========================================================

import sys

from awsglue.utils import getResolvedOptions
from awsglue.context import GlueContext
from awsglue.job import Job

from pyspark.context import SparkContext
from pyspark.sql import functions as F





# ==========================================================
# STEP 2 : INITIALIZE AWS GLUE JOB
# ==========================================================

args = getResolvedOptions(
    sys.argv,
    [
        "JOB_NAME",
        "RAW_BUCKET",
        "BRONZE_BUCKET"
    ]
)

sc = SparkContext()

glueContext = GlueContext(sc)

spark = glueContext.spark_session

job = Job(glueContext)

job.init(
    args["JOB_NAME"],
    args
)




# ==========================================================
# STEP 3 : DEFINE INPUT / OUTPUT PATHS
# ==========================================================

raw_bucket = args["RAW_BUCKET"]
bronze_bucket = args["BRONZE_BUCKET"]

raw_base = f"s3://{raw_bucket}/raw/enrich_raw/"
bronze_base = f"s3://{bronze_bucket}/bronze/"





# ==========================================================
# STEP 4 : LOAD RAW DATASETS
# ==========================================================

vendors_raw = spark.read.parquet(
    raw_base + "vendors/"
)

purchase_orders_raw = spark.read.parquet(
    raw_base + "purchase_orders/"
)

vendor_invoices_raw = spark.read.parquet(
    raw_base + "vendor_invoices/"
)

hsn_raw = spark.read.parquet(
    raw_base + "hsn_rate_schedule/"
)

blacklist_raw = spark.read.parquet(
    raw_base + "cbic_blacklist/"
)

ground_truth_raw = spark.read.parquet(
    raw_base + "ground_truth/"
)

historical_raw = spark.read.parquet(
    raw_base + "historical_po_values/"
)





# ==========================================================
# STEP 5 : PURCHASE ORDER BRONZE TRANSFORMATION
# ==========================================================
# Convert PO date into proper date format.
# Create partition columns.
# Add ingestion timestamp.
# Apply schema enforcement.

bronze_purchase_orders = (
    purchase_orders_raw
    .withColumn(
        "po_date",
        F.to_date(
            F.col("po_date"),
            "dd/MM/yyyy"
        )
    )
    .withColumn(
        "year",
        F.year("po_date")
    )
    .withColumn(
        "month",
        F.month("po_date")
    )
    .withColumn(
        "ingestion_timestamp",
        F.current_timestamp()
    )
    .filter(
        F.col("po_id").isNotNull()
    )
    .filter(
        F.col("vendor_id").isNotNull()
    )
)





# ==========================================================
# STEP 6 : VENDOR BRONZE TRANSFORMATION
# ==========================================================
# Add ingestion timestamp.
# Fill null compliance flags.

bronze_vendors = (
    vendors_raw
    .withColumn(
        "ingestion_timestamp",
        F.current_timestamp()
    )
    .fillna(
        {
            "composition_flag": 0,
            "filing_history_q1": 0,
            "filing_history_q2": 0,
            "filing_history_q3": 0,
            "filing_history_q4": 0,
            "filing_history_q5": 0,
            "filing_history_q6": 0
        }
    )
)



# ==========================================================
# Step 7 : Save to Bronze
# ==========================================================




# BRONZE PURCHASE ORDERS

bronze_purchase_orders.write \
    .mode("overwrite") \
    .partitionBy("year", "month") \
    .parquet(
        bronze_base + "purchase_orders/"
    )




# BRONZE VENDORS

bronze_vendors.write \
    .mode("overwrite") \
    .parquet(
        bronze_base + "vendors/"
    )




# BRONZE INVOICES

vendor_invoices_raw.write \
    .mode("overwrite") \
    .parquet(
        bronze_base + "vendor_invoices/"
    )




# WRITE BRONZE HSN

hsn_raw.write \
    .mode("overwrite") \
    .parquet(
        bronze_base + "hsn_rate_schedule/"
    )



# WRITE BRONZE BLACKLIST

blacklist_raw.write \
    .mode("overwrite") \
    .parquet(
        bronze_base + "cbic_blacklist/"
    )



# WRITE BRONZE GROUND TRUTH

ground_truth_raw.write \
    .mode("overwrite") \
    .parquet(
        bronze_base + "ground_truth/"
    )



# WRITE BRONZE HISTORICAL DATA

historical_raw.write \
    .mode("overwrite") \
    .parquet(
        bronze_base + "historical_po_values/"
    )

# ==========================================================
# STEP 8 : COMMIT GLUE JOB
# ==========================================================

job.commit()