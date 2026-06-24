# ==========================================================
# GST ANOMALY DETECTION : JOB 05
# SILVER TO GOLD
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
# STEP 2 : INITIALIZE GLUE JOB
# ==========================================================

args = getResolvedOptions(
    sys.argv,
    [
        "JOB_NAME",
        "SILVER_BUCKET",
        "GOLD_BUCKET"
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

silver_bucket = args["SILVER_BUCKET"]
gold_bucket = args["GOLD_BUCKET"]

silver_base = f"s3://{silver_bucket}/silver/"
gold_base = f"s3://{gold_bucket}/gold/"




# ==========================================================
# STEP 4 : LOAD SILVER DATA
# ==========================================================

silver_df = spark.read.parquet(
    silver_base +
    "silver_purchase_orders/"
)



# ==========================================================
# STEP 5 : CREATE GOLD ANOMALY FLAGS
# ==========================================================

gold_anomaly_flags = (
    silver_df
    .select(
        "po_id",
        "vendor_id",
        "po_date",
        "invoice_number",
        "invoice_date",

        "is_gstin_valid",
        "blacklisted",
        "rate_mismatch",
        "value_spike",
        "name_mismatch",
        "duplicate_invoice",
        "director_overlap_flag",
        "vendor_non_filer",
        "missing_ewb",
        "state_mismatch",
        "late_invoice",
        "billing_state",

        "rule_fail_count",
        "rule_score",

        "is_anomalous",
        "severity",
        "anomaly_code"
    )
    .withColumn(
        "gold_created_ts",
        F.current_timestamp()
    )
)












# ==========================================================
# STEP 6 : CREATE GOLD VENDOR RISK
# ==========================================================

gold_vendor_risk = (
    silver_df
    .groupBy(
        "vendor_id",
        "trade_name",
        "gstin"
    )
    .agg(
        F.sum(
            "total_amount"
        ).alias(
            "total_po_value"
        ),

        F.count("*").alias(
            "total_po_count"
        ),

        F.sum(
            "is_anomalous"
        ).alias(
            "anomaly_count"
        ),

        (
            F.sum("is_anomalous")
            /
            F.count("*")
        ).alias(
            "anomaly_rate"
        ),

        F.avg(
            "name_sim_score"
        ).alias(
            "avg_name_sim_score"
        ),

        F.avg(
            "filing_compliance_rate"
        ).alias(
            "filing_compliance_rate"
        )
    )
)






 ==========================================================
# STEP 7 : VENDOR RISK TIER
# ==========================================================

gold_vendor_risk = (
    gold_vendor_risk
    .withColumn(
        "risk_tier",
        F.when(
            F.col("anomaly_rate") >= 0.20,
            "Critical"
        )
        .when(
            F.col("anomaly_rate") >= 0.10,
            "High"
        )
        .when(
            F.col("anomaly_rate") >= 0.05,
            "Medium"
        )
        .otherwise(
            "Low"
        )
    )
)







# ==========================================================
# STEP 8 : DASHBOARD SEVERITY
# ==========================================================

gold_anomaly_flags = (
    gold_anomaly_flags
    .withColumn(
        "severity_dashboard",
        F.when(
            F.col("rule_fail_count") >= 6,
            "Critical"
        )
        .when(
            F.col("rule_fail_count") >= 4,
            "High"
        )
        .when(
            F.col("rule_fail_count") >= 2,
            "Medium"
        )
        .otherwise(
            "Low"
        )
    )
)




# ==========================================================
# STEP 9 : LOAD MODEL PREDICTIONS
# ==========================================================
# Assumes ML job already created
# rf_predictions and xgb_predictions datasets.
# ==========================================================

rf_predictions = spark.read.parquet(
    silver_base +
    "rf_predictions/"
)

xgb_predictions = spark.read.parquet(
    silver_base +
    "xgb_predictions/"
)



# ==========================================================
# STEP 10 : JOIN MODEL PREDICTIONS
# ==========================================================

gold_anomaly_flags = (
    gold_anomaly_flags
    .join(
        rf_predictions.select(
            "po_id",
            "rf_prediction",
            "rf_probability"
        ),
        "po_id",
        "left"
    )
    .join(
        xgb_predictions.select(
            "po_id",
            "xgb_prediction",
            "xgb_probability"
        ),
        "po_id",
        "left"
    )
)





# ==========================================================
# STEP 11 : SAVE GOLD ANOMALY FLAGS
# ==========================================================

gold_anomaly_flags.write \
    .mode("overwrite") \
    .parquet(
        gold_base +
        "gold_anomaly_flags/"
    )



# ==========================================================
# STEP 12 : SAVE GOLD VENDOR RISK
# ==========================================================

gold_vendor_risk.write \
    .mode("overwrite") \
    .parquet(
        gold_base +
        "gold_vendor_risk/"
    )



# ==========================================================
# STEP 13 : COMMIT JOB
# ==========================================================

job.commit()



