# ==========================================================
# GST ANOMALY DETECTION : JOB 04
# BRONZE TO SILVER
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
from pyspark.sql.window import Window

from itertools import chain

# Glue Job → Job Details → Advanced properties → Job parameters
# --additional-python-modules
# rapidfuzz==3.14.1

from rapidfuzz import fuzz
from pyspark.sql.types import DoubleType
from pyspark.sql.functions import udf




# ==========================================================
# STEP 2 : INITIALIZE AWS GLUE JOB
# ==========================================================

args = getResolvedOptions(
    sys.argv,
    [
        "JOB_NAME",
        "BRONZE_BUCKET",
        "SILVER_BUCKET"
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

bronze_bucket = args["BRONZE_BUCKET"]
silver_bucket = args["SILVER_BUCKET"]

bronze_base = f"s3://{bronze_bucket}/bronze/"
silver_base = f"s3://{silver_bucket}/silver/"





# ==========================================================
# STEP 4 : LOAD BRONZE DATASETS
# ==========================================================

purchase_orders = spark.read.parquet(
    bronze_base + "purchase_orders/"
)

vendors = spark.read.parquet(
    bronze_base + "vendors/"
)

vendor_invoices = spark.read.parquet(
    bronze_base + "vendor_invoices/"
)

hsn_rate_schedule = spark.read.parquet(
    bronze_base + "hsn_rate_schedule/"
)

cbic_blacklist = spark.read.parquet(
    bronze_base + "cbic_blacklist/"
)

historical_po_values = spark.read.parquet(
    bronze_base + "historical_po_values/"
)

ground_truth = spark.read.parquet(
    bronze_base + "ground_truth/"
)




# ==========================================================
# STEP 5 : ENFORCE PURCHASE ORDER DATATYPES
# ==========================================================
# Ensure numeric columns have proper datatypes.


silver_df = (
    purchase_orders
    .withColumn(
        "quantity",
        F.col("quantity").cast("int")
    )
    .withColumn(
        "base_amount",
        F.col("base_amount").cast("double")
    )
    .withColumn(
        "cgst_rate",
        F.col("cgst_rate").cast("double")
    )
    .withColumn(
        "sgst_rate",
        F.col("sgst_rate").cast("double")
    )
    .withColumn(
        "igst_rate",
        F.col("igst_rate").cast("double")
    )
    .withColumn(
        "cgst_amt",
        F.col("cgst_amt").cast("double")
    )
    .withColumn(
        "sgst_amt",
        F.col("sgst_amt").cast("double")
    )
    .withColumn(
        "igst_amt",
        F.col("igst_amt").cast("double")
    )
    .withColumn(
        "cess_amt",
        F.col("cess_amt").cast("double")
    )
    .withColumn(
        "total_amount",
        F.col("total_amount").cast("double")
    )
)





# ==========================================================
# STEP 6 : STANDARDIZE VENDOR DATASET
# ==========================================================
# Rename columns to avoid ambiguity after joins.

vendors = (
    vendors
    .withColumnRenamed(
        "status",
        "vendor_status"
    )
    .withColumnRenamed(
        "id",
        "vendor_row_id"
    )
)




# ==========================================================
# STEP 7 : JOIN VENDOR MASTER
# ==========================================================

silver_df = (
    silver_df
    .join(
        vendors,
        on="vendor_id",
        how="left"
    )
)




# ==========================================================
# STEP 8 : REMOVE DUPLICATE INVOICES
# ==========================================================
# One invoice record per PO.

vendor_invoices = (
    vendor_invoices
    .dropDuplicates(
        ["po_id"]
    )
)





# ==========================================================
# STEP 9 : JOIN INVOICE DATA
# ==========================================================

vendor_invoices = (
    vendor_invoices
    .drop(
        "ingestion_timestamp"
    )
)

silver_df = (
    silver_df
    .join(
        vendor_invoices,
        on="po_id",
        how="left"
    )
)





# ==========================================================
# STEP 10 : GSTIN FORMAT VALIDATION
# ==========================================================
# Validate GSTIN structure using regex.


GST_REGEX = (
    r'^[0-9]{2}'
    r'[A-Z]{5}'
    r'[0-9]{4}'
    r'[A-Z]'
    r'[A-Z0-9]'
    r'Z'
    r'[A-Z0-9]$'
)

silver_df = (
    silver_df
    .withColumn(
        "is_gstin_valid",
        F.col("buyer_gstin")
        .rlike(GST_REGEX)
    )
)




# ==========================================================
# STEP 11 : EXTRACT GST STATE CODE
# ==========================================================

silver_df = (
    silver_df
    .withColumn(
        "gst_state_code",
        F.substring(
            "buyer_gstin",
            1,
            2
        )
    )
)






# ==========================================================
# STEP 12 : GST STATE LOOKUP
# ==========================================================

state_map = {
    "01": "JAMMU AND KASHMIR",
    "02": "HIMACHAL PRADESH",
    "03": "PUNJAB",
    "04": "CHANDIGARH",
    "05": "UTTARAKHAND",
    "06": "HARYANA",
    "07": "DELHI",
    "08": "RAJASTHAN",
    "09": "UTTAR PRADESH",
    "10": "BIHAR",
    "11": "SIKKIM",
    "12": "ARUNACHAL PRADESH",
    "13": "NAGALAND",
    "14": "MANIPUR",
    "15": "MIZORAM",
    "16": "TRIPURA",
    "17": "MEGHALAYA",
    "18": "ASSAM",
    "19": "WEST BENGAL",
    "20": "JHARKHAND",
    "21": "ODISHA",
    "22": "CHHATTISGARH",
    "23": "MADHYA PRADESH",
    "24": "GUJARAT",
    "25": "DAMAN AND DIU",
    "26": "DADRA AND NAGAR HAVELI",
    "27": "MAHARASHTRA",
    "28": "ANDHRA PRADESH",
    "29": "KARNATAKA",
    "30": "GOA",
    "31": "LAKSHADWEEP ISLANDS",
    "32": "KERALA",
    "33": "TAMIL NADU",
    "34": "PONDICHERRY",
    "35": "ANDAMAN AND NICOBAR ISLANDS",
    "36": "TELANGANA",
    "37": "ANDHRA PRADESH (NEW)"
}

mapping_expr = F.create_map(
    [F.lit(x)
     for x in chain(*state_map.items())]
)

silver_df = (
    silver_df
    .withColumn(
        "gst_state_name",
        mapping_expr[
            F.col("gst_state_code")
        ]
    )
)




# ==========================================================
# STEP 13 : GST STATE MISMATCH FLAG
# ==========================================================
# Detect mismatch between GSTIN state
# and buyer state.
# ==========================================================

silver_df = (
    silver_df
    .withColumn(
        "state_mismatch",
        F.when(
            F.lower(
                F.col("buyer_state")
            )
            !=
            F.lower(
                F.col("gst_state_name")
            ),
            1
        ).otherwise(0)
    )
)




# ==========================================================
# STEP 14 : BLACKLIST JOIN
# ==========================================================
# Flag vendors present in CBIC blacklist.
# ==========================================================

cbic_blacklist = (
    cbic_blacklist
    .select("gstin")
    .dropDuplicates()
)

silver_df = (
    silver_df
    .join(
        cbic_blacklist.withColumn(
            "blacklisted",
            F.lit(1)
        ),
        on="gstin",
        how="left"
    )
)

silver_df = (
    silver_df
    .withColumn(
        "blacklisted",
        F.coalesce(
            F.col("blacklisted"),
            F.lit(0)
        )
    )
)




# ==========================================================
# STEP 14 : JOIN HSN RATE SCHEDULE
# ==========================================================
# Attach expected GST rate using HSN code.
# Used for TAX-001 HSN Rate Mismatch detection.

hsn_rate_schedule = (
    hsn_rate_schedule
    .withColumn(
        "gst_rate",
        F.col("gst_rate").cast("double")
    )
)

silver_df = (
    silver_df
    .join(
        hsn_rate_schedule.select(
            "hsn_code",
            "gst_rate"
        ),
        on="hsn_code",
        how="left"
    )
)



# ==========================================================
# STEP 15 : CALCULATE ACTUAL GST RATE
# ==========================================================
# Actual GST charged on invoice.

silver_df = (
    silver_df
    .withColumn(
        "actual_gst_rate",
        F.col("cgst_rate")
        + F.col("sgst_rate")
        + F.col("igst_rate")
    )
)




# ==========================================================
# STEP 16 : TAX-001 HSN RATE MISMATCH
# ==========================================================
# Compare expected GST rate vs actual GST rate.

silver_df = (
    silver_df
    .withColumn(
        "rate_mismatch",
        F.when(
            F.col("actual_gst_rate")
            != F.col("gst_rate"),
            1
        ).otherwise(0)
    )
)





# ==========================================================
# STEP 17 : CREATE YEAR_MONTH KEY
# ==========================================================
# Used to join historical PO statistics.
# ==========================================================

silver_df = (
    silver_df
    .withColumn(
        "year_month",
        F.date_format(
            F.col("po_date"),
            "yyyy-MM"
        )
    )
)





# ==========================================================
# STEP 18 : AGGREGATE HISTORICAL PO VALUES
# ==========================================================
# Create vendor-month level history.

historical_po_values = (
    historical_po_values
    .groupBy(
        "vendor_id",
        "year_month"
    )
    .agg(
        F.avg(
            "total_po_value"
        ).alias(
            "total_po_value"
        ),
        F.avg(
            "po_count"
        ).alias(
            "po_count"
        )
    )
)




# ==========================================================
# STEP 19 : JOIN HISTORICAL PO DATA
# ==========================================================
# Used for FRD-004 Value Spike detection.

silver_df = (
    silver_df
    .join(
        historical_po_values,
        on=[
            "vendor_id",
            "year_month"
        ],
        how="left"
    )
)




# ==========================================================
# STEP 20 : CALCULATE HISTORICAL AVERAGE
# ==========================================================

silver_df = (
    silver_df
    .withColumn(
        "avg_po_value",
        F.round(
            F.col("total_po_value")
            /
            F.col("po_count"),
            2
        )
    )
)




# ==========================================================
# STEP 21 : CALCULATE SPIKE RATIO
# ==========================================================
# Current PO Value / Historical Average

silver_df = (
    silver_df
    .withColumn(
        "spike_ratio",
        F.when(
            F.col("avg_po_value") > 0,
            F.round(
                F.col("total_amount")
                /
                F.col("avg_po_value"),
                2
            )
        ).otherwise(0)
    )
)




# ==========================================================
# STEP 22 : FRD-004 VALUE SPIKE FLAG
# ==========================================================
# Invoice value exceeds 3x historical average.

silver_df = (
    silver_df
    .withColumn(
        "value_spike",
        F.when(
            F.col("spike_ratio") > 3,
            1
        ).otherwise(0)
    )
)




# ==========================================================
# STEP 23 : NAME SIMILARITY UDF
# ==========================================================
# Compare invoice billing name
# with vendor trade name.



def similarity_score(name1, name2):

    if name1 is None or name2 is None:
        return 0.0

    return (
        fuzz.ratio(
            str(name1).lower(),
            str(name2).lower()
        )
        / 100.0
    )

similarity_udf = udf(
    similarity_score,
    DoubleType()
)





# ==========================================================
# STEP 24 : CALCULATE NAME SIMILARITY SCORE
# ==========================================================

silver_df = (
    silver_df
    .withColumn(
        "name_sim_score",
        similarity_udf(
            F.col(
                "invoice_billing_name"
            ),
            F.col(
                "trade_name"
            )
        )
    )
)






# ==========================================================
# STEP 25 : SIM-001 NAME MISMATCH FLAG
# ==========================================================
# Flag invoices where vendor name
# similarity is below threshold.

silver_df = (
    silver_df
    .withColumn(
        "name_mismatch",
        F.when(
            F.col(
                "name_sim_score"
            ) < 0.85,
            1
        ).otherwise(0)
    )
)






# ==========================================================
# STEP 26 : DUPLICATE INVOICE DETECTION
# ==========================================================
# Same GSTIN + Invoice Number + Year + Month

dup_window = Window.partitionBy(
    "gstin",
    "invoice_number",
    "year",
    "month"
)

silver_df = (
    silver_df
    .withColumn(
        "invoice_dup_count",
        F.count("*").over(dup_window)
    )
)

silver_df = (
    silver_df
    .withColumn(
        "duplicate_invoice",
        F.when(
            F.col("invoice_dup_count") > 1,
            1
        ).otherwise(0)
    )
)





# ==========================================================
# STEP 27 : DIRECTOR PAN OVERLAP
# ==========================================================
# Detect vendors sharing same PAN.

director_counts = (
    silver_df
    .groupBy("director_pan_1")
    .agg(
        F.countDistinct("vendor_id")
        .alias("director_vendor_count")
    )
)

silver_df = (
    silver_df
    .join(
        director_counts,
        "director_pan_1",
        "left"
    )
)

silver_df = (
    silver_df
    .withColumn(
        "director_overlap_flag",
        F.when(
            F.col("director_vendor_count") > 1,
            1
        ).otherwise(0)
    )
)





# ==========================================================
# STEP 28 : LATE INVOICE FLAG
# ==========================================================

silver_df = (
    silver_df
    .withColumn(
        "invoice_date",
        F.coalesce(
            F.to_date(
                "invoice_date",
                "dd/MM/yyyy"
            ),
            F.to_date(
                "invoice_date",
                "yyyy-MM-dd"
            )
        )
    )
)

silver_df = (
    silver_df
    .withColumn(
        "late_invoice",
        F.when(
            F.datediff(
                F.col("invoice_date"),
                F.col("po_date")
            ) > 30,
            1
        ).otherwise(0)
    )
)






# ==========================================================
# STEP 29 : FILING COMPLIANCE RATE
# ==========================================================

silver_df = (
    silver_df
    .withColumn(
        "filing_compliance_rate",
        (
            F.col("filing_history_q1")
            + F.col("filing_history_q2")
            + F.col("filing_history_q3")
            + F.col("filing_history_q4")
            + F.col("filing_history_q5")
            + F.col("filing_history_q6")
        ) / 6
    )
)







# ==========================================================
# STEP 30 : NON FILER FLAG
# ==========================================================

silver_df = (
    silver_df
    .withColumn(
        "vendor_non_filer",
        F.when(
            (
                F.col("filing_history_q1")
                + F.col("filing_history_q2")
                + F.col("filing_history_q3")
                + F.col("filing_history_q4")
                + F.col("filing_history_q5")
                + F.col("filing_history_q6")
            ) < 4,
            1
        ).otherwise(0)
    )
)







# ==========================================================
# STEP 31 : EWB VALIDATION
# ==========================================================

silver_df = (
    silver_df
    .withColumn(
        "missing_ewb",
        F.when(
            (
                F.col("total_amount") > 50000
            )
            &
            (
                F.col("ewb_number").isNull()
            ),
            1
        ).otherwise(0)
    )
)







# ==========================================================
# STEP 32 : COMPOSITE RULE SCORE
# ==========================================================

silver_df = (
    silver_df
    .withColumn(
        "rule_fail_count",
        F.col("blacklisted")
        + F.col("rate_mismatch")
        + F.col("value_spike")
        + F.col("name_mismatch")
        + F.col("duplicate_invoice")
        + F.col("missing_ewb")
        + F.col("director_overlap_flag")
        + F.col("state_mismatch")
        + F.col("late_invoice")
        + F.col("vendor_non_filer")
    )
)

silver_df = (
    silver_df
    .withColumn(
        "rule_score",
        F.round(
            (
                1-(
                    F.col("rule_fail_count")
                    / 10
                )
            )
            * 100,2)
    )
)






# ==========================================================
# STEP 33 : DATE STANDARDIZATION
# ==========================================================

silver_df = (
    silver_df
    .withColumn(
        "registration_date",
        F.coalesce(
            F.to_date(F.col("registration_date"), "dd/MM/yyyy"),
            F.to_date(F.col("registration_date"), "yyyy-MM-dd")
        )
    )
)



silver_df = (
    silver_df
    .withColumn(
        "ewb_generated_date",
        F.coalesce(
            F.to_date(
                F.col("ewb_generated_date"),
                "dd/MM/yyyy"
            ),
            F.to_date(
                F.col("ewb_generated_date"),
                "yyyy-MM-dd"
            )
        )
    )
)






# ==========================================================
# STEP 34 : YEAR MONTH TYPE FIX
# ==========================================================

silver_df = (
    silver_df
    .withColumn(
        "year",
        F.col("year").cast("int")
    )
    .withColumn(
        "month",
        F.col("month").cast("int")
    )
)







# ==========================================================
# STEP 35 : GROUND TRUTH CLEANUP
# ==========================================================

ground_truth = (
    ground_truth
    .toDF(
        "po_id",
        "anomaly_code",
        "severity",
        "is_anomalous"
    )
    .filter(
        F.col("po_id").startswith("PO-")
    )
    .withColumn(
        "is_anomalous",
        F.col("is_anomalous").cast("int")
    )
)







# ==========================================================
# STEP 36 : JOIN GROUND TRUTH
# ==========================================================

silver_df = (
    silver_df
    .join(
        ground_truth,
        on="po_id",
        how="left"
    )
)

silver_df = (
    silver_df
    .withColumn(
        "is_anomalous",
        F.coalesce(
            F.col("is_anomalous"),
            F.lit(0)
        )
    )
    .withColumn(
        "anomaly_code",
        F.coalesce(
            F.col("anomaly_code"),
            F.lit("CLEAN")
        )
    )
)







# ==========================================================
# STEP 37 : ADVANCED ML FEATURES
# ==========================================================

silver_df = (
    silver_df
    .withColumn(
        "vendor_age_days",
        F.datediff(
            F.current_date(),
            F.col("registration_date")
        )
    )
    .withColumn(
        "is_march_invoice",
        F.when(
            F.month("invoice_date") == 3,
            1
        ).otherwise(0)
    )
    .withColumn(
        "hsn_rate_delta",
        F.abs(
            F.col("actual_gst_rate")
            - F.col("gst_rate")
        )
    )
)







# ==========================================================
# STEP 38 : SAME DAY INVOICE FREQUENCY
# ==========================================================

same_day_window = Window.partitionBy(
    "vendor_id",
    "invoice_date"
)

silver_df = (
    silver_df
    .withColumn(
        "invoice_count_same_vendor_same_day",
        F.count("*").over(
            same_day_window
        )
    )
)






# ==========================================================
# STEP 39 : VENDOR PO FREQUENCY
# ==========================================================

vendor_window = Window.partitionBy(
    "vendor_id"
)

silver_df = (
    silver_df
    .withColumn(
        "vendor_po_frequency",
        F.count("po_id").over(
            vendor_window
        )
    )
)







# ==========================================================
# STEP 40 : HIGH VALUE FLAG
# ==========================================================

silver_df = (
    silver_df
    .withColumn(
        "high_value_invoice_flag",
        F.when(
            F.col("total_amount") > 500000,
            1
        ).otherwise(0)
    )
)









# ==========================================================
# STEP 41 : INVOICE TO AVG RATIO
# ==========================================================

silver_df = (
    silver_df
    .withColumn(
        "invoice_to_avg_ratio",
        F.when(
            F.col("avg_po_value") > 0,
            F.col("total_amount")
            /
            F.col("avg_po_value")
        ).otherwise(0)
    )
)







# ==========================================================
# STEP 42 : SAVE SILVER DATASET
# ==========================================================

silver_df.write \
    .mode("overwrite") \
    .parquet(
        silver_base + "silver_purchase_orders/"
    )


# ==========================================================
# STEP 43 : COMMIT GLUE JOB
# ==========================================================

job.commit()






