# ==========================================================
# GST ANOMALY DETECTION : JOB 02 
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

from pyspark.sql.functions import (
    col,
    when,
    rand,
    concat,
    lit,
    lpad,
    row_number
)
from pyspark.sql.functions import monotonically_increasing_id


# ==========================================================
# STEP 2 : INITIALIZE AWS GLUE JOB
# ==========================================================

args = getResolvedOptions(
    sys.argv,
    [
        "JOB_NAME",
        "RAW_BUCKET"
    ]
)

sc = SparkContext()
glueContext = GlueContext(sc)

spark = glueContext.spark_session

job = Job(glueContext)
job.init(args["JOB_NAME"], args)

# ==========================================================
# STEP 3 : DEFINE S3 PATHS
# ==========================================================

raw_bucket = args["RAW_BUCKET"]

base_path = f"s3://{raw_bucket}/raw/raw/"

po_path = base_path + "PO_Records.csv"
vendor_path = base_path + "Vendor_Master.csv"
hsn_path = base_path + "HSN_Rate_Schedule.csv"
blacklist_path = base_path + "CBIC_Blacklist.csv"
ground_truth_path = base_path + "Ground_Truth.csv"


# ==========================================================
# STEP 4 : LOAD SOURCE DATASETS
# ==========================================================


po_df = (
    spark.read
    .option("header", "true")
    .option("inferSchema", "true")
    .csv(po_path)
)

vendor_df = (
    spark.read
    .option("header", "true")
    .option("inferSchema", "true")
    .csv(vendor_path)
)

hsn_df = (
    spark.read
    .option("header", "true")
    .option("inferSchema", "true")
    .csv(hsn_path)
)

blacklist_df = (
    spark.read
    .option("header", "true")
    .option("inferSchema", "true")
    .csv(blacklist_path)
)

ground_truth_raw = (
    spark.read
    .option("header", "true")
    .csv(ground_truth_path)
)


# ==========================================================
# STEP 5 : CLEAN GROUND TRUTH FILE
# ==========================================================


ground_truth_df = (
    ground_truth_raw
    .toDF(
        "po_id",
        "anomaly_code",
        "severity",
        "is_anomalous"
    )
    .filter(
        col("po_id").startswith("PO-")
    )
    .withColumn(
        "is_anomalous",
        col("is_anomalous").cast("int")
    )
)



# ==========================================================
# STEP 6 : STANDARDIZE ANOMALY CODE FRD006->FRD002(ground Truth)
# ==========================================================

ground_truth_df = ground_truth_df.withColumn(
    "anomaly_code",
    when(
        col("anomaly_code") == "FRD-006",
        "FRD-002"
    ).otherwise(
        col("anomaly_code")
    )
)




# ==========================================================
# STEP 7 : CREATE STANDARDIZED VENDOR MASTER
# ==========================================================

vendors_df = vendor_df.select(
    F.col("Vendor ID").alias("vendor_id"),
    F.col("GSTIN").alias("gstin"),
    F.col("Trade Name").alias("trade_name"),
    F.col("Legal Name").alias("legal_name"),
    F.col("Billing State").alias("billing_state"),
    F.col("Reg Date").alias("registration_date"),
    F.col("Status").alias("status"),
    F.col("Composition?").alias("composition_flag"),
    F.col("Q1").alias("filing_history_q1"),
    F.col("Q2").alias("filing_history_q2"),
    F.col("Q3").alias("filing_history_q3"),
    F.col("Q4").alias("filing_history_q4"),
    F.col("Q5").alias("filing_history_q5"),
    F.col("Q6").alias("filing_history_q6"),
    F.col("Director PAN 1").alias("director_pan_1"),
    F.col("Director PAN 2").alias("director_pan_2")
)





# ==========================================================
# STEP 8 : CREATE STANDARDIZED PURCHASE ORDERS
# ==========================================================

purchase_orders_df = po_df.select(
    F.col("PO ID").alias("po_id"),
    F.col("PO Date").alias("po_date"),
    F.col("Vendor ID").alias("vendor_id"),
    F.col("Buyer GSTIN").alias("buyer_gstin"),
    F.col("Buyer State").alias("buyer_state"),
    F.col("HSN Code").alias("hsn_code"),
    F.col("Product").alias("product_desc"),
    F.col("Qty").cast("int").alias("quantity"),
    F.col("Unit").alias("unit"),
    F.col("`Base Amt (₹)`").cast("double").alias("base_amount"),
    F.col("`CGST%`").cast("double").alias("cgst_rate"),
    F.col("`SGST%`").cast("double").alias("sgst_rate"),
    F.col("`IGST%`").cast("double").alias("igst_rate"),
    F.col("CGST Amt").cast("double").alias("cgst_amt"),
    F.col("SGST Amt").cast("double").alias("sgst_amt"),
    F.col("IGST Amt").cast("double").alias("igst_amt"),
    F.col("Cess").cast("double").alias("cess_amt"),
    F.col("`Total Inv (₹)`").cast("double").alias("total_amount"),
    F.col("`EWB No.`").alias("ewb_number"),
    F.col("EWB Date").alias("ewb_generated_date"),
    F.col("Place of Supply").alias("place_of_supply"),
    F.col("Delivery State").alias("delivery_state"),
    F.col("Billing Name").alias("invoice_billing_name"),
    F.col("Trade Name").alias("po_vendor_name")
)





# ==========================================================
# STEP 9 : CREATE HSN RATE SCHEDULE
# ==========================================================

hsn_rate_schedule_df = hsn_df.select(
    F.col("HSN Code").alias("hsn_code"),
    F.col("Product Description").alias("description"),
    F.col("`GST Rate %`").cast("double").alias("gst_rate"),
    F.col("`Cess?`").alias("cess_applicable"),
    F.col("`Effective From`").alias("effective_from")
)





# ==========================================================
# STEP 10 : CREATE VENDOR INVOICE DATASET
# ==========================================================

vendor_invoices_df = po_df.select(
    F.col("PO ID").alias("po_id"),
    F.col("Inv Date").alias("invoice_date"),
    F.col("`Invoice No.`").alias("invoice_number"),
    F.col("`GSTR2B ITC (₹)`").cast("double").alias("gstr2b_itc_available"),
    F.col("`ITC Claimed (₹)`").cast("double").alias("itc_claimed_by_buyer")
)





# ==========================================================
# STEP 11 : CREATE CBIC BLACKLIST DATASET
# ==========================================================

#: Original Blacklist

original_blacklist_df = blacklist_df.select(
    F.col("Blacklisted GSTIN").alias("gstin"),
    F.col("Blacklist Date").alias("blacklist_date"),
    F.col("Reason").alias("reason"),
    F.col("Case Reference").alias("case_reference")
)

#: GENERATE SYNTHETIC BLACKLIST

synthetic_blacklist_df = (
    vendors_df
    .select("gstin")
    .sample(False, 0.05, seed=42)
    .withColumn("blacklist_date", F.current_date())
    .withColumn("reason", F.lit("Fake GST Registration"))
    .withColumn("case_reference", F.lit("CBIC-TEST"))
)



#UNION Original + Synethetic
cbic_blacklist_df = (
    original_blacklist_df
    .unionByName(synthetic_blacklist_df)
    .dropDuplicates(["gstin"])
)





# ==========================================================
# STEP 12 : CREATE HISTORICAL PO VALUES
# ==========================================================



historical_po_values_df = po_df.select(
    F.col("Vendor ID").alias("vendor_id"),
    F.date_format(
        F.to_date(
            F.col("PO Date"),
            "dd/MM/yyyy"
        ),
        "yyyy-MM"
    ).alias("year_month"),
    F.col("`6M Avg (₹)`").cast("double").alias("total_po_value")
)

historical_po_values_df = (
    historical_po_values_df
    .groupBy(
        "vendor_id",
        "year_month"
    )
    .agg(
        F.avg("total_po_value").alias("total_po_value"),
        F.count("*").alias("po_count")
    )
)


# ==========================================================
# Available DataFrames:
# vendors_df
# purchase_orders_df
# vendor_invoices_df
# hsn_rate_schedule_df
# cbic_blacklist_df
# historical_po_values_df
# ground_truth_df
# ==========================================================



# ==========================================================
# STEP 13 : EXPAND VENDOR MASTER TO 1000 RECORDS
# ==========================================================

target_vendors = 1000

vendor_multiplier = (
    int(target_vendors / vendors_df.count())
    + 1
)

raw_vendors_df = (
    vendors_df
    .crossJoin(
        spark.range(vendor_multiplier)
    )
    .limit(target_vendors)
)





# ==========================================================
# STEP 14 : DIRECTOR PAN OVERLAP INJECTION
# ==========================================================
# Inject shared PAN values to simulate
# related-party / circular billing scenarios.

raw_vendors_df = raw_vendors_df.withColumn(
    "director_pan_1",
    F.when(
        F.rand(seed=42) < 0.05,
        F.lit("ABCDE1234F")
    ).otherwise(
        F.col("director_pan_1")
    )
)





# ==========================================================
# STEP 15 : GENERATE UNIQUE VENDOR IDS
# ==========================================================

vendor_window = Window.orderBy(
    F.monotonically_increasing_id()
)

raw_vendors_df = raw_vendors_df.withColumn(
    "vendor_id",
    F.concat(
        F.lit("V"),
        F.lpad(
            F.row_number().over(vendor_window),
            5,
            "0"
        )
    )
)





# ==========================================================
# STEP 16 : GENERATE UNIQUE GSTINS
# ==========================================================
# Prevent duplicate GSTINs after vendor expansion.

raw_vendors_df = raw_vendors_df.withColumn(
    "gstin",
    F.concat(
        F.substring("gstin", 1, 10),
        F.lpad(
            F.row_number().over(vendor_window),
            4,
            "0"
        ),
        F.substring("gstin", 15, 1)
    )
)




# ==========================================================
# STEP 17 : EXPAND PURCHASE ORDERS TO 50000 RECORDS
# ==========================================================

target_pos = 50000

po_multiplier = (
    int(target_pos / purchase_orders_df.count())
    + 1
)

raw_purchase_orders_df = (
    purchase_orders_df
    .crossJoin(
        spark.range(po_multiplier)
    )
    .limit(target_pos)
)





# ==========================================================
# STEP 18 : CANCELLED VENDOR INJECTION
# ==========================================================
# Randomly mark vendors as cancelled.
# Used later for GST-001 anomaly generation.

cancelled_vendors = (
    raw_vendors_df
    .orderBy(F.rand(seed=42))
    .limit(50)
    .select("vendor_id")
)

raw_purchase_orders_df = (
    raw_purchase_orders_df
    .join(
        cancelled_vendors.withColumn(
            "status",
            F.lit("Cancelled")
        ),
        "vendor_id",
        "left"
    )
)





# ==========================================================
# STEP 19 : GENERATE UNIQUE PO IDs
# ==========================================================

po_window = Window.orderBy(
    F.monotonically_increasing_id()
)

raw_purchase_orders_df = (
    raw_purchase_orders_df
    .withColumn(
        "po_id",
        F.concat(
            F.lit("PO"),
            F.lpad(
                F.row_number().over(po_window),
                7,
                "0"
            )
        )
    )
)





# ==========================================================
# STEP 20 : CREATE VENDOR LOOKUP TABLE
# ==========================================================

vendor_lookup = (
    raw_vendors_df
    .select("vendor_id")
    .withColumn(
        "rn",
        F.row_number().over(
            Window.orderBy("vendor_id")
        )
    )
)



# ==========================================================
# STEP 21 : RANDOM VENDOR ASSIGNMENT
# ==========================================================

raw_purchase_orders_df = (
    raw_purchase_orders_df
    .drop("vendor_id")
    .withColumn(
        "rn",
        (
            F.floor(
                F.rand(seed=42) * 1000
            ) + 1
        ).cast("int")
    )
)





# ==========================================================
# STEP 22 : STATE MISMATCH INJECTION
# ==========================================================
# Inject approximately 300 GST state mismatch
# anomalies into the dataset.
# ==========================================================

raw_purchase_orders_df = (
    raw_purchase_orders_df
    .withColumn(
        "state_mismatch_seed",
        F.rand(seed=42)
    )
)

raw_purchase_orders_df = (
    raw_purchase_orders_df
    .withColumn(
        "buyer_state",
        F.when(
            F.col("state_mismatch_seed") < 0.006,
            F.lit("Karnataka")
        ).otherwise(
            F.col("buyer_state")
        )
    )
    .drop("state_mismatch_seed")
)






# ==========================================================
# STEP 23 : ASSIGN EXPANDED VENDOR IDS
# ==========================================================

raw_purchase_orders_df = (
    raw_purchase_orders_df
    .withColumn(
        "rn",
        (
            F.rand(seed=42) * 1000
        ).cast("int") + 1
    )
)

raw_purchase_orders_df = (
    raw_purchase_orders_df
    .join(
        vendor_lookup,
        on="rn",
        how="left"
    )
    .drop("rn")
)






# ==========================================================
# STEP 24 : EXPAND INVOICES TO 50000 RECORDS
# ==========================================================

target_invoices = 50000

invoice_multiplier = (
    int(
        target_invoices /
        vendor_invoices_df.count()
    )
    + 1
)

raw_vendor_invoices_df = (
    vendor_invoices_df
    .crossJoin(
        spark.range(invoice_multiplier)
    )
    .limit(target_invoices)
)




# ==========================================================
# STEP 25 : GENERATE UNIQUE INVOICE NUMBERS
# ==========================================================

invoice_window = Window.orderBy(
    F.monotonically_increasing_id()
)

raw_vendor_invoices_df = (
    raw_vendor_invoices_df
    .withColumn(
        "invoice_number",
        F.concat(
            F.lit("INV"),
            F.lpad(
                F.row_number().over(
                    invoice_window
                ),
                8,
                "0"
            )
        )
    )
)





# ==========================================================
# STEP 26 : CREATE PO LOOKUP TABLE
# ==========================================================

po_lookup = (
    raw_purchase_orders_df
    .select("po_id")
    .withColumn(
        "rn",
        F.row_number().over(
            Window.orderBy("po_id")
        )
    )
)





# ==========================================================
# STEP 27 : MAP INVOICES TO PURCHASE ORDERS
# ==========================================================

raw_vendor_invoices_df = (
    raw_vendor_invoices_df
    .withColumn(
        "rn",
        F.row_number().over(
            Window.orderBy(
                F.monotonically_increasing_id()
            )
        )
    )
)

raw_vendor_invoices_df = (
    raw_vendor_invoices_df
    .drop("po_id")
    .join(
        po_lookup,
        "rn"
    )
    .drop("rn")
)




# ==========================================================
# raw_vendors_df
# raw_purchase_orders_df
# raw_vendor_invoices_df
# ==========================================================





# ==========================================================
# STEP 28 : EXPAND HISTORICAL PO VALUES
# ==========================================================
# requires approximately 12000
# historical vendor records.


target_history = 12000

history_multiplier = (
    int(
        target_history /
        historical_po_values_df.count()
    )
    + 1
)

raw_historical_po_values_df = (
    historical_po_values_df
    .crossJoin(
        spark.range(history_multiplier)
    )
    .limit(target_history)
)





# ==========================================================
# STEP 29 : ASSIGN EXPANDED VENDOR IDS
# ==========================================================

vendor_lookup = (
    raw_vendors_df
    .select("vendor_id")
    .withColumn(
        "rn",
        F.row_number().over(
            Window.orderBy("vendor_id")
        )
    )
)

raw_historical_po_values_df = (
    raw_historical_po_values_df
    .drop("vendor_id")
)

raw_historical_po_values_df = (
    raw_historical_po_values_df
    .withColumn(
        "rn",
        (
            (F.rand(seed=42) * 1000)
            .cast("int")
            + 1
        )
    )
)

raw_historical_po_values_df = (
    raw_historical_po_values_df
    .join(
        vendor_lookup,
        "rn",
        "left"
    )
    .drop("rn")
)






# ==========================================================
# STEP 30 : STANDARDIZE BOOLEAN FLAGS
# ==========================================================
# Convert Yes/No, ✓/✗ values into
# binary integer flags.


bool_cols = [
    "composition_flag",
    "filing_history_q1",
    "filing_history_q2",
    "filing_history_q3",
    "filing_history_q4",
    "filing_history_q5",
    "filing_history_q6"
]

for c in bool_cols:

    raw_vendors_df = raw_vendors_df.withColumn(
        c,
        F.when(
            F.col(c).isin(
                "✓",
                "✔",
                "Y",
                "Yes",
                "TRUE",
                "true"
            ),
            1
        )
        .when(
            F.col(c).isin(
                "✗",
                "✘",
                "N",
                "No",
                "FALSE",
                "false"
            ),
            0
        )
        .otherwise(F.lit(0))
    )






# ==========================================================
# STEP 31 : GST-005 COMPOSITION DEALER
# ==========================================================
# Inject approximately 80 composition
# dealer vendors.

raw_vendors_df = (
    raw_vendors_df
    .withColumn(
        "composition_flag",
        when(
            rand(200) < 0.08,
            1
        ).otherwise(
            col("composition_flag")
        )
    )
)






# ==========================================================
# STEP 32 : ITC-001 NON FILER VENDORS
# ==========================================================
# Reduce filing compliance across
# filing history quarters.

for q in range(1, 7):

    raw_vendors_df = raw_vendors_df.withColumn(
        f"filing_history_q{q}",
        when(
            rand(q * 10) < 0.4,
            0
        ).otherwise(
            col(f"filing_history_q{q}")
        )
    )






# ==========================================================
# STEP 33 : REGISTRATION DATE CLEANUP
# ==========================================================

raw_vendors_df = (
    raw_vendors_df
    .withColumn(
        "registration_date",
        F.to_date(
            col("registration_date"),
            "dd/MM/yyyy"
        )
    )
)





# ==========================================================
# STEP 34 : FRD-005 FLY-BY-NIGHT VENDOR
# ==========================================================
# Recently registered vendors.

raw_vendors_df = (
    raw_vendors_df
    .withColumn(
        "registration_date",
        when(
            rand(1000) < 0.06,
            F.date_sub(
                F.current_date(),
                30
            )
        ).otherwise(
            col("registration_date")
        )
    )
)





# ==========================================================
# STEP 35 : TAX-001 HSN RATE MISMATCH
# ==========================================================

raw_purchase_orders_df = (
    raw_purchase_orders_df
    .withColumn(
        "cgst_rate",
        when(
            rand(400) < 0.007,
            col("cgst_rate") + 5
        ).otherwise(
            col("cgst_rate")
        )
    )
)






# ==========================================================
# STEP 36 : TAX-002 WRONG TAX TYPE
# ==========================================================

raw_purchase_orders_df = (
    raw_purchase_orders_df
    .withColumn(
        "igst_rate",
        when(
            rand(500) < 0.004,
            18
        ).otherwise(
            col("igst_rate")
        )
    )
)






# ==========================================================
# STEP 37 : EWB-001 MISSING EWB
# ==========================================================

raw_purchase_orders_df = (
    raw_purchase_orders_df
    .withColumn(
        "ewb_number",
        when(
            rand(600) < 0.01,
            None
        ).otherwise(
            col("ewb_number")
        )
    )
)





# ==========================================================
# STEP 38 : EWB-002 QUANTITY MISMATCH
# ==========================================================

raw_purchase_orders_df = (
    raw_purchase_orders_df
    .withColumn(
        "quantity",
        when(
            rand(700) < 0.0036,
            col("quantity") * 10
        ).otherwise(
            col("quantity")
        )
    )
)






# ==========================================================
# STEP 39 : FRD-002 CIRCULAR BILLING
# ==========================================================

cluster_vendors = (
    raw_vendors_df
    .orderBy(rand(42))
    .limit(120)
    .withColumn(
        "cluster_id",
        (
            (
                row_number().over(
                    Window.orderBy("vendor_id")
                ) - 1
            ) / 4
        ).cast("int")
    )
)

shared_pans = (
    cluster_vendors
    .select("cluster_id")
    .distinct()
    .withColumn(
        "shared_pan_1",
        concat(
            lit("AAAAA"),
            lpad(
                col("cluster_id"),
                6,
                "0"
            )
        )
    )
    .withColumn(
        "shared_pan_2",
        concat(
            lit("BBBBB"),
            lpad(
                col("cluster_id"),
                6,
                "0"
            )
        )
    )
)

cluster_vendors = (
    cluster_vendors
    .join(
        shared_pans,
        "cluster_id"
    )
    .drop(
        "director_pan_1",
        "director_pan_2"
    )
    .withColumnRenamed(
        "shared_pan_1",
        "director_pan_1"
    )
    .withColumnRenamed(
        "shared_pan_2",
        "director_pan_2"
    )
)

remaining_vendors = (
    raw_vendors_df
    .join(
        cluster_vendors.select("vendor_id"),
        "vendor_id",
        "left_anti"
    )
)

raw_vendors_df = (
    remaining_vendors
    .unionByName(
        cluster_vendors.select(
            raw_vendors_df.columns
        )
    )
)





# ==========================================================
# STEP 40 : FRD-003 SPLIT INVOICE
# ==========================================================

split_po = (
    raw_purchase_orders_df
    .orderBy(rand(99))
    .limit(240)
)

split_po = (
    split_po
    .withColumn(
        "base_amount",
        lit(30000.0)
    )
    .withColumn(
        "cgst_amt",
        lit(2700.0)
    )
    .withColumn(
        "sgst_amt",
        lit(2700.0)
    )
    .withColumn(
        "igst_amt",
        lit(0.0)
    )
    .withColumn(
        "total_amount",
        lit(35400.0)
    )
)

split_po_1 = (
    split_po
    .withColumn(
        "po_id",
        concat(
            col("po_id"),
            lit("_A")
        )
    )
)

split_po_2 = (
    split_po
    .withColumn(
        "po_id",
        concat(
            col("po_id"),
            lit("_B")
        )
    )
)

raw_purchase_orders_df = (
    raw_purchase_orders_df
    .unionByName(split_po_1)
    .unionByName(split_po_2)
)

# ==========================================================
# STEP 41 : FRD-004 VALUE SPIKE
# ==========================================================

raw_purchase_orders_df = (
    raw_purchase_orders_df
    .withColumn(
        "base_amount",
        when(
            rand(800) < 0.0018,
            col("base_amount") * 4
        ).otherwise(
            col("base_amount")
        )
    )
)

# ==========================================================
# STEP 42 : SIM-001 NAME MISMATCH
# ==========================================================

raw_purchase_orders_df = (
    raw_purchase_orders_df
    .withColumn(
        "invoice_billing_name",
        when(
            rand(900) < 0.012,
            lit("XYZ TRADING COMPANY")
        ).otherwise(
            col("invoice_billing_name")
        )
    )
)

# ==========================================================
# STEP 43 : ITC-002 EXCESS ITC CLAIM
# ==========================================================

raw_vendor_invoices_df = (
    raw_vendor_invoices_df
    .withColumn(
        "itc_claimed_by_buyer",
        when(
            rand(300) < 0.005,
            col(
                "gstr2b_itc_available"
            ) * 1.5
        ).otherwise(
            col("itc_claimed_by_buyer")
        )
    )
)

# ==========================================================
# STEP 44 : DQ-001 DUPLICATE INVOICE
# ==========================================================

duplicate_rows = (
    raw_vendor_invoices_df
    .limit(100)
)

raw_vendor_invoices_df = (
    raw_vendor_invoices_df
    .union(duplicate_rows)
)

# ==========================================================
# STEP 45 : FINAL TYPE STANDARDIZATION
# ==========================================================

flag_cols = [
    "composition_flag",
    "filing_history_q1",
    "filing_history_q2",
    "filing_history_q3",
    "filing_history_q4",
    "filing_history_q5",
    "filing_history_q6"
]

for c in flag_cols:

    raw_vendors_df = (
        raw_vendors_df
        .withColumn(
            c,
            F.col(c)
            .cast("double")
            .cast("int")
        )
    )

raw_historical_po_values_df = (
    raw_historical_po_values_df
    .withColumn(
        "po_count",
        F.col("po_count")
        .cast("double")
        .cast("int")
    )
)


# ==========================================================
# STEP 46 : GENERATE GROUND TRUTH DATASET
# ==========================================================
# Create anomaly labels for ML training.
# Each PO is assigned an anomaly code
# based on controlled probability ranges.


raw_ground_truth_df = (
    raw_purchase_orders_df
    .select("po_id")
    .withColumn(
        "r",
        F.rand(42)
    )
    .withColumn(
        "anomaly_code",
        F.when(F.col("r") < 0.006, "GST-001")
        .when(F.col("r") < 0.012, "GST-003")
        .when(F.col("r") < 0.015, "GST-004")
        .when(F.col("r") < 0.017, "GST-005")
        .when(F.col("r") < 0.025, "ITC-001")
        .when(F.col("r") < 0.030, "ITC-002")
        .when(F.col("r") < 0.037, "TAX-001")
        .when(F.col("r") < 0.041, "TAX-002")
        .when(F.col("r") < 0.051, "EWB-001")
        .when(F.col("r") < 0.055, "EWB-002")
        .when(F.col("r") < 0.057, "FRD-002")
        .when(F.col("r") < 0.0585, "FRD-003")
        .when(F.col("r") < 0.060, "FRD-004")
        .otherwise("CLEAN")
    )
    .withColumn(
        "is_anomalous",
        F.when(
            F.col("anomaly_code") == "CLEAN",
            0
        ).otherwise(1)
    )
    .withColumn(
        "severity",
        F.when(
            F.col("anomaly_code") == "CLEAN",
            "Low"
        ).otherwise("High")
    )
    .drop("r")
)






# ==========================================================
# STEP 47 : DEFINE OUTPUT PATHS
# ==========================================================

output_base = f"s3://{raw_bucket}/raw/enrich_raw/"

vendors_output = output_base + "vendors/"
purchase_orders_output = output_base + "purchase_orders/"
vendor_invoices_output = output_base + "vendor_invoices/"
ground_truth_output = output_base + "ground_truth/"
historical_output = output_base + "historical_po_values/"
hsn_output = output_base + "hsn_rate_schedule/"
blacklist_output = output_base + "cbic_blacklist/"

# ==========================================================
# STEP 48 : SAVE RAW VENDOR DATA
# ==========================================================

#Vendor Data

raw_vendors_df.write.mode(
    "overwrite"
).parquet(
    vendors_output
)


#RAW PURCHASE ORDERS

raw_purchase_orders_df.write.mode(
    "overwrite"
).parquet(
    purchase_orders_output
)


#RAW INVOICES

raw_vendor_invoices_df.write.mode(
    "overwrite"
).parquet(
    vendor_invoices_output
)


#GROUND TRUTH

raw_ground_truth_df.write.mode(
    "overwrite"
).parquet(
    ground_truth_output
)

#HISTORICAL PO VALUES

raw_historical_po_values_df.write.mode(
    "overwrite"
).parquet(
    historical_output
)

#HSN RATE SCHEDULE

hsn_rate_schedule_df.write.mode(
    "overwrite"
).parquet(
    hsn_output
)

#CBIC BLACKLIST

cbic_blacklist_df.write.mode(
    "overwrite"
).parquet(
    blacklist_output
)



#COMMIT GLUE JOB

job.commit()






