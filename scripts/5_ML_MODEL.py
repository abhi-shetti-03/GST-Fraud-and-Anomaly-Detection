# ==========================================================
# STEP 1 : IMPORT REQUIRED LIBRARIES
# ==========================================================
# Load AWS Glue, Spark, ML and S3 libraries.

import sys
import joblib
import boto3
import pandas as pd

from awsglue.utils import getResolvedOptions
from awsglue.context import GlueContext
from awsglue.job import Job

from pyspark.context import SparkContext
from pyspark.sql import functions as F


# ==========================================================
# STEP 2 : INITIALIZE AWS GLUE JOB
# ==========================================================
# Create Spark and Glue contexts.

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
# Define Silver and Gold layer locations.

silver_base = (f"s3://{args['SILVER_BUCKET']}/silver/")

gold_base = (f"s3://{args['GOLD_BUCKET']}/gold/")


# ==========================================================
# STEP 4 : LOAD SILVER DATASET
# ==========================================================


silver_df = spark.read.parquet(
    silver_base +
    "silver_purchase_orders/"
)


# ==========================================================
# STEP 5 : DEFINE ML FEATURE LIST
# ==========================================================

feature_cols = [

    "name_sim_score",
    "spike_ratio",
    "vendor_age_days",
    "filing_compliance_rate",
    "is_march_invoice",
    "invoice_count_same_vendor_same_day",
    "hsn_rate_delta",
    "director_overlap_flag",
    "blacklisted",
    "rate_mismatch",
    "value_spike",
    "duplicate_invoice",
    "vendor_non_filer",
    "missing_ewb",
    "high_value_invoice_flag",
    "vendor_po_frequency",
    "invoice_to_avg_ratio",
    "rule_fail_count"
]


# ==========================================================
# STEP 6 : HANDLE NULL VALUES
# ==========================================================
# Replace null values before inference.

silver_df = (
    silver_df
    .fillna(
        0,
        subset=feature_cols
    )
)


# ==========================================================
# STEP 7 : CONVERT TO PANDAS
# ==========================================================

pdf = (
    silver_df
    .select(
        ["po_id", "is_anomalous"]
        +
        feature_cols
    )
    .toPandas()
)


# ==========================================================
# STEP 8 : DOWNLOAD TRAINED MODELS FROM S3
# ==========================================================
# Download Random Forest and XGBoost models.

s3 = boto3.client("s3")

s3.download_file(
    "gst-fraud-and-anomaly-detection",
    "ML_models/xgboost_model.pkl",
    "/tmp/xgboost_model.pkl"
)

s3.download_file(
    "gst-fraud-and-anomaly-detection",
    "ML_models/Random_forest_model.pkl",
    "/tmp/random_forest_model.pkl"
)


# ==========================================================
# STEP 9 : LOAD TRAINED MODELS
# ==========================================================
# Load pkl models using joblib.

xgb = joblib.load(
    "/tmp/xgboost_model.pkl"
)

rf = joblib.load(
    "/tmp/random_forest_model.pkl"
)


# ==========================================================
# STEP 10 : PREPARE FEATURE MATRIX
# ==========================================================


X = pdf[feature_cols]




# ==========================================================
# STEP 11 : GENERATE RANDOM FOREST PREDICTIONS
# ==========================================================
# Generate Random Forest prediction labels and probabilities.

rf_prob = (
    rf
    .predict_proba(X)[:, 1]
)

rf_pred = (
    rf
    .predict(X)
)


# ==========================================================
# STEP 12 : GENERATE XGBOOST PREDICTIONS
# ==========================================================
# Generate XGBoost prediction labels and probabilities.

xgb_prob = (
    xgb
    .predict_proba(X)[:, 1]
)

xgb_pred = (
    xgb_prob >= 0.35
).astype(int)



# ==========================================================
# STEP 12A : ASSIGN RISK TIERS
# ==========================================================
# Convert prediction probability into business risk levels.

def get_risk_tier(prob):

    if prob >= 0.85:
        return "Critical"

    elif prob >= 0.70:
        return "High"

    elif prob >= 0.50:
        return "Medium"

    else:
        return "Low"



# ==========================================================
# STEP 13 : CREATE RF PREDICTION TABLE
# ==========================================================

rf_predictions = pd.DataFrame({

    "po_id":
        pdf["po_id"],

    "actual":
        pdf["is_anomalous"],

    "rf_prediction":
        rf_pred,

    "rf_probability":
        rf_prob,

    "model_name":
        "Random Forest"

})


# ==========================================================
# STEP 14 : CREATE XGB PREDICTION TABLE
# ==========================================================

xgb_predictions = pd.DataFrame({

    "po_id":
        pdf["po_id"],

    "actual":
        pdf["is_anomalous"],

    "xgb_prediction":
        xgb_pred,

    "xgb_probability":
        xgb_prob,

    "model_name":
        "XGBoost",
    
    "risk_tier":
        [get_risk_tier(x) for x in xgb_prob]

})


# ==========================================================
# STEP 15 : CREATE MODEL COMPARISON TABLE
# ==========================================================
# Store evaluation metrics obtained during model training.



from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score
)

actual = pdf["is_anomalous"]



#RF
rf_acc = accuracy_score(actual, rf_pred)

rf_precision = precision_score(actual,rf_pred,zero_division=0)

rf_recall = recall_score(actual,rf_pred,zero_division=0)

rf_f1 = f1_score(actual,rf_pred,zero_division=0)

try:
    rf_auc = roc_auc_score(actual, rf_prob)
except:
    rf_auc = 0.0


#XGB
xgb_acc = accuracy_score(actual, xgb_pred)

xgb_precision = precision_score(actual, xgb_pred,zero_division=0)

xgb_recall = recall_score(actual, xgb_pred,zero_division=0)

xgb_f1 = f1_score(actual, xgb_pred,zero_division=0)

try:
    xgb_auc = roc_auc_score(actual, xgb_prob)
except:
    xgb_auc = 0.0


comparison_table = pd.DataFrame(

    [

        [
            "Random Forest",
            rf_acc,
            rf_precision,
            rf_recall,
            rf_f1,
            rf_auc
        ],

        [
            "XGBoost",
            xgb_acc,
            xgb_precision,
            xgb_recall,
            xgb_f1,
            xgb_auc
        ]

    ],

    columns=[

        "Model",
        "Accuracy",
        "Precision",
        "Recall",
        "F1",
        "ROC_AUC"

    ]
)


# ==========================================================
# STEP 16 : GENERATE FEATURE IMPORTANCE TABLE
# ==========================================================
# Extract feature importance from Random Forest and XGBoost.

fi_rows = []

for feature_name, importance in zip(
    feature_cols,
    rf.feature_importances_
):

    fi_rows.append(

        [
            "Random Forest",
            feature_name,
            float(importance)
        ]

    )

for feature_name, importance in zip(
    feature_cols,
    xgb.feature_importances_
):

    fi_rows.append(

        [
            "XGBoost",
            feature_name,
            float(importance)
        ]

    )

feature_importance = pd.DataFrame(

    fi_rows,

    columns=[
        "model_name",
        "feature_name",
        "importance_score"
    ]

)


# ==========================================================
# STEP 17 : SAVE RF PREDICTIONS TO GOLD
# ==========================================================
# Save Random Forest predictions to Gold layer.

spark.createDataFrame(
    rf_predictions
).write.mode(
    "overwrite"
).parquet(

    gold_base +
    "rf_predictions/"

)


# ==========================================================
# STEP 18 : SAVE XGB PREDICTIONS TO GOLD
# ==========================================================
# Save XGBoost predictions to Gold layer.

spark.createDataFrame(
    xgb_predictions
).write.mode(
    "overwrite"
).parquet(

    gold_base +
    "xgb_predictions/"

)


# ==========================================================
# STEP 19 : SAVE COMPARISON TABLE TO GOLD
# ==========================================================
# Save model comparison metrics table.

spark.createDataFrame(
    comparison_table
).write.mode(
    "overwrite"
).parquet(

    gold_base +
    "comparison_table/"

)


# ==========================================================
# STEP 20 : SAVE FEATURE IMPORTANCE TO GOLD
# ==========================================================
# Save feature importance table.

spark.createDataFrame(
    feature_importance
).write.mode(
    "overwrite"
).parquet(

    gold_base +
    "feature_importance/"

)


# ==========================================================
# STEP 21 : COMMIT GLUE JOB
# ==========================================================
# Commit AWS Glue Job.

job.commit()













