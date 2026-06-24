# ==========================================================
# GST ANOMALY DETECTION : JOB 01 
# ==========================================================



# --additional-python-modules
# pandas==2.2.2,openpyxl==3.1.2


# =====================================================
# STEP 1 : IMPORT REQUIRED LIBRARIES
# =====================================================

import sys
from awsglue.utils import getResolvedOptions
from awsglue.context import GlueContext
from awsglue.job import Job
from pyspark.context import SparkContext

# =====================================================
# STEP 2 : INITIALIZE GLUE JOB
# =====================================================
# Create Spark Context
# Create Glue Context
# Start Job Session
# =====================================================

args = getResolvedOptions(
    sys.argv,
    [
        "JOB_NAME",
        "INPUT_BUCKET",
        "OUTPUT_BUCKET"
    ]
)

sc = SparkContext.getOrCreate()
glueContext = GlueContext(sc)

spark = glueContext.spark_session

job = Job(glueContext)
job.init(args["JOB_NAME"], args)





# ==========================================================
# STEP 1: IMPORT REQUIRED LIBRARIES
# ==========================================================

import pandas as pd
import boto3
import re
from io import BytesIO

# ==========================================================
# STEP 2: DEFINE S3 LOCATIONS
# ==========================================================


input_bucket = args["INPUT_BUCKET"]
output_bucket = args["OUTPUT_BUCKET"]

input_key = "raw/raw/GST_Anomaly_Dataset_NexuSolve.xlsx"

output_prefix = "raw/raw/"


# ==========================================================
# STEP 3: DOWNLOAD Excel file from S3
# ==========================================================

s3 = boto3.client("s3")

excel_obj = s3.get_object(
    Bucket=input_bucket,
    Key=input_key
)

excel_bytes = excel_obj["Body"].read()


# ==========================================================
# STEP 4: LOAD Excel
# ==========================================================



excel_file = pd.ExcelFile(BytesIO(excel_bytes))

print("Found Sheets:")
print(excel_file.sheet_names)

for sheet_name in excel_file.sheet_names:

    print(f"Processing Sheet: {sheet_name}")

    # Read sheet
    df = pd.read_excel(
        excel_file,
        sheet_name=sheet_name
    )

# ==========================================================
# STEP 5: Clean Sheet Name
# ==========================================================


    clean_name = (
        re.sub(
            r"[^\w\s-]",
            "",
            sheet_name
        )
        .strip()
        .replace(" ", "_")
    )

    if not clean_name:
        clean_name = "sheet"

    csv_file_name = f"{clean_name}.csv"

    
# ==========================================================
# STEP 6: Convert dataframe to CSV text
# ==========================================================

    csv_content = df.to_csv(index=False)

    s3.put_object(
        Bucket=output_bucket,
        Key=f"{output_prefix}{csv_file_name}",
        Body=csv_content.encode("utf-8")
    )


# ==========================================================
# STEP 7: Convert dataframe to CSV text
# ==========================================================
    s3.put_object(
        Bucket=output_bucket,
        Key=f"{output_prefix}{csv_file_name}",
        Body=csv_content
    )

    print(
        f"Saved: s3://{output_bucket}/{output_prefix}{csv_file_name}"
    )




#===========================================================
job.commit()


