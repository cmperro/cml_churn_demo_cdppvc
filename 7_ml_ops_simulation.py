## Model Operations
# This script show cases how to use the model operations features of CML

import cdsw, time, os, random, json
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import chisquare
import requests
from sklearn.metrics import classification_report
from cmlbootstrap import CMLBootstrap
import seaborn as sns
import copy


## Set the model ID
# Get the model id from the model you deployed in step 5. These are unique to each 
# model on CML.

model_id = "88"

# Grab the data from Hive.
from pyspark.sql import SparkSession
from pyspark.sql.types import *
spark = SparkSession\
    .builder\
    .appName("PythonSQL")\
    .master("local[*]")\
    .getOrCreate()

df = spark.sql("SELECT * FROM default.telco_churn").toPandas()

# Get the various Model CRN details
HOST = os.getenv("CDSW_API_URL").split(
    ":")[0] + "://" + os.getenv("CDSW_DOMAIN")
USERNAME = os.getenv("CDSW_PROJECT_URL").split(
    "/")[6]  # args.username  # "vdibia"
API_KEY = os.getenv("CDSW_API_KEY") 
PROJECT_NAME = os.getenv("CDSW_PROJECT")  

cml = CMLBootstrap(HOST, USERNAME, API_KEY, PROJECT_NAME)

latest_model = cml.get_model({"id": model_id, "latestModelDeployment": True, "latestModelBuild": True})

Model_CRN = latest_model ["crn"]
Deployment_CRN = latest_model["latestModelDeployment"]["crn"]
model_endpoint = HOST.split("//")[0] + "//modelservice." + HOST.split("//")[1] + "/model"

## Generate Sample Data
# This section will grab 1000 random samples from the data set and simulate 1000 predictions 
# The live model will be called each time in the loop and while the `churn_error` function
# adds an increasing amount of error to the data to make the model less accurate.

  
def churn_error(item,percent):
  if random.random() < percent:
    return True
  else:
    return True if item=='Yes' else False

  
# Get 1000 samples  
df_sample = df.sample(1000)

df_sample.groupby('Churn')['Churn'].count() 

df_sample_clean = df_sample.\
  replace({'SeniorCitizen': {"1": 'Yes', "0": 'No'}}).\
  replace(r'^\s$', np.nan, regex=True).\
  dropna()

# Create an array of model responses.
response_labels_sample = []

percent_counter = 0
percent_max = len(df_sample_clean)

# make 1000 calls to the model with increasing error
for record in json.loads(df_sample_clean.to_json(orient='records')):
  print("Added {} records".format(percent_counter)) if (percent_counter%50 == 0) else None
  percent_counter += 1
  no_churn_record = copy.deepcopy(record)
  no_churn_record.pop('customerID')
  no_churn_record.pop('Churn')
  response = cdsw.call_model(latest_model["accessKey"],no_churn_record)
  
  response_labels_sample.append(
    {
      "uuid":response["response"]["uuid"],
      "final_label":churn_error(record["Churn"],percent_counter/percent_max),
      "response_label":response["response"]["prediction"]["probability"] >= 0.5,
      "timestamp_ms":int(round(time.time() * 1000))
    }
  )

#New loop with ground truth
  
for index, vals in enumerate(response_labels_sample):
  print("Update {} records".format(index)) if (index%50 == 0) else None  
  cdsw.track_delayed_metrics({"final_label":vals['final_label']}, vals['uuid'])
  if (index%100 == 0):
    start_timestamp_ms = vals['timestamp_ms']
    final_labels = []
    response_labels = []
  final_labels.append(vals['final_label'])
  response_labels.append(vals['response_label'])
  if (index%100 == 99):
    print("Adding accuracy metrc")
    end_timestamp_ms = vals['timestamp_ms']
    accuracy = classification_report(final_labels,response_labels,output_dict=True)["accuracy"]
    cdsw.track_aggregate_metrics({"accuracy": accuracy}, start_timestamp_ms , end_timestamp_ms, model_deployment_crn=Deployment_CRN)


