## Plot some Model Metrics
# Here are some plots from the model metrics.  

import cdsw, time, os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import classification_report
from cmlbootstrap import CMLBootstrap
import seaborn as sns


## Set the model ID
# Get the model id from the model you deployed in step 5. These are unique to each 
# model on CML.

model_id = "83"

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

# Read in the model metrics dict.
model_metrics = cdsw.read_metrics(model_crn=Model_CRN,model_deployment_crn=Deployment_CRN)

# This is a handy way to unravel the dict into a big pandas dataframe.
metrics_df = pd.io.json.json_normalize(model_metrics["metrics"]) # [metric_start_index:])
metrics_df.tail().T

# Do some conversions & calculations
metrics_df['startTimeStampMs'] = pd.to_datetime(metrics_df['startTimeStampMs'], unit='ms')
metrics_df['endTimeStampMs'] = pd.to_datetime(metrics_df['endTimeStampMs'], unit='ms')
metrics_df["processing_time"] = (metrics_df["endTimeStampMs"] - metrics_df["startTimeStampMs"]).dt.microseconds * 1000

# This shows how to plot specific metrics.
sns.set_style("whitegrid")
sns.despine(left=True,bottom=True)

prob_metrics = metrics_df.dropna(subset=['metrics.probability']).sort_values('startTimeStampMs')
sns.lineplot(x=range(len(prob_metrics)), y="metrics.probability", data=prob_metrics, color='grey')

time_metrics = metrics_df.dropna(subset=['processing_time']).sort_values('startTimeStampMs')
sns.lineplot(x=range(len(prob_metrics)), y="processing_time", data=prob_metrics, color='grey')

# This shows how the model accuracy drops over time.
agg_metrics = metrics_df.dropna(subset=["metrics.accuracy"]).sort_values('startTimeStampMs')
sns.barplot(x=list(range(1,len(agg_metrics)+1)), y="metrics.accuracy", color="grey", data=agg_metrics)
