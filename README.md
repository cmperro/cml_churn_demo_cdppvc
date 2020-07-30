# Churn Prediction Project Prototype for Private Cloud
The goal of this project is to use pieces of the CML Churn Demo on CDP Private Cloud.


## 0 Bootstrap and Ingest Data
Open the file `0_bootstrap_and_ingest.py` in a normal workbench python3 session. You only need a 
1 CPU / 2 GB instance. Then **Run > Run All Lines**

This script will read in the data csv from the file uploaded to the s3 bucket setup 
during the bootstrap and create a managed table in Hive. This is all done using Spark.

## 2 Explore Data
This is a Jupyter Notebook that does some basic data exploration and visualistaion. It 
is to show how this would be part of the data science workflow.

![data](https://raw.githubusercontent.com/fletchjeff/cml_churn_demo_mlops/master/images/data.png)

Open a Jupyter Notebook session (rather than a work bench): python3, 1 CPU, 2 GB and 
open the `2_data_exploration.ipynb` file. 

At the top of the page click **Cells > Run All**.

## 3 Model Building
This is also a Jupyter Notebook to show the process of selecting and building the model 
to predict churn. It also shows more details on how the LIME model is created and a bit 
more on what LIME is actually doing.

Open a Jupyter Notebook session (rather than a work bench): python3, 1 CPU, 2 GB and 
open the `	3_model_building.ipynb` file. 

At the top of the page click **Cells > Run All**.
