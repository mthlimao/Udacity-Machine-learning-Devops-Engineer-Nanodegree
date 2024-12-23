# Predict Customer Churn

- Project **Predict Customer Churn** of ML DevOps Engineer Nanodegree Udacity

## Project Description
Your project description here.
Project has information on customers in a certain bank, with the goal to develop a model to predict teh customers that churned, in order for the bank to be able to put measures in place to keep these and potential customers from churning in the future. This project aims 
to implement routines using clean code best practices to improve the code to make it into a production ready code, by refactoring, logging and testing.

## Files and data description
### data - contains the data used for the analysis and training
### models - contains the two models used for the training, random forest and logistic regression
### images - contains plots of several aspects of the data and the models' performance
### logs - contains information of every aspect of the code to track what is going on

## Running Files
### There are two python files, source/churn_library.py and source/churn_script_logging_and_tests.py
### The source/churn_library.py contains the actual code for data ingestion, pre-processing, feature engineering, training and validation 
### The source/churn_script_logging_and_tests.py contains code to test and log the output.

To run the *python source/churn_library.py* command, one must first run the requirements.txt file to install the libraires by using pip install -r requirements.txt and add the root directory of this repo to PYTHONPATH.

To test the implemented routines, one can run *python source/churn_script_logging_and_tests.py*

### You also can use Docker to run a containerized version of this project, using the *docker compose up* command



