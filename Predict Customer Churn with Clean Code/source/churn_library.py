# library doc string


# import libraries
import os
os.environ['QT_QPA_PLATFORM']='offscreen'

import shap
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns; sns.set()

from sklearn.preprocessing import normalize
from sklearn.model_selection import train_test_split

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV

from sklearn.metrics import plot_roc_curve, classification_report
from source.constants import *

def import_data(pth):
    '''
    returns dataframe for the csv found at pth

    input:
            pth: a path to the csv
    output:
            df: pandas dataframe
    '''	
    df = pd.read_csv(pth)

    return df


def perform_eda(df):
    '''
    perform eda on df and save figures to images folder
    input:
            df: pandas dataframe

    output:
            None
    '''
    # Save Churn Histogram plot
    fig = plt.figure(figsize=(20,10)) 
    ax = df['Churn'].hist();
    ax.set_title('Churn Histogram')
    fig.savefig(IMAGES_PATH / 'churn_histogram.png')

    # Save Customer Age Histogram plot
    fig = plt.figure(figsize=(20,10)) 
    ax = df['Customer_Age'].hist();
    ax.set_title('Customer Age Histogram')
    fig.savefig(IMAGES_PATH / 'customer_age.png')

    # Save Marital Status Bar plot
    fig = plt.figure(figsize=(20,10)) 
    ax = df.Marital_Status.value_counts('normalize').plot(kind='bar')
    ax.set_title('Marital Status Bar Plot (Normalized)')
    fig.savefig(IMAGES_PATH / 'marital_status.png')

    # Save Total Transactions Histogram plot
    fig = plt.figure(figsize=(20,10)) 
    ax = sns.histplot(df['Total_Trans_Ct'], stat='density', kde=True)
    ax.set_title('Total Transactions Histogram (KDE)')
    fig.savefig(IMAGES_PATH / 'total_trans.png')

    # Save Heat Map
    fig = plt.figure(figsize=(20,10)) 
    ax = sns.heatmap(df.corr(), annot=False, cmap='Dark2_r', linewidths = 2)
    ax.set_title('Heat Map')
    fig.savefig(IMAGES_PATH / 'heat_map.png')


def encoder_helper(df, category_lst, response='Churn'):
    '''
    helper function to turn each categorical column into a new column with
    propotion of churn for each category - associated with cell 15 from the notebook

    input:
            df: pandas dataframe
            category_lst: list of columns that contain categorical features
            response: string of response name [optional argument that could be used for naming variables or index y column]

    output:
            df: pandas dataframe with new columns for
    '''
    # Categorical encoded columns
    for col in category_lst:
        col_lst = []
        col_groups = df.groupby(col).mean()['Churn']

        for val in df[col]:
            col_lst.append(col_groups.loc[val])

        df[f'{col}_{response}'] = col_lst

    return df


def perform_feature_engineering(df, response):
    '''
    input:
              df: pandas dataframe
              response: string of response name [optional argument that could be used for naming variables or index y column]

    output:
              X_train: X training data
              X_test: X testing data
              y_train: y training data
              y_test: y testing data
    '''

def classification_report_image(y_train,
                                y_test,
                                y_train_preds_lr,
                                y_train_preds_rf,
                                y_test_preds_lr,
                                y_test_preds_rf):
    '''
    produces classification report for training and testing results and stores report as image
    in images folder
    input:
            y_train: training response values
            y_test:  test response values
            y_train_preds_lr: training predictions from logistic regression
            y_train_preds_rf: training predictions from random forest
            y_test_preds_lr: test predictions from logistic regression
            y_test_preds_rf: test predictions from random forest

    output:
             None
    '''
    pass


def feature_importance_plot(model, X_data, output_pth):
    '''
    creates and stores the feature importances in pth
    input:
            model: model object containing feature_importances_
            X_data: pandas dataframe of X values
            output_pth: path to store the figure

    output:
             None
    '''
    pass

def train_models(X_train, X_test, y_train, y_test):
    '''
    train, store model results: images + scores, and store models
    input:
              X_train: X training data
              X_test: X testing data
              y_train: y training data
              y_test: y testing data
    output:
              None
    '''
    pass


if __name__ == "__main__":
    # Import Dataframe
    df = import_data((DATA_PATH / 'bank_data.csv').as_posix())

    # Create 'Churn' column
    df[TARGET_COLUMN] = df['Attrition_Flag'].apply(lambda val: 0 if val == "Existing Customer" else 1)

    # Perform EDA
    perform_eda(df)

    # Encode Categorical Funcions
    df = encoder_helper(df, CAT_COLUMNS, TARGET_COLUMN)

    # Define X and y
    X, y = df[KEEP_COLUMNS], df[TARGET_COLUMN]

    # train test split 
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size= 0.3, random_state=42)