'''
File containing functions to load and preprocess data
and to train ML models to predict churn.

Author: Matheus Scramignon
'''
import os
import shap
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import plot_roc_curve, classification_report
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from source.constants import (DATA_PATH, IMAGES_PATH, MODELS_PATH, TARGET_COLUMN,
                             CAT_COLUMNS, KEEP_COLUMNS)

os.environ['QT_QPA_PLATFORM'] = 'offscreen'
sns.set()


def import_data(pth):
    '''
    returns dataframe for the csv found at pth

    input:
            pth: a path to the csv
    output:
            df: pandas dataframe
    '''
    return pd.read_csv(pth)


def perform_eda(df_churn):
    '''
    perform eda on df and save figures to images folder
    input:
            df_churn: pandas dataframe

    output:
            None
    '''
    # Save Churn Histogram plot
    fig = plt.figure(figsize=(20, 10))
    ax_hist = df_churn['Churn'].hist()
    ax_hist.set_title('Churn Histogram')
    fig.savefig(IMAGES_PATH / 'churn_histogram.png')

    # Save Customer Age Histogram plot
    fig = plt.figure(figsize=(20, 10))
    ax_hist = df_churn['Customer_Age'].hist()
    ax_hist.set_title('Customer Age Histogram')
    fig.savefig(IMAGES_PATH / 'customer_age.png')

    # Save Marital Status Bar plot
    fig = plt.figure(figsize=(20, 10))
    ax_bar = df_churn.Marital_Status.value_counts('normalize').plot(kind='bar')
    ax_bar.set_title('Marital Status Bar Plot (Normalized)')
    fig.savefig(IMAGES_PATH / 'marital_status.png')

    # Save Total Transactions Histogram plot
    fig = plt.figure(figsize=(20, 10))
    ax_hist = sns.histplot(df_churn['Total_Trans_Ct'], stat='density', kde=True)
    ax_hist.set_title('Total Transactions Histogram (KDE)')
    fig.savefig(IMAGES_PATH / 'total_trans.png')

    # Save Heat Map
    fig = plt.figure(figsize=(20, 10))
    ax_heat = sns.heatmap(df_churn.corr(), annot=False, cmap='Dark2_r', linewidths=2)
    ax_heat.set_title('Heat Map')
    fig.savefig(IMAGES_PATH / 'heat_map.png')


def encoder_helper(df_churn, category_lst, response='Churn'):
    '''
    helper function to turn each categorical column into a new column with
    propotion of churn for each category - associated with cell 15 from the notebook

    input:
            df_churn: pandas dataframe
            category_lst: list of columns that contain categorical features
            response: string of response name [optional argument that could
                      be used for naming variables or index y column]

    output:
            df_encoded: pandas dataframe with new encoded columns
    '''
    df_encoded = df_churn.copy()

    # Categorical encoded columns
    for col in category_lst:
        col_lst = []
        col_groups = df_encoded.groupby(col).mean()['Churn']

        for val in df_encoded[col]:
            col_lst.append(col_groups.loc[val])

        df_encoded[f'{col}_{response}'] = col_lst

    return df_encoded


def perform_feature_engineering(df_encoded, response='Churn'):
    '''
    input:
              df_encoded: pandas dataframe
              response: string of response name [optional argument that could
                        be used for naming variables or index y column]

    output:
              X_train: X training data
              X_test: X testing data
              y_train: y training data
              y_test: y testing data
    '''
    # Define X and y
    columns = [col for col in df_encoded.columns.tolist() if col != response]

    # train test split
    return train_test_split(
        df_encoded[columns], df_encoded[response],
        test_size=0.3, random_state=42
    )


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
    print('random forest results')
    print('test results')
    report_rf_test = classification_report(y_test, y_test_preds_rf)
    print(report_rf_test)

    plot_classification_report(
        report_rf_test,
        IMAGES_PATH,
        'report_rf_test.png')

    print('train results')
    report_rf_train = classification_report(y_train, y_train_preds_rf)
    print(report_rf_train)

    plot_classification_report(
        report_rf_train,
        IMAGES_PATH,
        'report_rf_train.png')

    print('logistic regression results')
    print('test results')
    report_lr_test = classification_report(y_test, y_test_preds_lr)
    print(report_lr_test)

    plot_classification_report(
        report_lr_test,
        IMAGES_PATH,
        'report_lr_test.png')

    print('train results')
    report_lr_train = classification_report(y_train, y_train_preds_lr)
    print(report_lr_train)

    plot_classification_report(
        report_lr_train,
        IMAGES_PATH,
        'report_lr_train.png')


def plot_classification_report(report, output_folder, filename):
    """
    Plots the classification report as an image and saves it to a specified folder.

    input:
        report (str): The classification report string generated by
                      sklearn.metrics.classification_report.
        output_folder (str): Path to the folder where the image will be saved.
        filename (str): Name of the saved image file (default: 'classification_report.png').

    output:
        None
    """
    # Parse the report string into a dictionary
    lines = report.split('\n')
    classes = []
    plot_data = []

    # for line in lines[2:-3]:
    for line in lines[2:]:
        tokens = line.split()
        data_line = []
        if len(tokens) < 2:
            continue
        classes.append(tokens[0])

        if classes[-1] == 'accuracy':
            tokens.insert(1, '0.0')
            tokens.insert(1, '0.0')

        data_line = [float(x) for x in tokens[1:] if x != 'avg']
        plot_data.append(data_line)

    # Extract metrics from the dictionary
    metrics = ["precision", "recall", "f1-score", "support"]
    data = np.array(plot_data)

    # Create a heatmap plot
    plt.figure(figsize=(10, len(classes) * 0.6))
    sns.heatmap(data[:,
                     :-1],
                annot=True,
                fmt=".2f",
                cmap="coolwarm",
                cbar=True,
                xticklabels=metrics[:-1],
                yticklabels=classes,
                linewidths=0.5)

    plt.title("Classification Report", fontsize=16)
    plt.xlabel("Metrics", fontsize=12)
    plt.ylabel("Classes", fontsize=12)

    # Ensure the output folder exists
    os.makedirs(output_folder, exist_ok=True)

    # Save the plot
    output_path = os.path.join(output_folder, filename)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    print(f"Classification report saved to {output_path}")


def roc_curves_plot(model_rf, model_lr, X_data, y_data):
    '''
    creates and stores the feature importances in pth
    input:
            model_rf: model object containing random forest model
            model_lr: model object containing logistic regression model
            X_data: pandas dataframe of X values
            y_data: array of y values
            outputs_pth: paths to store the figures

    output:
             None
    '''

    fig, ax_roc = plt.subplots()
    plot_roc_curve(model_rf, X_data, y_data, ax=ax_roc)
    fig.savefig(IMAGES_PATH / 'roc_random_forest.png')

    fig, ax_roc = plt.subplots()
    lrc_plot = plot_roc_curve(model_lr, X_data, y_data, ax=ax_roc)
    fig.savefig(IMAGES_PATH / 'roc_logistic_regression.png')

    fig, ax_roc = plt.subplots(figsize=(15, 8))
    plot_roc_curve(model_rf, X_data, y_data, ax=ax_roc, alpha=0.8)
    lrc_plot.plot(ax=ax_roc, alpha=0.8)
    fig.savefig(IMAGES_PATH / 'roc_curves.png')


def feature_importance_plot(model, X_data):
    '''
    creates and stores the feature importances in pth
    input:
            model: model object containing feature_importances_
            X_data: pandas dataframe of X values

    output:
             None
    '''
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_data)

    fig = plt.figure()
    shap.summary_plot(shap_values, X_data, plot_type="bar")
    ax_shap = plt.gca()
    ax_shap.set_title('Features Importance')
    fig.savefig(IMAGES_PATH / 'features_importance.png')


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
    # Train models
    rfc = RandomForestClassifier(random_state=42)
    lrc = LogisticRegression(solver='lbfgs', max_iter=3000)

    param_grid = {
        'n_estimators': [200, 500],
        'max_features': ['auto', 'sqrt'],
        'max_depth': [4, 5, 100],
        'criterion': ['gini', 'entropy']
    }

    cv_rfc = GridSearchCV(estimator=rfc, param_grid=param_grid, cv=5)
    cv_rfc.fit(X_train, y_train)

    lrc.fit(X_train, y_train)

    y_train_preds_rf = cv_rfc.best_estimator_.predict(X_train)
    y_test_preds_rf = cv_rfc.best_estimator_.predict(X_test)

    y_train_preds_lr = lrc.predict(X_train)
    y_test_preds_lr = lrc.predict(X_test)

    # Generate classification reports images
    classification_report_image(
        y_train,
        y_test,
        y_train_preds_lr,
        y_train_preds_rf,
        y_test_preds_lr,
        y_test_preds_rf)

    # Generate features importance plot
    feature_importance_plot(cv_rfc.best_estimator_, X_test)

    # Generate ROC curves
    roc_curves_plot(cv_rfc.best_estimator_, lrc, X_test, y_test)

    # Save models
    print('Saving models')
    joblib.dump(cv_rfc, MODELS_PATH / 'rfc_model.pkl')
    joblib.dump(lrc, MODELS_PATH / 'logistic_model.pkl')
    print('Success')


if __name__ == "__main__":
    # Import Dataframe
    df = import_data((DATA_PATH / 'bank_data.csv').as_posix())

    # Create 'Churn' column
    df[TARGET_COLUMN] = df['Attrition_Flag'].apply(
        lambda val: 0 if val == "Existing Customer" else 1)

    # Perform EDA
    perform_eda(df)

    # Encode Categorical Funcions
    df = encoder_helper(df, CAT_COLUMNS, TARGET_COLUMN)

    # Perform feature engineering
    X_train_, X_test_, y_train_, y_test_ = perform_feature_engineering(
        df[KEEP_COLUMNS + [TARGET_COLUMN]], TARGET_COLUMN)

    # Train and save models related files
    train_models(X_train_, X_test_, y_train_, y_test_)
