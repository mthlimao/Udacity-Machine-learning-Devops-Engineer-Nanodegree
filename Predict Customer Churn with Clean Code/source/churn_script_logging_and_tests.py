'''
File containing tests for churn_library.py functions.

Author: Matheus Scramignon
'''

import os
import logging
import pytest
from source.churn_library import (import_data, perform_eda, encoder_helper,
                                  perform_feature_engineering, train_models)
from source.constants import (
    DATA_PATH,
    IMAGES_PATH,
    MODELS_PATH,
    LOGS_PATH,
    TARGET_COLUMN,
    CAT_COLUMNS,
    KEEP_COLUMNS)

logging.basicConfig(
    filename=(LOGS_PATH / 'churn_library.log').as_posix(),
    level=logging.INFO,
    filemode='w',
    format='%(name)s - %(levelname)s - %(message)s')

logger = logging.getLogger()


@pytest.fixture()
def df_import():
    '''
    fixture defining initial imported dataframe.

    input:
        None
    output:
        df_import: pandas dataframe
    '''
    try:
        df_import = import_data((DATA_PATH / 'bank_data.csv').as_posix())
        logging.info("Testing import_data: SUCCESS")
    except FileNotFoundError as err:
        logging.error("Testing import_eda: The file wasn't found")
        raise err

    return df_import


@pytest.fixture()
def df_churn(df_import):
    '''
    fixture defining dataframe with 'Churn' column.

    input:
        df_import: pandas dataframe
    output:
        df_churn: pandas dataframe
    '''
    # Create 'Churn' column
    df_churn = df_import.copy()
    df_churn[TARGET_COLUMN] = df_churn['Attrition_Flag'].apply(
        lambda val: 0 if val == "Existing Customer" else 1)

    return df_churn


@pytest.fixture()
def df_encoded(df_churn):
    '''
    fixture defining encoded dataframe.

    input:
        df_churn: pandas dataframe
    output:
        df_encoded: pandas dataframe
    '''
    df_encoded = df_churn.copy()

    # Call the encoder_helper function
    df_encoded = encoder_helper(df_churn, CAT_COLUMNS, TARGET_COLUMN)

    return df_encoded


@pytest.fixture()
def df_featurized(df_encoded):
    '''
    fixture defining featurized dataframe.

    input:
        df_encoded: pandas dataframe
    output:
        df_featurized: pandas dataframe
    '''
    df_featurized = df_encoded.copy()
    df_featurized = df_featurized[KEEP_COLUMNS + [TARGET_COLUMN]]

    return df_featurized


def test_import(df_import):
    '''
    test data import - this example is completed for you to assist with the other test functions
    '''
    try:
        assert df_import.shape[0] > 0
        assert df_import.shape[1] > 0
    except AssertionError as err:
        logging.error(
            "Testing import_data: The file doesn't appear to have rows and columns")
        raise err


def test_eda(df_churn):
    '''
    test perform eda function
    '''
    # Ensure the images folder exists
    assert IMAGES_PATH.exists()
    logging.info("Testing perform_eda: IMAGES_PATH exists")

    # Call the perform_eda function
    perform_eda(df_churn)

    # Check if the files are created
    expected_files = [
        'churn_histogram.png',
        'customer_age.png',
        'marital_status.png',
        'total_trans.png',
        'heat_map.png'
    ]

    try:
        for file in expected_files:
            assert (IMAGES_PATH / file).exists()
    except AssertionError as err:
        logging.error("Testing perform_eda: File %s was not created.", file)
        raise err


def test_encoder_helper(df_encoded):
    '''
    test encoder helper
    '''
    category_lst = CAT_COLUMNS
    response = TARGET_COLUMN

    # Check if the new columns are created
    try:
        for col in category_lst:
            new_col = f'{col}_{response}'
            assert new_col in df_encoded.columns
    except AssertionError as err:
        logging.error("Testing encoder_helper: Column %s was not created.", new_col)
        raise err

    logging.info("Testing encoder_helper: New columns successfully created.")

    # Verify the new column values
    try:
        for col in category_lst:
            new_col = f'{col}_{response}'
            assert not df_encoded[new_col].isnull().any()
    except AssertionError as err:
        logging.error("Testing encoder_helper: Column %s contains null values.", new_col)
        raise err

    logging.info(
        "Testing encoder_helper: New columns created with sound values.")


def test_perform_feature_engineering(df_featurized):
    '''
    test perform_feature_engineering
    '''
    # Call the perform_feature_engineering function
    X_train, X_test, y_train, y_test = perform_feature_engineering(
        df_featurized, response=TARGET_COLUMN)

    # Check if the splits are correct
    try:
        assert len(X_train) > 0
        assert len(X_test) > 0
        assert len(y_train) > 0
        assert len(y_test) > 0
    except AssertionError as err:
        logging.error(
            "Testing perform_feature_engineering: One of the returned sets is empty")
        raise err

    logging.info(
        "Testing perform_feature_engineering: Returned sets are not empty")

    # Check if the shapes match
    try:
        assert X_train.shape[1] == X_test.shape[1]
    except AssertionError as err:
        logging.error(
            "Testing perform_feature_engineering: Mismatch \
            in number of features between X_train and X_test.")
        raise err

    logging.info(
        "Testing perform_feature_engineering: Sets successfully returned by function.")


def test_train_models(df_featurized):
    '''
    test train_models
    '''
    X_train, X_test, y_train, y_test = perform_feature_engineering(
        df_featurized, response=TARGET_COLUMN)

    # Ensure the models folder exists
    assert MODELS_PATH.exists()
    logging.info("Testing train_models: MODELS_PATH exists.")

    # Call the train_models function
    train_models(X_train, X_test, y_train, y_test)

    # Check if the images files are created
    expected_images = [
        'report_rf_test.png',
        'report_rf_train.png',
        'report_lr_test.png',
        'report_lr_train.png',
        'roc_random_forest.png',
        'roc_logistic_regression.png',
        'roc_curves.png',
        'features_importance.png',
    ]

    try:
        for file in expected_images:
            assert (IMAGES_PATH / file).exists()
    except AssertionError as err:
        logging.error("Testing train_models: File %s was not created.", file)
        raise err

    logging.info("Testing train_models: Images files successfully created.")

    # Check if models are saved
    try:
        assert (MODELS_PATH / 'rfc_model.pkl').exists()
        assert (MODELS_PATH / 'logistic_model.pkl').exists()
    except AssertionError as err:
        logging.error("Testing train_models: One or more models were not created.")
        raise err

    logging.info("Testing train_models: Models successfully created.")


if __name__ == "__main__":
    logger.info('About to start the tests')
    pytest.main(args=['-s', os.path.abspath(__file__)])
    logger.info('Done executing the tests')
