'''
this module implements unit tests for churn_library.py
'''

# pylint: disable=import-error
import os
import logging
import churn_library as cls

logging.basicConfig(
    filename='./logs/churn_library.log',
    level=logging.INFO,
    filemode='w',
    format='%(name)s - %(levelname)s - %(message)s')


def test_import(import_data):
    '''
    test data import - this example is completed for you to assist with the other test functions
    '''
    try:
        df = import_data("./data/bank_data.csv")
        logging.info("SUCCESS: Testing import_data")
    except FileNotFoundError as err:
        logging.error("Testing import_eda: The file wasn't found")
        raise err

    try:
        assert df.shape[0] > 0
        assert df.shape[1] > 0
    except AssertionError as err:
        logging.error(
            "Testing import_data: The file doesn't appear to have rows and columns")
        raise err


def test_eda(perform_eda):
    '''
    test perform eda function
    '''
    df = cls.import_data("./data/bank_data.csv")

    try:
        perform_eda(df)
        logging.info("SUCCESS: perform_eda")
    except KeyError as err:
        logging.error("ERROR perform_eda: column '%s' not found", err.args[0])
        raise err


def test_encoder_helper(encoder_helper):
    '''
    test encoder helper
    '''
    df = cls.import_data("./data/bank_data.csv")

    # Define churn feature
    df['Churn'] = df['Attrition_Flag'].apply(
        lambda val: 0 if val == "Existing Customer" else 1)

    cat_columns = ['Gender', 'Education_Level', 'Marital_Status',
                   'Income_Category', 'Card_Category']

    df_encoded_len = 0

    try:
        df_encoded = encoder_helper(
            df=df,
            category_lst=cat_columns,
            response='Churn')
        # Data should have changed
        assert df_encoded.equals(df) is False
        logging.info(
            "SUCCESS encoder_helper: dataset has chaged by encoder helper")
        df_encoded_len = df_encoded.shape[1]

    except AssertionError as err:
        logging.error(
            "ERROR encoder_helper: dataset should not be the \
				same after encoding")
        raise err

    try:
        # Check if the new columns were created
        assert df_encoded_len == df.shape[1] + len(cat_columns)
        logging.info(
            "SUCCESS encoder_helper: %d new columns created",
            len(cat_columns))
    except AssertionError as err:
        logging.error(
            "ERROR encoder_helper: Could not create new columns \
                df colums=%d, df_encoded columns=%d", df.shape[1], df_encoded_len)
        raise err

    try:
        df_encoded = encoder_helper(
            df=df,
            category_lst=cat_columns)
        # Data should have changed
        assert df_encoded.columns.equals(df.columns) is True
        logging.info(
            "SUCCESS encoder_helper: dataset columns has not changed \
                for response=None")
        df_encoded_len = df_encoded.shape[1]

    except AssertionError as err:
        logging.error(
            "ERROR encoder_helper: dataset columns should not have \
                 changed for response=None")
        raise err


def test_perform_feature_engineering(perform_feature_engineering):
    '''
    test perform_feature_engineering
    '''

    df = cls.import_data("./data/bank_data.csv")

    # Define churn feature
    df['Churn'] = df['Attrition_Flag'].\
        apply(lambda val: 0 if val == "Existing Customer" else 1)

    cat_columns = ['Gender', 'Education_Level', 'Marital_Status',
                   'Income_Category', 'Card_Category']

    df_encoded = cls.encoder_helper(df, cat_columns, 'Churn')

    try:
        _, X_test, _, _ = perform_feature_engineering(df=df_encoded,
                                                      response='Churn')

        # X_test size should be >= 30%
        assert (X_test.shape[0] >= df.shape[0] * 0.3) is True
        logging.info(
            'SUCCESS perform_feature_engineering: test size corresponds to 30% of dataset')

    except AssertionError as err:
        logging.error(
            'ERROR perform_feature_engineering: test size should be 30% of dataset')
        raise err


def test_train_models(train_models):
    '''
    test train_models
    '''

    df = cls.import_data("./data/bank_data.csv")
    # Define churn feature
    df['Churn'] = df['Attrition_Flag'].\
        apply(lambda val: 0 if val == "Existing Customer" else 1)

    cat_columns = ['Gender', 'Education_Level', 'Marital_Status',
                   'Income_Category', 'Card_Category']

    df_encoded = cls.encoder_helper(df, cat_columns, 'Churn')

    X_train, X_test, y_train, y_test = cls.perform_feature_engineering(
        df=df_encoded,
        response='Churn')
    try:
        train_models(X_train, X_test, y_train, y_test)
        # Check if logistic model was saved
        assert os.path.isfile("./models/logistic_model.pkl") is True
        logging.info(
            'SUCCESS train_models: model %s saved',
            'logistic_model.pkl')

    except AssertionError as err:
        logging.error(
            'ERROR train_models: model %s not saved',
            'logistic_model.pkl')
        raise err

    try:
        # Check if random forest model was saved
        assert os.path.isfile("./models/rfc_model.pkl") is True
        logging.info('SUCCESS train_models: model %s saved', 'rfc_model.pkl')
    except AssertionError as err:
        logging.error(
            'ERROR train_models: model %s not saved',
            'rfc_model.pkl')
        raise err

    try:
        # Check roc_curve
        assert os.path.isfile("./images/results/roc_curve.png") is True
        logging.info('SUCCESS train_models: %s saved', 'roc_curve.png')
    except AssertionError as err:
        logging.error('ERROR train_models: %s  not saved', 'roc_curve.png')
        raise err


if __name__ == "__main__":
    test_import(cls.import_data)
    test_eda(cls.perform_eda)
    test_encoder_helper(cls.encoder_helper)
    test_perform_feature_engineering(cls.perform_feature_engineering)
    test_train_models(cls.train_models)
