# This serves as a template which will guide you through the implementation of this task.  It is advised
# to first read the whole template and get a sense of the overall structure of the code before trying to fill in any of the TODO gaps
# First, we import necessary libraries:
import numpy as np
import pandas as pd

from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import SimpleImputer, IterativeImputer

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, RBF, Matern, WhiteKernel

def data_loading():
    """
    This function loads the training and test data, preprocesses it, removes the NaN values and interpolates the missing 
    data using imputation

    Returns
    ----------
    X_train: matrix of floats, training input with features
    y_train: array of floats, training output with labels
    X_test: matrix of floats: dim = (100, ?), test input with features
    """
    train_df = pd.read_csv("train.csv")
    test_df = pd.read_csv("test.csv")

    # Perform data preprocessing
    train_df['season'] = train_df['season'].map({"spring":0, "summer":1, "autumn":2, "winter":3})
    test_df['season'] = test_df['season'].map({"spring":0, "summer":1, "autumn":2, "winter":3})
    # print("Training data - Shape:" + str(train_df.shape))
    # print(train_df.head(2))
    # print("\nTest data - " + str(test_df.shape))
    # print(test_df.head(2))

    # discard rows that lack of price_CHF
    clear_train_df = train_df.dropna(subset=['price_CHF'])
    X_train = clear_train_df.drop(['price_CHF'], axis=1)
    y_train = clear_train_df['price_CHF']
    # print(X_train)

    # Imputation and extract X_train, y_train and X_test
    X_data = pd.concat((X_train, test_df))
    # imp_X = SimpleImputer(missing_values=pd.NA, strategy='mean')
    imp_X = IterativeImputer(missing_values=np.nan)
    imp_X.fit(X_data)
    # X_imputed = imp_X.transform(X_data)
    # X_train = X_imputed[:-100, :]
    # X_test = X_imputed[-100:, :]
    X_train = imp_X.transform(X_train)
    X_test = imp_X.transform(test_df)
    # print(X_test.shape)

    assert (X_train.shape[1] == X_test.shape[1]) \
        and (X_train.shape[0] == y_train.shape[0]) \
        and (X_test.shape[0] == 100), "Invalid data shape"
    return X_train, y_train, X_test

def modeling_and_prediction(X_train, y_train, X_test):
    """
    This function defines the model, fits training data and then does the prediction with the test data 

    Parameters
    ----------
    X_train: matrix of floats, training input with 10 features
    y_train: array of floats, training output
    X_test: matrix of floats: dim = (100, ?), test input with 10 features

    Returns
    ----------
    y_test: array of floats: dim = (100,), predictions on test set
    """
    y_pred=np.zeros(X_test.shape[0])

    # Define the model and fit it using training data.
    kernel_1 = ConstantKernel(1.0) + RBF(1.0, length_scale_bounds="fixed")
    kernel_2 = ConstantKernel() * Matern() + WhiteKernel()
    GPR = GaussianProcessRegressor(kernel=kernel_2, random_state=0, n_restarts_optimizer=10)
    GPR.fit(X_train, y_train)
    # use test data to make predictions
    y_pred = GPR.predict(X_test)

    assert y_pred.shape == (100,), "Invalid data shape"
    return y_pred

# Main function. You don't have to change this
if __name__ == "__main__":
    # Data loading
    X_train, y_train, X_test = data_loading()
    print("Load and impute data finish!")

    # The function retrieving optimal LR parameters
    y_pred = modeling_and_prediction(X_train, y_train, X_test)
    # Save results in the required format
    dt = pd.DataFrame(y_pred) 
    dt.columns = ['price_CHF']
    dt.to_csv('results.csv', index=False)
    print("\nResults file successfully generated!")

