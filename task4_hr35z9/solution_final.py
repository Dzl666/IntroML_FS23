import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
from matplotlib import pyplot as plt

from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge, RidgeCV

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def load_data():
    """ This function loads the data from the csv files and returns it as numpy arrays.
    output: x_pretrain: np.ndarray, the features of the pretraining set
            y_pretrain: np.ndarray, the labels of the pretraining set
            x_train: np.ndarray, the features of the training set
            y_train: np.ndarray, the labels of the training set
            x_test: np.ndarray, the features of the test set
    """
    x_pretrain = pd.read_csv("public/pretrain_features.csv.zip", index_col="Id", compression='zip').drop("smiles", axis=1).to_numpy()
    y_pretrain = pd.read_csv("public/pretrain_labels.csv.zip", index_col="Id", compression='zip').to_numpy().squeeze(-1)
    x_train = pd.read_csv("public/train_features.csv.zip", index_col="Id", compression='zip').drop("smiles", axis=1).to_numpy()
    y_train = pd.read_csv("public/train_labels.csv.zip", index_col="Id", compression='zip').to_numpy().squeeze(-1)
    x_test = pd.read_csv("public/test_features.csv.zip", index_col="Id", compression='zip').drop("smiles", axis=1)
    return x_pretrain, y_pretrain, x_train, y_train, x_test

class Net(nn.Module):
    def __init__(self, input_dim, latent_dim):
        """ The constructor of the model."""
        super().__init__()
        # TODO: It should be able to be trained on pretraing data 
        # and then used to extract features from the training and test data.
        # 1000 -> 512 -> 256
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU(),
            nn.Linear(512, latent_dim),
            nn.Sigmoid()
        )
        # 256 -> 1
        self.predictor = nn.Sequential(
            nn.Linear(latent_dim, 1)
        )

    def forward(self, x):
        z = self.encoder(x)
        output = self.predictor(z)
        return output
    
def make_feature_extractor(x, y, batch_size=512, eval_size=1000):
    """ This function trains the feature extractor on the pretraining data and returns a function which
    can be used to extract features from the training and test data.

    input: x: np.ndarray, the features of the pretraining set
              y: np.ndarray, the labels of the pretraining set
                batch_size: int, the batch size used for training
                eval_size: int, the size of the validation set
            
    output: make_features: function, a function which can be used to extract features from the training and test data
    """
    # split the pre-train data
    x_tr, x_val, y_tr, y_val = train_test_split(x, y, test_size=eval_size, random_state=0, shuffle=True)
    
    x_tr = torch.from_numpy(x_tr).type(torch.float)
    y_tr = torch.from_numpy(y_tr).type(torch.float)
    x_val = torch.from_numpy(x_val).type(torch.float)
    y_val = torch.from_numpy(y_val).type(torch.float)

    # TODO: The model should be trained on the pretraining data.
    train_dataset = TensorDataset(x_tr, y_tr)
    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size,
        shuffle=True, pin_memory=True, num_workers=1)

    # model declaration
    in_features_dim = x.shape[-1]
    model = Net(input_dim=in_features_dim, latent_dim=256).to(device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=1.5e-4)
    loss_func = torch.nn.MSELoss()

    # unsupervised training the autoencoder
    model.train()
    Epochs = 40
    pbar = tqdm(range(Epochs))
    for _ in pbar:
        train_loss = 0
        for [x_batch, y_batch] in train_loader:
            output = model.forward(x_batch.to(device))
            optimizer.zero_grad()
            loss = loss_func(output.squeeze(), y_batch.to(device))
            train_loss += loss.item() * len(x_batch)/len(x_tr)
            loss.backward()
            optimizer.step()

        with torch.no_grad():
            output = model.forward(x_val.to(device))
            val_loss = torch.sqrt(loss_func(output.squeeze(), y_val.to(device))).item()

        pbar.set_postfix(TrainLoss=f"{train_loss:.8f}", ValLoss=f"{val_loss:.8f}")

    def make_features(x):
        """ Extracts features from the training and test data, used in the pipeline after the pretraining.
        input:
            x: np.ndarray, the features of the training or test set
        output:
            features: np.ndarray, the features extracted from the training or test set
        """
        model.eval()
        # TODO: Implement the feature extraction, a part of a pretrained model used later in the pipeline.
        x = torch.from_numpy(x).type(torch.float)
        z = model.encoder(x.to(device)).squeeze()
        feature = z.detach().cpu().numpy()
        return feature

    return make_features

def make_pretraining_class(feature_extractors):
    """ The wrapper function which makes pretraining API compatible with sklearn pipeline
    input:
        feature_extractors: dict, a dictionary of feature extractors
    output:
        PretrainedFeatures: class, a class which implements sklearn API
    """

    class PretrainedFeatures(BaseEstimator, TransformerMixin):
        """
        The wrapper class for Pretraining pipeline.
        """
        def __init__(self, *, feature_extractor=None, mode=None):
            self.feature_extractor = feature_extractors[feature_extractor]
            self.mode = mode

        # already have the pre-train feature extracor, doesn't need fit
        def fit(self, X=None, y=None):
            return self
        
        # only needs to transform the training data to get feature
        def transform(self, X):
            assert self.feature_extractor is not None
            # the same as the make_features(x)
            X_new = self.feature_extractor(X)
            return X_new
        
    return PretrainedFeatures

def get_regression_model():
    """ This function returns the regression model used in the pipeline.
    output:
        model: sklearn compatible model, the regression model
    """
    # TODO: Implement the regression model. It should be able to be trained on the features extracted
    # by the feature extractor.
    # RidgeCV(alphas=[0.5, 1, 5, 10, 20, 50, 100], cv=10)
    model = Ridge(alpha=5.)
    return model

# Main function. You don't have to change this
if __name__ == '__main__':
    # np.random.seed(1)
    # torch.manual_seed(1)
    # Load data
    x_pretrain, y_pretrain, x_train, y_train, x_test = load_data()
    print("Data loaded!")
    # Utilize pretraining data by creating feature extractor 
    # which extracts lumo energy features from available initial features
    print("========== Start pre-training ==========")
    feature_extractor = make_feature_extractor(x_pretrain, y_pretrain)

    # pass many kinds of feature_extractors
    PretrainedFeatureClass = make_pretraining_class({"pretrain": feature_extractor})
    
    # TODO: Implement the pipeline. It should contain feature extraction and regression. You can optionally
    # use other sklearn tools, such as StandardScaler, FunctionTransformer, etc.
    pipeline = Pipeline([
        ("extractor", PretrainedFeatureClass(feature_extractor="pretrain")),
        ("scaler", StandardScaler()), # Regularization
        ("regresser", get_regression_model()), # regression model
    ])

    print("========== Start to transfer and predict ==========")
    # execute transform of the feature_extractor, then fit the regressor
    pipeline.fit(x_train, y_train)
    # do transform and then predict
    y_pred = pipeline.predict(x_test.to_numpy())

    # Final checking and save the result
    assert y_pred.shape == (x_test.shape[0],)
    y_pred = pd.DataFrame({"y": y_pred}, index=x_test.index)
    y_pred.to_csv("results_1.csv", index_label="Id")
    print("Predictions saved, all done!")