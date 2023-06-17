# This serves as a template which will guide you through the implementation of this task.  It is advised
# to first read the whole template and get a sense of the overall structure of the code before trying to fill in any of the TODO gaps
# First, we import necessary libraries:
import numpy as np

import os
import torch
from torch.utils.data import DataLoader, TensorDataset, random_split
from torchvision import transforms, models
import torchvision.datasets as datasets
import torch.nn as nn
import torch.nn.functional as F
from torchmetrics.classification import BinaryAccuracy
from tqdm import tqdm
from matplotlib import pyplot as plt

from sklearn.model_selection import KFold

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def generate_embeddings():
    """ Transform, resize and normalize the images and then use a pretrained model to extract 
    the embeddings.
    """
    # TODO: define a transform to pre-process the images
    # transforms.Normalize(mean=[0.608, 0.515, 0.411], std=[0.223, 0.239, 0.257])
    # mean=[0.611, 0.501, 0.375], std=[0.216, 0.230, 0.239]
    train_transforms = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    train_dataset = datasets.ImageFolder(root="dataset/", transform=train_transforms)
    train_loader = DataLoader(dataset=train_dataset, batch_size=32, shuffle=False,
                              pin_memory=True, num_workers=8)

    # TODO: define a model for extraction of the embeddings
    model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
    # model = models.efficientnet_b7(weights=models.EfficientNet_B7_Weights.IMAGENET1K_V1)
    # TODO: Use the model to extract the embeddings. 
    # Remove the last layers to access the embeddings the model generates. 
    model = torch.nn.Sequential(*list(model.children())[:-1]).to(device)
    
    embeddings_list = []
    model.eval()
    with torch.no_grad():
        for images, _ in train_loader:
            embed = model(images.to(device))
            embeddings_list.append(embed.cpu().numpy())

        embeddings = np.concatenate(embeddings_list, axis=0).squeeze()
        print("Embed shape", embeddings.shape) #[10000, 2048]
        np.save('dataset/embeddings.npy', embeddings)


def get_data(file, train=True):
    """ Load the triplets from the file and generate the features and labels.

    input: file: string, the path to the file containing the triplets
          train: boolean, whether the data is for training or testing

    output: X: numpy array, the features
            y: numpy array, the labels
    """
    triplets = []
    with open(file) as f:
        for line in f:
            triplets.append(line)

    # generate training data from triplets
    train_dataset = datasets.ImageFolder(root="dataset/", transform=None)
    filenames = [s[0].split("\\")[-1].replace('.jpg', '') for s in train_dataset.samples]
    # print(filenames)
    embeddings = np.load('dataset/embeddings.npy')
    # TODO: Normalize the embeddings across the dataset
    norms = np.linalg.norm(embeddings, axis=1)
    norm_embeddings = embeddings / np.expand_dims(norms, axis=-1)
    # norm_embeddings = (embeddings - np.mean(embeddings, 0)) / np.std(embeddings, 0)

    file_to_embedding = {}
    for i in range(len(filenames)):
        file_to_embedding[filenames[i]] = norm_embeddings[i, :]

    X = []
    y = []
    # use the individual embeddings to generate the features and labels for triplets
    for t in triplets:
        emb = [file_to_embedding[img_id] for img_id in t.split()]
        X.append(np.hstack([emb[0], emb[1], emb[2]]))
        y.append(1.)
        # Generating negative samples (data augmentation)
        if train:
            X.append(np.hstack([emb[0], emb[2], emb[1]]))
            y.append(0.)
    X = np.vstack(X)
    y = np.hstack(y)
    print("Data loaded!")
    return X, y

# Hint: adjust batch_size and num_workers to your PC configuration, so that you don't run out of memory
def create_loader_from_np(X, y = None, train = True, batch_size=64, shuffle=True, num_workers = 6):
    """ Create a torch.utils.data.DataLoader object from numpy arrays containing the data.
    input:
        X: numpy array, the features  
        y: numpy array, the labels
    output:
        loader: torch.data.util.DataLoader, the object containing the data
    """
    if train:
        dataset = TensorDataset(torch.from_numpy(X).type(torch.float), 
                                torch.from_numpy(y).type(torch.long))
    else:
        dataset = TensorDataset(torch.from_numpy(X).type(torch.float))
    loader = DataLoader(dataset=dataset, batch_size=batch_size,
                        shuffle=shuffle, pin_memory=True, num_workers=num_workers)
    return loader

# TODO: define a model.
class Net(nn.Module):
    """ The model class, which defines our classifier.
    """
    def __init__(self, dropout_p = 0.5, LeakyRelu_k = 0.01):
        """ The constructor of the model.
        """
        super().__init__()
        self.network = torch.nn.Sequential(
            # 3*2048, 3072
            # 3*2560, 3840
            torch.nn.Linear(3*2048, 3072),
            torch.nn.BatchNorm1d(3072),
            torch.nn.LeakyReLU(LeakyRelu_k),
            torch.nn.Dropout(dropout_p),

            torch.nn.Linear(3072, 2048),
            torch.nn.BatchNorm1d(2048),
            torch.nn.LeakyReLU(LeakyRelu_k),
            torch.nn.Dropout(dropout_p),

            torch.nn.Linear(2048, 1024),
            torch.nn.BatchNorm1d(1024),
            torch.nn.LeakyReLU(LeakyRelu_k),
            torch.nn.Dropout(dropout_p),

            torch.nn.Linear(1024, 1)
        )

    def forward(self, x):
        """ The forward pass of the model.
        input: x: torch.Tensor, the input to the model
        output: x: torch.Tensor, the output of the model
        """
        x = self.network(x)
        x = F.sigmoid(x)
        return x

def train_model(X, y):
    """ The training procedure of the model
    input: 
        train_loader: torch.data.util.DataLoader, the object containing the training data
    output:
        model: torch.nn.Module, the trained model
    """
    train_dataset = create_loader_from_np(X, y, batch_size=1024, train=True)
    # Use the part of the training data as a validation split.
    split_ratio = 0.1
    len_val = (int)(len(train_dataset.dataset) * split_ratio)
    len_train = len(train_dataset.dataset) - len_val
    train_set, val_set = random_split(train_dataset.dataset, [len_train, len_val])
    # Create data loaders for the training and validating data
    train_loader = DataLoader(train_set, batch_size=2048, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=1024, shuffle=False)

    model = Net()
    model.to(device)
    # TODO: define a loss function, optimizer and proceed with training.
    loss_func = F.binary_cross_entropy
    metric_func = BinaryAccuracy().to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=2e-2, momentum=0.9)
    # optimizer = torch.optim.Adam(model.parameters(), lr=5e-3, weight_decay=5e-4)
    
    # This enables you to see how your model is performing on the validation data 
    # After choosing the best model, train it on the whole training data. ?
    best_val_loss = np.inf
    torch.save(model.state_dict(), "best_model.pth")
    training_loss = []
    validating_loss = []
    stop_thres = 10
    stop_cnt = 0
    for epoch in tqdm(range(60)):

        model.train()
        loss = 0
        for [X_batch, y_batch] in train_loader:
            # y=0 for A more like B, y=1 for A more like C
            z = model(X_batch.to(device)).squeeze()
            y_batch = y_batch.to(torch.float).to(device)
            optimizer.zero_grad()
            train_loss = loss_func(z, y_batch)
            loss += train_loss.item()
            train_loss.backward()
            optimizer.step()
        training_loss.append(loss / len_train)

        # Compute the loss on the validation split and print it out.
        model.eval()
        with torch.no_grad():
            val_loss = 0
            val_acc = 0
            for [X_batch, y_batch] in val_loader:
                z = model(X_batch.to(device)).squeeze()
                y_batch = y_batch.to(torch.float).to(device)
                val_loss += loss_func(z, y_batch).item()
                acc = metric_func(z, y_batch)
                val_acc += (int)(acc * X_batch.shape[0]) / len_val
            val_loss = val_loss / len_val
            validating_loss.append(val_loss)
            print(f"\n[VAL] Acc: {val_acc:.4f}, Loss: {val_loss:.8f}, Best loss: {best_val_loss:.8f}")
            if(val_loss < best_val_loss):
                stop_cnt = 0
                best_val_loss = val_loss
                torch.save(model.state_dict(), "best_model.pth")
                print("Update best model")
            else:
                stop_cnt += 1

            if(stop_cnt >= stop_thres):
                break

    plt.figure(figsize=(6,4))
    plt.plot(training_loss, "b-", label="Train loss")
    plt.plot(validating_loss, "g-", label="Val loss")
    plt.legend()
    plt.show()

    # final training
    model.load_state_dict(torch.load("best_model.pth"))
    model.to(device)
    model.train()
    for param in model.parameters():
        param.requires_grad = False
    for param in model.network[-1].parameters():
        param.requires_grad = True
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    for epoch in tqdm(range(5)):
        final_acc = 0
        for [X_batch, y_batch] in train_dataset:
            z = model(X_batch.to(device)).squeeze()
            y_batch = y_batch.to(torch.float).to(device)

            optimizer.zero_grad()
            train_loss = loss_func(z, y_batch)
            train_loss.backward()
            optimizer.step()

            acc = metric_func(z, y_batch)
            final_acc += (int)(acc * X_batch.shape[0]) / len(train_dataset.dataset)
        print(f"\n[Final TRAIN] Acc: {final_acc}")

    return model
    # torch.save(model.state_dict(), "best_model_final.pth")


def test_model(model, loader):
    """ The testing procedure of the model
    input: 
        model: torch.nn.Module, the trained model
        loader: torch.data.util.DataLoader, the object containing the testing data
    """
    # model.load_state_dict(torch.load("best_model.pth"))
    model.to(device)
    model.eval()
    predictions = []
    # Iterate over the test data
    with torch.no_grad(): # We don't need to compute gradients for testing
        for [x_batch] in loader:
            x_batch= x_batch.to(device)
            predicted = model(x_batch)
            predicted = predicted.cpu().numpy()
            # Rounding the predictions to 0 or 1
            predicted[predicted >= 0.5] = 1
            predicted[predicted < 0.5] = 0
            predictions.append(predicted)
        predictions = np.vstack(predictions)
    np.savetxt("results.txt", predictions, fmt='%i')


# Main function. You don't have to change this
if __name__ == '__main__':
    TRAIN_TRIPLETS = 'train_triplets.txt'
    TEST_TRIPLETS = 'test_triplets.txt'

    # generate embedding for each image in the dataset
    if(os.path.exists('dataset/embeddings.npy') == False):
        generate_embeddings()

    np.random.seed(1)
    torch.manual_seed(1)

    # load the training and testing data
    print("Loading Training Data...")
    X, y = get_data(TRAIN_TRIPLETS)
    trained_model = train_model(X, y)

    print("Loading Testing Data...")
    X_test, _ = get_data(TEST_TRIPLETS, train=False)
    test_loader = create_loader_from_np(X_test, train = False, batch_size=1024, shuffle=False)
    # test the model on the test data
    # trained_model = Net()
    test_model(trained_model, test_loader)
    print("Results saved to results.txt")


#  for epoch in range(10):
#     print(f"========== Epoch {epoch} ==========")
#     # Use the part of the training data as a validation split.
#     kf = KFold(10, shuffle=True, random_state=0)
#     for train_split, val_split in kf.split(X):
#         # Create data loaders for the training and validating data
#         train_loader = create_loader_from_np(
#             X[train_split], y[train_split], train = True, batch_size=512)
#         val_loader = create_loader_from_np(
#             X[val_split], y[val_split], train = True, batch_size=512)
        