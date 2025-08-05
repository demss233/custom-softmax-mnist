import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import os

def load_data():
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    train_df = pd.read_csv(os.path.join(BASE_DIR, 'data', 'train.csv'))
    test_df = pd.read_csv(os.path.join(BASE_DIR, 'data', 'test.csv'))
    return train_df, test_df

def sigmoid(logit):
    return 1 / (np.exp(-logit) + 1)

def softmax(z):
    # Subtracting the max value in each row of the matrix z, keeping the shape intact
    z -= np.max(z, axis = 1, keepdims = True)
    exps = np.exp(z)
    return exps / np.sum(exps, axis = 1, keepdims = True)

def categorical_crossentropy(predictions, labels, weights):
    return -np.mean(np.sum(labels * np.log(predictions + 1e-9), axis = 1))

def gradient(X, y, probabilities, logits, weights):
    m = X.shape[0]
    derivative = (1 / m) * np.dot(X.T, (probabilities - y))
    return derivative

def print_status_bar(epoch, accuracy, loss):
    print(f"\n⤷ Epoch: {epoch}")
    print(f"Accuracy: {accuracy: .4f}")
    print(f"Log Loss: {loss: .4f}")