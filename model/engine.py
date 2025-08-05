import numpy as np
from sklearn.model_selection import train_test_split
from .utils import load_data
from .config import engine

SEED = 42
TEST_SPLIT_SIZE = 0.2

train_df, test_df = load_data()
X = train_df.iloc[:, 1:]
y = train_df.iloc[:, 0] 

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = TEST_SPLIT_SIZE, random_state = SEED)

# For X_train, p = wx where w -> weight matrix (shape: [784, 10]) 
def processed():
    weights, biases = engine(X_train, y_train, epochs = 1000, learning_rate = 0.01)
    return weights, biases