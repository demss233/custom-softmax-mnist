import numpy as np
from .utils import softmax
from .utils import categorical_crossentropy
from .utils import gradient
from .utils import print_status_bar

def engine(X_train, y_train, epochs, learning_rate = 0.01):
    X_train = np.hstack((np.ones((X_train.shape[0], 1)), X_train))
    y_train = np.eye(10)[y_train.astype(int)]
    weights = np.random.randn(X_train.shape[1], 10)
    biases = np.zeros((1, 10))

    for epoch in range(epochs):
        logits = np.dot(X_train, weights) + biases
        probabilities = softmax(logits)
        loss = categorical_crossentropy(probabilities, y_train, weights) 

        gradients = gradient(X_train, y_train, probabilities = probabilities, logits = logits, weights = weights)
        db = np.mean(probabilities - y_train, axis = 0, keepdims = True) 

        weights -= learning_rate * gradients
        biases -= learning_rate * db

        predictions = np.argmax(probabilities, axis = 1)
        targets = np.argmax(y_train, axis = 1)
        accuracy = np.mean(predictions == targets)

        if epoch % 100 == 0:
            print_status_bar(epoch, accuracy, loss)

    return weights, biases
    