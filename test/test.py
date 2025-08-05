import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from model.utils import load_data
from model.utils import softmax
from model.engine import processed
import numpy as np
import pandas as pd

train_df, test_df = load_data()
weights, biases = processed()

test_df = np.hstack((np.ones((test_df.shape[0], 1)), test_df))
predictions = np.dot(test_df, weights)
predictions = softmax(predictions)
predictions = np.argmax(predictions, axis = 1) 

result_csv = pd.DataFrame({
    'ImageId': np.arange(1, len(predictions) + 1),
    'Label': predictions,
})

print("\nSaving the results..")
result_csv.to_csv('submission.csv', index = False)
print("The results were saved..")