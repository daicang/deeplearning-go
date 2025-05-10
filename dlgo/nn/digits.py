
import numpy as np
from dlgo.nn.load import load_data
from dlgo.nn.layers import sigmoid_double


def average_digit(data, digit):
    filtered_data = [x[0] for x in data if np.argmax(x[1]) == digit]
    filtered_array = np.asarray(filtered_data)
    return np.array(filtered_array, axis=0)


train, test = load_data()
avg_eight = average_digit(train, 8)
