
import numpy as np
from matplotlib import pyplot as plt

from nn.load import load_data
from nn.layers import sigmoid


def average_digit(data, digit):
    filtered_data = [x[0] for x in data if np.argmax(x[1]) == digit]
    filtered_array = np.asarray(filtered_data)
    return np.average(filtered_array, axis=0)


train, test = load_data()
avg_eight = average_digit(train, 8)
img = np.reshape(avg_eight, (28, 28))
plt.imshow(img)
# plt.show()


def predict(x, w, b):
    return sigmoid(np.dot(w, x) + b)


def evaluate(data, digit, threshold, w, b):
    total = 1.0 * len(data)
    correct = 0.0
    for x in data:
        if predict(x[0], w, b) > threshold and np.argmax(x[1]) == digit:
            correct += 1
        elif predict(x[0], w, b) <= threshold and np.argmax(x[1]) != digit:
            correct += 1
    return correct / total


w = np.transpose(avg_eight)
b = -45

x_3 = train[2][0]
x_18 = train[17][0]

plt.imshow(np.reshape(x_3, (28, 28)))
# plt.show()
plt.imshow(np.reshape(x_18, (28, 28)))
# plt.show()

print(np.dot(w, x_3))
print(np.dot(w, x_18))

print(predict(x_3, w, b))
print(predict(x_18, w, b))

print(evaluate(train, 8, 0.5, w, b))
print(evaluate(test, 8, 0.5, w, b))
