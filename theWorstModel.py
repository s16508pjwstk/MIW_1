import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

print('model y = ax + b')
a = np.loadtxt('misc/dane2.txt')

x = a[:, [0]]
y = a[:, [1]]

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.5, random_state=1)
x_train_sorted = np.sort(x_train, axis=0)
x_test_sorted = np.sort(x_test, axis=0)

c_train = np.hstack([x_train, np.ones(x_train.shape)])
c_test = np.hstack([x_test, np.ones(x_test.shape)])

v_train = np.linalg.pinv(c_train) @ y_train
v_test = np.linalg.pinv(c_test) @ y_test

e_train = sum((y - (v_train[0] * x + v_train[1])) ** 2) / len(x)
print("trained data error index: {}".format(e_train))
e_test = sum((y - (v_test[0] * x * x + v_test[1])) ** 2) / len(x)
print("test data error index: {}".format(e_test))

plt.plot(x_train_sorted, v_train[0] * x_train_sorted + v_train[1], 'r')
plt.plot(x_test_sorted, v_test[0] * x_test_sorted + v_test[1], 'g')

plt.plot(x_train, y_train, 'ro')
plt.plot(x_test, y_test, 'go')

plt.show()


