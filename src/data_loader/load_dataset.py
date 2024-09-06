from tensorflow.keras.datasets import cifar10
import numpy as np

def load_cifar10():
    (x_train, _), (x_test, _) = cifar10.load_data()
    x_train = x_train.astype(np.float32) / 255.
    x_test = x_test.astype(np.float32) / 255.
    return x_train, x_test
