import numpy as np
import matplotlib.pyplot as plt
import random


def h_func(x, theta, j):
    if x[j] <= theta:
        return 1
    else:
        return -1


def adaboost(x, y, T):
    m = x.shape[0]
    d = x.shape[1]
    p = [1/m] * m  # initialize an array of length m

    theta_arr = list(set(i for j in x for i in j))  # find all unique pixel values, these are the options for theta
    j_arr = range(d)
    h_params_mat = np.array(np.meshgrid(theta_arr, j_arr)).T.reshape(-1, 2)  # matrix of all possible parameters for h

    h_t_params = np.zeros(T)
    alpha = np.zeros(T)

    for t in range(T):
        print(f"Running iteration {t}/{T}")
        p_arr = np.zeros(h_params_mat.shape[0] * 2)
        for index, [theta, j] in enumerate(h_params_mat):
            j = int(j)
            h_left_x = np.array(x[:, j] <= theta, dtype=int)
            h_right_x = np.array(x[:, j] > theta, dtype=int)
            p_arr[index] = np.array(h_left_x == y, dtype=int).sum() / m

    res = np.sign(np.dot(alpha, h_arr))


def main():
    x_train = np.loadtxt(fname="data/MNIST_train_images.csv", delimiter=",")
    y_train = np.loadtxt(fname="data/MNIST_train_labels.csv", delimiter=",")
    T = 30

    index = random.randint(0, len(x_train))
    plt.imshow(np.asarray(np.reshape(x_train[index, :], (28, 28))), cmap="gray", vmin=0, vmax=255)
    plt.title(f"Sample index = {index}")
    plt.savefig("random_MNIST_sample")
    plt.show()

    adaboost(x_train, y_train, T)

if __name__ == "__main__":
    main()