import numpy as np
import matplotlib.pyplot as plt
import random


def adaboost(x, y, x_test, y_test, T):
    m = x.shape[0]
    m_test = x_test.shape[0]
    d = x.shape[1]
    p = [1/m] * m  # initialize an array of length m

    theta_arr = list(set(i for j in x for i in j))  # find all unique pixel values, these are the options for theta
    j_arr = range(d)
    sign_arr = [1, -1]  # by using 1*h(x) or (-1)*h(x), we have the two h function defined in the exercise
    h_params_mat = np.array(np.meshgrid(theta_arr, j_arr, sign_arr)).T.reshape(-1, 3)  # matrix of all possible parameters for h

    indices_to_remove = []
    for index, [theta, j, sign] in enumerate(h_params_mat):
        if theta not in x[:, int(j)]:
            indices_to_remove += [index]
    h_params_mat = np.delete(h_params_mat, indices_to_remove, axis=0)

    h_t_params = np.zeros([T, 3])
    alpha = np.zeros(T)
    h_t_x = np.zeros([T, m])
    h_t_x_test = np.zeros([T, m_test])

    err_rate_train = np.zeros(T)
    err_rate_test = np.zeros(T)

    for t in range(T):
        print(f"Running iteration {t}/{T}")
        err_prob = np.zeros(h_params_mat.shape[0])
        for index, [theta, j, sign] in enumerate(h_params_mat):
            j = int(j)
            h_x = np.array(x[:, j] <= theta, dtype=int)
            h_x[h_x == 0] = -1
            h_x *= int(sign)
            err_prob[index] = np.dot(p, np.array(h_x != y, dtype=int))

        h_t_params[t] = h_params_mat[np.argmin(err_prob)]
        epsilon_t = np.min(err_prob)
        alpha[t] = 0.5 * np.log((1-epsilon_t)/epsilon_t)
        theta_t = h_t_params[t][0]
        j_t = int(h_t_params[t][1])
        sign_t = int(h_t_params[t][2])
        h_t_x[t] = np.array(x[:, j_t] <= theta_t, dtype=int)
        h_t_x[t][h_t_x[t] == 0] = -1
        h_t_x[t] *= sign_t
        sum_t = np.dot(p, np.exp(-alpha[t] * y * h_t_x[t]))
        p = p * np.exp(-alpha[t] * y * h_t_x[t]) / sum_t

        h_x_final_t = np.sign(np.sum([np.multiply(alpha[i], h_t_x[i]) for i in range(t+1)], axis=0))
        h_x_final_t_err = np.sum(np.array(h_x_final_t != y, dtype=int))
        err_rate_train[t] = 100 * h_x_final_t_err / m
        print(f"TRAIN SET :: Error rate is {err_rate_train[t]}% [{h_x_final_t_err}/{m}]")

        h_t_x_test[t] = np.array(x_test[:, j_t] <= theta_t, dtype=int)
        h_t_x_test[t][h_t_x_test[t] == 0] = -1
        h_t_x_test[t] *= sign_t
        h_x_test_final_t = np.sign(np.sum([np.multiply(alpha[i], h_t_x_test[i]) for i in range(t + 1)], axis=0))
        h_x_test_final_t_err = np.sum(np.array(h_x_test_final_t != y_test, dtype=int))
        err_rate_test[t] = 100 * h_x_test_final_t_err / m_test
        print(f"TEST SET :: Error rate is {err_rate_test[t]}% [{h_x_test_final_t_err}/{m_test}]")

    return err_rate_train, err_rate_test


def plot_acc(acc, T, set_name):
    plt.plot(range(T), acc)
    plt.xlabel("Iteration")
    plt.ylabel("Accuracy [%]")
    plt.title(f"Accuracy per iteration - {set_name} set")
    plt.savefig(f"Ex2_accuracy_{set_name}")
    plt.show()

def main():
    x_train = np.loadtxt(fname="data/MNIST_train_images.csv", delimiter=",")
    y_train = np.loadtxt(fname="data/MNIST_train_labels.csv", delimiter=",")
    x_test = np.loadtxt(fname="data/MNIST_test_images.csv", delimiter=",")
    y_test = np.loadtxt(fname="data/MNIST_test_labels.csv", delimiter=",")
    T = 30
    new_training = False

    index = random.randint(0, len(x_train))
    plt.imshow(np.asarray(np.reshape(x_train[index, :], (28, 28))), cmap="gray", vmin=0, vmax=255)
    plt.title(f"Sample index = {index}")
    plt.savefig("random_MNIST_sample")
    plt.show()

    if new_training:
        err_rate_train, err_rate_test = adaboost(x_train, y_train, x_test, y_test, T)

        with open('err_rate_train.npy', 'wb') as f:
            np.save(f, err_rate_train)
        with open('err_rate_test.npy', 'wb') as f:
            np.save(f, err_rate_test)
    else:
        with open('err_rate_train.npy', 'rb') as f:
            err_rate_train = np.load(f)
        with open('err_rate_test.npy', 'rb') as f:
            err_rate_test = np.load(f)
        plot_acc(100-err_rate_train, T, "training")
        plot_acc(100-err_rate_test, T, "test")


if __name__ == "__main__":
    main()
