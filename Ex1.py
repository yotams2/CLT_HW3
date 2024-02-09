import numpy as np
import matplotlib.pyplot as plt

train = np.loadtxt("data/train.csv", delimiter=",")
test = np.loadtxt("data/test.csv", delimiter=",")


def calc_k(set_1, set_2, kernel, sigma=1, q=1):
    m_1 = set_1.shape[0]
    m_2 = set_2.shape[0]
    K = np.zeros([m_1, m_2])

    for (i, j), _ in np.ndenumerate(K):
        x_i = set_1[i, 0:2]
        x_j = set_2[j, 0:2]
        if kernel == "RBF":
            K[i, j] = np.exp(-((np.linalg.norm(x_i - x_j) ** 2) / (2 * (sigma ** 2))))
        elif kernel == "Polynomial":
            K[i, j] = (1 + np.dot(x_i, x_j)) ** q
        else:
            print("unsupported kernel")
            exit()

    return K


def estimate(dataset, k, alpha):
    dataset_est = np.zeros(dataset.shape)

    for i in range(dataset.shape[0]):
        y_hat_i = np.sign(np.dot(alpha, k[:, i]))
        if y_hat_i == 0:
            y_hat_i = 1
        dataset_est[i, 0:2] = dataset[i, 0:2]
        dataset_est[i, 2] = y_hat_i

    return dataset_est


def get_arrays(dataset, dataset_est):
    tp = dataset[(dataset_est[:, 2] == 1) & (dataset[:, 2] == 1), 0:2]
    tn = dataset[(dataset_est[:, 2] == -1) & (dataset[:, 2] == -1), 0:2]
    fp = dataset[(dataset_est[:, 2] == 1) & (dataset[:, 2] == -1), 0:2]
    fn = dataset[(dataset_est[:, 2] == -1) & (dataset[:, 2] == 1), 0:2]
    return tp, tn, fp, fn


def run_perceptron(kernel, T, sigma=1, q=1, show_results=True):
    m = train.shape[0]
    alpha = np.zeros(m)
    k = calc_k(train, train, kernel, sigma, q)
    k_test = calc_k(train, test, kernel, sigma, q)

    for t in range(T):
        for i in range(m):
            y_hat_i = np.sign(np.dot(alpha, k[:, i]))
            if y_hat_i == 0:
                y_hat_i = 1
            y_i = train[i, 2]
            alpha[i] += (y_i - y_hat_i) / 2

    train_est = estimate(train, k, alpha)
    test_est = estimate(test, k_test, alpha)

    test_tp, test_tn, test_fp, test_fn = get_arrays(test, test_est)
    train_tp, train_tn, train_fp, train_fn = get_arrays(train, train_est)

    if show_results:
        plt.scatter(test_tn[:, 0], test_tn[:, 1], c="red", marker="*")
        plt.scatter(test_tp[:, 0], test_tp[:, 1], c="blue", marker="*")
        plt.scatter(test_fp[:, 0], test_fp[:, 1], c="green", marker="*")
        plt.scatter(test_fn[:, 0], test_fn[:, 1], c="black", marker="*")

        plt.scatter(train_tn[:, 0], train_tn[:, 1], c="red")
        plt.scatter(train_tp[:, 0], train_tp[:, 1], c="blue")
        plt.scatter(train_fp[:, 0], train_fp[:, 1], c="green")
        plt.scatter(train_fn[:, 0], train_fn[:, 1], c="black")

        title = f"Estimated classification of the datasets\nKernel = {kernel}, T = {T}, "
        if kernel == "RBF":
            title += f"sigma = {sigma}"
        elif kernel == "Polynomial":
            title += f"q = {q}"
        plt.title(title)
        plt.savefig(f"est_classification_{kernel}")
        plt.show()

        print(f"The number of mistakes for the {kernel} kernel is:")
        print(f"{len(test_fp) + len(test_fn)} mistakes for the test dataset")
        print(f"{len(train_fp) + len(train_fn)} mistakes for the training dataset")

    return len(train_fp) + len(train_fn) + len(test_fp) + len(test_fn)


def main():
    sigma = 0.5
    T = 6
    q = 5

    train_pos = train[train[:, 2] == 1, 0:2]
    train_neg = train[train[:, 2] == -1, 0:2]

    plt.scatter(train_neg[:, 0], train_neg[:, 1], c="red")
    plt.scatter(train_pos[:, 0], train_pos[:, 1], c="blue")
    plt.title("True classification of the training set")
    plt.savefig("true_classification")
    plt.show()

    run_perceptron("RBF", T, sigma=sigma)
    run_perceptron("Polynomial", T, q=q)

    # best_res = 100
    # for T_i in range(50):
    #     print(f"T = {T_i}")
    #     for q_i in range(50):
    #         res = run_perceptron("Polynomial", T_i, q=q_i, show_results=False)
    #         if res < best_res:
    #             best_res = res
    #             best_T = T_i
    #             best_q = q_i
    # print(f"best_T = {best_T}, best_q = {best_q}, best_res = {best_res}")


if __name__ == "__main__":
    main()
