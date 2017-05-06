# simple NN example

import numpy as np
import time


def target(x1, x2):
    return int(x1+x2)%2


def sigmoid(x):
    return 1.0/(1.0 + np.exp(-x))


def sigmoid_prime(x):
    return sigmoid(x) * (1-sigmoid(x))


def tanh_prime(x):
    return 1 - np.tanh(x)**2


def train(x, y, V, W, bv, bw):
    # forward
    A = np.dot(x, V) + bv
    Z = np.tanh(A)

    B = np.dot(Z, W) + bw
    Y = sigmoid(B)

    # backward
    Ew = Y - y
    Ev = tanh_prime(A) * np.dot(W, Ew)

    # loss
    dW = np.outer(Z, Ew)
    dV = np.outer(x, Ev)

    loss = -np.mean(y*np.log(Y)+(1-y)*np.log(1-Y))
    print(loss)

    return loss, (dV, dW, Ev, Ew)


def predict(x, V, W, bv, bw):
    A = np.dot(x, V) + bv
    B = np.dot(np.tanh(A), W) + bw
    return (sigmoid(B) > 0.5).astype(int)


def main():
    n_samples = 100
    n_features = 10
    n_hidden = 10
    n_output = n_features

    learning_rate = 0.01
    momentum = 0.9

    # np.random.seed(0)

    X = np.random.binomial(1, 0.5, (n_samples, n_features))

    Y = X ^ 1

    Q1 = np.random.normal(scale=0.1, size=(n_features, n_hidden))
    Q2 = np.random.normal(scale=0.1, size=(n_hidden, n_output))

    bv = np.zeros(n_hidden)
    bw = np.zeros(n_output)

    params = [Q1, Q2, bv, bw]

    for epoch in range(10):
        err = []
        upd = [0]*len(params)

        t0 = time.clock()
        for i in range(X.shape[0]):
            loss, grad = train(X[i], Y[i], *params)

            for j in range(len(params)):
                upd[j] = learning_rate * grad[j] + momentum * upd[j]
                params[j] -= upd[j]

            err.append(loss)

        print('Epoch: %d, Loss: %.8f, Time: %fs' % (
            epoch, np.mean(err), time.clock()-t0))

    x = np.random.binomial(1, 0.5, n_features)
    print('XOR prediction:')
    print(x)
    pred_x = predict(x, *params)
    print(pred_x)

    print('Accuracy: ', np.sum(pred_x == x ^ 1)/x.shape[0])


if __name__ == "__main__":
    main()