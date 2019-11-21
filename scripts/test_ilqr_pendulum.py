"""Author: Brandon Trabucco, Copyright 2019, MIT License"""


from controls.iterative_lqr import iterative_lqr
import tensorflow as tf


if __name__ == "__main__":

    A = tf.constant([[[0., 1., 0., 0.],
                      [0., -.1818, 2.6727, 0.],
                      [0., 0., 0., 1.],
                      [0., -.4545, 31.1818, 0.]]])

    B = tf.constant([[[0.],
                      [1.8182],
                      [0.],
                      [4.5455]]])

    Q = tf.constant([[[1.0, 0.0, 0.0, 0.0],
                      [0.0, 0.0, 0.0, 0.0],
                      [0.0, 0.0, 1.0, 0.0],
                      [0.0, 0.0, 0.0, 0.0]]])

    R = tf.constant([[[1.0]]])

    def dynamics_model(x):
        return A @ x[0] + B @ x[1]

    def cost_model(x):
        return 0.5 * (
            tf.matmul(tf.matmul(x[0], Q, transpose_a=True), x[0]) +
            tf.matmul(tf.matmul(x[1], R, transpose_a=True), x[1]))

    results = iterative_lqr(
        tf.random.normal([1, 4, 1]),
        1,
        dynamics_model,
        cost_model,
        20,
        5)

    for i in range(20):

        costs = results[2][i, ...]

        print("Cost: {}".format(costs.numpy().sum()))
