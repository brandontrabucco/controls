"""Author: Brandon Trabucco, Copyright 2019, MIT License"""


from controls.iterative_lqr import iterative_lqr
import tensorflow as tf


if __name__ == "__main__":

    A = tf.constant([[[-0.313, 56.7, 0.0],
                      [-0.0139, -0.426, 0.0],
                      [0.0, 56.7, 0.0]]])

    B = tf.constant([[[0.232],
                      [0.0203],
                      [0.0]]])

    Q = tf.constant([[[0.0, 0.0, 0.0],
                      [0.0, 0.0, 0.0],
                      [0.0, 0.0, 1.0]]])

    R = tf.constant([[[1.0]]])

    def controls_model(x):
        return tf.zeros([1, 1, 1])

    def dynamics_model(x):
        return A @ x[0] + B @ x[1]

    def cost_model(x):
        return 0.5 * tf.matmul(
            tf.matmul(x[0], Q, transpose_a=True), x[0]) + 0.5 * tf.matmul(
            tf.matmul(x[1], R, transpose_a=True), x[1])

    results = iterative_lqr(
        tf.random.normal([1, 3, 1]),
        controls_model,
        dynamics_model,
        cost_model,
        20,
        5)

    for i in range(20):

        costs = results[2][i, ...]

        print("Cost: {}".format(costs.numpy().sum()))
