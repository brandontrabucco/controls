"""Author: Brandon Trabucco, Copyright 2019, MIT License"""

from controls.lqr.lqr import lqr
import tensorflow as tf

if __name__ == "__main__":

    A = tf.constant([[
        [-0.313, 56.7, 0.0],
        [-0.0139, -0.426, 0.0],
        [0.0, 56.7, 0.0]]])

    B = tf.constant([[
        [0.232],
        [0.0203],
        [0.0]]])

    Q = tf.constant([[
        [0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0],
        [0.0, 0.0, 1.0]]])

    R = tf.constant([[
        [1.0]]])

    K, P = lqr(
        A,
        B,
        Q,
        R,
        horizon=100)

    states = tf.random.normal([1, 3, 1])

    for i in range(100):

        controls = K @ states

        costs = (
            tf.matmul(tf.matmul(states, Q, transpose_a=True), states) +
            tf.matmul(tf.matmul(controls, R, transpose_a=True), controls))

        print("Cost: {}".format(costs.numpy().sum()))

        states = A @ states + B @ controls
