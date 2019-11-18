"""Author: Brandon Trabucco, Copyright 2019, MIT License"""

from controls.lqr.lqr import lqr
import tensorflow as tf

if __name__ == "__main__":

    A = tf.constant([[[
        [-0.313, 56.7, 0.0],
        [-0.0139, -0.426, 0.0],
        [0.0, 56.7, 0.0]]]])

    A = tf.tile(A, [100, 1, 1, 1])

    B = tf.constant([[[
        [0.232],
        [0.0203],
        [0.0]]]])

    B = tf.tile(B, [100, 1, 1, 1])

    Q = tf.constant([[[
        [0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0],
        [0.0, 0.0, 1.0]]]])

    Q = tf.tile(Q, [100, 1, 1, 1])

    R = tf.constant([[[
        [1.0]]]])

    R = tf.tile(R, [100, 1, 1, 1])

    K, k, P, p = lqr(
        A,
        B,
        tf.zeros([100, 1, 3, 1]),
        Q,
        tf.zeros([100, 1, 3, 1]),
        tf.zeros([100, 1, 1, 3]),
        R,
        tf.zeros([100, 1, 3, 1]),
        tf.zeros([100, 1, 1, 1]))

    states = tf.random.normal([1, 3, 1])

    for i in range(100):

        controls = K[i, :, :, :] @ states + k[i, :, :, :]

        costs = (
            tf.matmul(tf.matmul(states, Q[i, :, :, :], transpose_a=True), states) +
            tf.matmul(tf.matmul(controls, R[i, :, :, :], transpose_a=True), controls))

        print("Cost: {}".format(costs.numpy().sum()))

        states = A[i, :, :, :] @ states + B[i, :, :, :] @ controls
