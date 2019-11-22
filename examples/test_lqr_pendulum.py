"""Author: Brandon Trabucco, Copyright 2019, MIT License"""


from controls import lqr
from controls.time_varying import linear_model
from controls.shooting.shooting import shooting
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

    K, k, P, p = lqr(
        tf.tile(A[None], [20, 1, 1, 1]),
        tf.tile(B[None], [20, 1, 1, 1]),
        tf.zeros([20, 1, 4, 1]),
        tf.tile(Q[None], [20, 1, 1, 1]),
        tf.zeros([20, 1, 4, 1]),
        tf.zeros([20, 1, 1, 4]),
        tf.tile(R[None], [20, 1, 1, 1]),
        tf.zeros([20, 1, 4, 1]),
        tf.zeros([20, 1, 1, 1]))

    def dynamics_model(x):
        return A @ x[0] + B @ x[1]

    def cost_model(x):
        return (tf.matmul(tf.matmul(x[0], Q, transpose_a=True), x[0]) +
                tf.matmul(tf.matmul(x[1], R, transpose_a=True), x[1])) / 2.

    initial_states = tf.random.normal([1, 4, 1])

    controls_model = linear_model(
        k,
        [tf.zeros([20, 1, 4, 1])],
        [K])

    shooting_states, shooting_controls, shooting_costs = shooting(
        initial_states, controls_model, dynamics_model, cost_model, 20)

    for i in range(20):

        costs = shooting_costs[i, ...]

        print("Cost: {}".format(costs.numpy().sum()))
