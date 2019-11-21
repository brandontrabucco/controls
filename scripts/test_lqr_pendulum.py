"""Author: Brandon Trabucco, Copyright 2019, MIT License"""


from controls import lqr
from controls.time_varying_linear import time_varying_linear
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

    def dynamics(x):
        return A @ x[0] + B @ x[1]

    def cost(x):
        return 0.5 * tf.matmul(
            tf.matmul(x[0], Q, transpose_a=True), x[0]) + 0.5 * tf.matmul(
            tf.matmul(x[1], R, transpose_a=True), x[1])

    states = tf.random.normal([1, 4, 1])

    policy = time_varying_linear(
        k,
        [tf.zeros([20, 1, 4, 1])],
        [K])

    shooting_states, shooting_controls, shooting_costs = shooting(
        states, policy, dynamics, cost, 20)

    policy = time_varying_linear(
        k,
        [tf.zeros([20, 1, 4, 1])],
        [K])

    costs_list = []

    for i in range(20):

        controls = policy(states)

        costs = 0.5 * (tf.matmul(tf.matmul(states, Q, transpose_a=True), states) +
                       tf.matmul(tf.matmul(controls, R, transpose_a=True), controls))

        states = A @ states + B @ controls

        print("Cost: {} Shooting Costs: {}".format(
            costs.numpy()[0][0][0], shooting_costs[i].numpy()[0][0][0]))
