"""Author: Brandon Trabucco, Copyright 2019, MIT License"""


from controls import infinite_horizon_lqr
from controls.time_varying_linear import time_varying_linear
from controls.shooting.shooting import shooting
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

    K, P = infinite_horizon_lqr(A, B, Q, R)

    print(K)

    def dynamics(x):
        return A @ x[0] + B @ x[1]

    def cost(x):
        return 0.5 * tf.matmul(
            tf.matmul(x[0], Q, transpose_a=True), x[0]) + 0.5 * tf.matmul(
            tf.matmul(x[1], R, transpose_a=True), x[1])

    states = tf.random.normal([1, 3, 1])

    policy = time_varying_linear(
        tf.zeros([100, 1, 1, 1]),
        [tf.zeros([100, 1, 3, 1])],
        [tf.tile(K[None, ...], [100, 1, 1, 1])])

    shooting_states, shooting_controls, shooting_costs = shooting(
        states, policy, dynamics, cost, 100)

    policy = time_varying_linear(
        tf.zeros([100, 1, 1, 1]),
        [tf.zeros([100, 1, 3, 1])],
        [tf.tile(K[None, ...], [100, 1, 1, 1])])

    costs_list = []

    for i in range(100):

        controls = policy(states)

        costs = 0.5 * (tf.matmul(tf.matmul(states, Q, transpose_a=True), states) +
                       tf.matmul(tf.matmul(controls, R, transpose_a=True), controls))

        states = A @ states + B @ controls

        print("Cost: {} Shooting Costs: {}".format(
            costs.numpy()[0][0][0], shooting_costs[i].numpy()[0][0][0]))
