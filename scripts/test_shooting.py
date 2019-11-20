"""Author: Brandon Trabucco, Copyright 2019, MIT License"""


from controls.shooting.shooting import shooting
from controls.taylor_series import first_order, second_order
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

    def policy(x):
        return tf.random.normal([1, 1, 1])

    def dynamics(x):
        return A @ x[0] + B @ x[1]

    def cost(x):
        return 0.5 * tf.matmul(
            tf.matmul(x[0], Q, transpose_a=True), x[0]) + 0.5 * tf.matmul(
            tf.matmul(x[1], R, transpose_a=True), x[1])

    states = tf.random.normal([1, 3, 1])

    shooting_states, shooting_controls, shooting_costs = shooting(
        states, policy, dynamics, cost, 10)

    for i in range(10):

        controls = shooting_controls[i]

        z = first_order(dynamics, states, controls)
        print(z)

        states = A @ states + B @ controls

        z = second_order(cost, states, controls)
        print(z)
