"""Author: Brandon Trabucco, Copyright 2019, MIT License"""


from controls.distributions.deterministic import Deterministic
from controls import cem
from controls import shooting
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

    controls_model = Deterministic(lambda time, inputs: tf.zeros([1, 1, 1]))
    dynamics_model = Deterministic(lambda time, inputs: A @ inputs[0] + B @ inputs[1])
    cost_model = Deterministic(lambda time, inputs: 0.5 * (
        tf.matmul(tf.matmul(inputs[0], Q, transpose_a=True), inputs[0]) +
        tf.matmul(tf.matmul(inputs[1], R, transpose_a=True), inputs[1])))

    initial_states = tf.random.normal([1, 3, 1])

    controls_model = cem(
        initial_states,
        controls_model,
        dynamics_model,
        cost_model,
        h=20,
        c=1000,
        n=100,
        k=100,
        s=1.0)

    xi, xi_log_prob, ui, ui_log_prob, ci, ci_log_prob = shooting(
        initial_states, controls_model, dynamics_model, cost_model, 20)

    for i in range(20):

        costs = ci[i, ...]
        print("Cost: {}".format(costs.numpy().sum()))
