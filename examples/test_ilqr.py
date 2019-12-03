"""Author: Brandon Trabucco, Copyright 2019, MIT License"""


from controls.distributions.gaussian import Gaussian
from controls import iterative_lqr
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

    controls_model = Gaussian(lambda time, inputs: (
        tf.zeros([1, 1]), tf.ones([1, 1, 1]), tf.ones([1, 1, 1]), tf.zeros([1])))

    def dynamics_model(time, inputs):
        return (A @ inputs[0][:, :, tf.newaxis] + B @ inputs[1][:, :, tf.newaxis])[:, :, 0]

    def cost_model(time, inputs):
        return 0.5 * (
            tf.matmul(tf.matmul(inputs[0][:, :, tf.newaxis], Q, transpose_a=True),
                      inputs[0][:, :, tf.newaxis]) +
            tf.matmul(tf.matmul(inputs[1][:, :, tf.newaxis], R, transpose_a=True),
                      inputs[1][:, :, tf.newaxis]))[:, 0, 0]

    initial_states = tf.random.normal([1, 3])

    controls_model = iterative_lqr(
        initial_states,
        controls_model,
        dynamics_model,
        cost_model,
        h=20,
        n=10,
        a=0.1,
        random=False)

    shooting_states, shooting_controls, shooting_costs = shooting(
        initial_states, controls_model, dynamics_model, cost_model, h=20, random=False)

    for i in range(20):

        costs = shooting_costs[i, ...]

        print("Cost: {}".format(costs.numpy().sum()))
