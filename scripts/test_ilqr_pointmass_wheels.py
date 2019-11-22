"""Author: Brandon Trabucco, Copyright 2019, MIT License"""


from controls import iterative_lqr
from controls import shooting
import tensorflow as tf


if __name__ == "__main__":

    goal = tf.ones([1, 2, 1])

    def controls_model(x):
        return tf.zeros([1, 2, 1])

    def dynamics_model(x):
        return x[0] + tf.stack([
            tf.clip_by_value(x[1][:, 0, :], -1., 1.) * tf.cos(x[0][:, 2, :]),
            tf.clip_by_value(x[1][:, 0, :], -1., 1.) * tf.sin(x[0][:, 2, :]),
            tf.clip_by_value(x[1][:, 1, :], -1., 1.)], 1)

    def cost_model(x):
        return 0.5 * tf.matmul(
            x[0][:, :2, :] - goal, x[0][:, :2, :] - goal, transpose_a=True)

    initial_states = -tf.ones([1, 3, 1])

    controls_model = iterative_lqr(
        initial_states,
        controls_model,
        dynamics_model,
        cost_model,
        horizon=10,
        num_iterations=100,
        trust_region_alpha=0.1)

    shooting_states, shooting_controls, shooting_costs = shooting(
        initial_states, controls_model, dynamics_model, cost_model, 10)

    for i in range(10):

        costs = shooting_costs[i, ...]

        print("Cost: {}".format(costs.numpy().sum()))
