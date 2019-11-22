"""Author: Brandon Trabucco, Copyright 2019, MIT License"""


from controls import iterative_lqr
from controls import shooting
import tensorflow as tf


if __name__ == "__main__":

    size = 5

    goal = tf.ones([1, size, 1])

    def controls_model(x):
        return tf.zeros([1, size, 1])

    def dynamics_model(x):
        return x[0] + x[1]

    def cost_model(x):
        return 0.5 * (
            tf.matmul(x[0] - goal, x[0] - goal, transpose_a=True) +
            tf.matmul(x[1], x[1], transpose_a=True))

    initial_states = -tf.ones([1, size, 1])

    controls_model = iterative_lqr(
        initial_states,
        controls_model,
        dynamics_model,
        cost_model,
        horizon=20,
        num_iterations=10,
        alpha=0.1)

    shooting_states, shooting_controls, shooting_costs = shooting(
        initial_states, controls_model, dynamics_model, cost_model, 20)

    for i in range(20):

        costs = shooting_costs[i, ...]

        print("Cost: {}".format(costs.numpy().sum()))
