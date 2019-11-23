"""Author: Brandon Trabucco, Copyright 2019, MIT License"""


from controls.distributions.deterministic import Deterministic
from controls import iterative_lqr
from controls import shooting
import tensorflow as tf


if __name__ == "__main__":

    goal = tf.ones([1, 2, 1])

    controls_model = Deterministic(lambda time, inputs: tf.zeros([1, 2, 1]))
    dynamics_model = Deterministic(lambda time, inputs: inputs[0] + tf.stack([
            tf.clip_by_value(inputs[1][:, 0, :], -1., 1.) * tf.cos(inputs[0][:, 2, :]),
            tf.clip_by_value(inputs[1][:, 0, :], -1., 1.) * tf.sin(inputs[0][:, 2, :]),
            tf.clip_by_value(inputs[1][:, 1, :], -1., 1.)], 1))
    cost_model = Deterministic(lambda time, inputs: tf.matmul(
        inputs[0][:, :2, :] - goal, inputs[0][:, :2, :] - goal, transpose_a=True))

    initial_states = -tf.ones([1, 3, 1])

    controls_model = iterative_lqr(
        initial_states,
        controls_model,
        dynamics_model,
        cost_model,
        h=10,
        n=100,
        a=0.1)

    shooting_states, shooting_controls, shooting_costs = shooting(
        initial_states, controls_model, dynamics_model, cost_model, h=10)

    for i in range(10):

        costs = shooting_costs[i, ...]

        print("Cost: {}".format(costs.numpy().sum()))
