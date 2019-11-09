"""Author: Brandon Trabucco, Copyright 2019, MIT License"""


from controls.kf.kf import kf
import tensorflow as tf


if __name__ == "__main__":

    batch_dim = 1
    horizon = 20
    state_dim = 32
    measurement_dim = state_dim
    controls_dim = state_dim

    A = tf.eye(state_dim, batch_shape=[batch_dim, horizon])
    B = tf.eye(state_dim, batch_shape=[batch_dim, horizon])
    C = tf.eye(state_dim, batch_shape=[batch_dim, horizon])

    states = [
        tf.random.normal([batch_dim, 1, state_dim, 1])]

    controls = [
        tf.random.normal([batch_dim, 1, controls_dim, 1])]

    measurements = [
        C[:, 0:1, :, :] @ states[-1] +
        tf.random.normal([batch_dim, 1, measurement_dim, 1]) * 0.01]

    for i in range(1, horizon):

        states.append(
            A[:, i:i+1, :, :] @ states[-1] +
            B[:, i:i+1, :, :] @ controls[-1] +
            tf.random.normal([batch_dim, 1, state_dim, 1]) * 0.01)

        controls.append(tf.random.normal([batch_dim, 1, controls_dim, 1]))

        measurements.append(
            C[:, i:i+1, :, :] @ states[-1] +
            tf.random.normal([batch_dim, 1, measurement_dim, 1]) * 0.01)

    states = tf.concat(states, 1)
    controls = tf.concat(controls, 1)
    measurements = tf.concat(measurements, 1)

    result = kf(
        measurements[:, 1:, :, :],
        tf.zeros([batch_dim, state_dim, 1]),
        tf.eye(state_dim, batch_shape=[batch_dim]),
        controls[:, :-1, :, :],
        A[:, :-1, :, :],
        B[:, :-1, :, :],
        tf.eye(state_dim, batch_shape=[batch_dim]) * 0.01,
        C[:, :-1, :, :],
        tf.eye(measurement_dim, batch_shape=[batch_dim]) * 0.01)

    error = tf.linalg.norm(result[0] - states[:, 1:, :, :], ord=1)
    print("error prediction", error.numpy())

    error = tf.linalg.norm(result[4] - states[:, 1:, :, :], ord=1)
    print("error filter", error.numpy())

    error = tf.linalg.norm(result[8] - states[:, 1:, :, :], ord=1)
    print("error smooth", error.numpy())

    for t in range(horizon - 1):
        print(tf.linalg.norm(result[0][0, t, ...] - states[0, t + 1, ...], ord=1).numpy())
        print(tf.linalg.norm(result[4][0, t, ...] - states[0, t + 1, ...], ord=1).numpy())
        print()

