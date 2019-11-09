"""Author: Brandon Trabucco, Copyright 2019, MIT License"""


import tensorflow as tf
import numpy as np


def kf_update(
        measurement,
        mean,
        covariance,
        controls,
        dynamics_state_jacobian,
        dynamics_controls_jacobian,
        dynamics_covariance,
        measurement_jacobian,
        measurement_covariance
):
    """Compute the bayes optimal estimate of a linear gaussian.

    Args:
    - measurement: a vector of measurements with shape [batch_dim, measurement_dim, 1].
    - mean: the mean of the state estimate with shape [batch_dim, state_dim, 1].
    - covariance: the covariance of the state estimate with shape [batch_dim, state_dim, state_dim].
    - controls: the controls that modify the state with shape [batch_dim, controls_dim, 1].

    - dynamics_state_jacobian: the jacobian of the dynamics wrt. the previous state with shape [batch_dim, state_dim, state_dim].
    - dynamics_controls_jacobian: the jacobian of the dynamics wrt. the controls with shape [batch_dim, state_dim, controls_dim].
    - dynamics_covariance: the covariance of the dynamics noise with shape [batch_dim, state_dim, state_dim].

    - measurement_jacobian: the jacobian of the measurement wrt. the state with shape [batch_dim, measurement_dim, state_dim].
    - measurement_covariance: the covariance of the measurement noise with shape [batch_dim, measurement_dim, measurement_dim].

    Returns:
    - predicted_mean: the predicted next state without measurement.
    - predicted_covariance: the covariance of the predicted next state

    - innovation: the disagreement between the predicted and actual measurement.
    - innovation_covariance: the covariance of the disagreement.

    - mean: the mean of the bayes optimal estimate of the state after measurement.
    - covariance: the covariance of the bayes optimal state after measurement.

    - log_prob: the log likelihood of the current measurement.
        """

    # get the batch shape and vector sizes

    measurement_dim = tf.shape(measurement)[-1]
    state_dim = tf.shape(mean)[-1]
    batch_dim = tf.shape(measurement)[0]

    # make sure all inputs conform to the batch shape

    tf.debugging.assert_equal(
        [batch_dim],
        tf.shape(mean)[:-2],
        message="mean should have same batch size as measurement")

    tf.debugging.assert_equal(
        [batch_dim],
        tf.shape(covariance)[:-2],
        message="covariance should have same batch size as measurement")

    tf.debugging.assert_equal(
        [batch_dim],
        tf.shape(controls)[:-2],
        message="controls should have same batch size as measurement")

    # predict into the future using the controls

    predicted_mean = tf.matmul(
        dynamics_state_jacobian,
        mean) + tf.matmul(
            dynamics_controls_jacobian,
            controls)

    predicted_covariance = dynamics_covariance + tf.matmul(
        dynamics_state_jacobian,
        tf.matmul(
            covariance,
            dynamics_state_jacobian,
            transpose_b=True))

    # compute the innovation

    innovation = measurement - tf.matmul(
        measurement_jacobian,
        predicted_mean)

    innovation_covariance = measurement_covariance + tf.matmul(
        measurement_jacobian,
        tf.matmul(
            predicted_covariance,
            measurement_jacobian,
            transpose_b=True))

    inverse_innovation_covariance = tf.linalg.inv(innovation_covariance)

    # compute the kalman gain

    gain = tf.matmul(
        tf.matmul(
            predicted_covariance,
            measurement_jacobian,
            transpose_b=True), inverse_innovation_covariance)

    # compute the bayes optimal state estimate

    mean = predicted_mean + tf.matmul(gain, innovation)

    covariance = tf.matmul(
        tf.eye(state_dim) - tf.matmul(
            gain,
            measurement_jacobian), predicted_covariance)

    # compute the log likelihood of the measurement

    log_prob_normalizer = tf.math.log(2. * np.pi) * tf.cast(
        measurement_dim, innovation_covariance.dtype) + tf.linalg.logdet(
            innovation_covariance)

    log_prob = -1. / 2. * (log_prob_normalizer + tf.matmul(
        innovation,
        tf.matmul(
            inverse_innovation_covariance,
            innovation), transpose_a=True)[:, 0, 0])

    return (
        predicted_mean,
        predicted_covariance,
        innovation,
        innovation_covariance,
        mean,
        covariance,
        log_prob)


def kf_loop_body(
        measurements,
        mean,
        covariance,
        controls,
        dynamics_state_jacobian,
        dynamics_controls_jacobian,
        dynamics_covariance,
        measurement_jacobian,
        measurement_covariance,
        predicted_mean_array,
        predicted_covariance_array,
        innovation_array,
        innovation_covariance_array,
        mean_array,
        covariance_array,
        log_prob,
        time,
        horizon
):
    """Inner body of kalman filter update loop."""

    # run the kalman filter and collect results

    update_result = kf_update(
        measurements[:, time, :, :],
        mean,
        covariance,
        controls[:, time, :, :],
        dynamics_state_jacobian[:, time, :, :],
        dynamics_controls_jacobian[:, time, :, :],
        dynamics_covariance,
        measurement_jacobian[:, time, :, :],
        measurement_covariance)

    # push the results into arrays for safe keeping

    predicted_mean_array = predicted_mean_array.write(
        time, update_result[0])

    predicted_covariance_array = predicted_covariance_array.write(
        time, update_result[1])

    innovation_array = innovation_array.write(
        time, update_result[2])

    innovation_covariance_array = innovation_covariance_array.write(
        time, update_result[3])

    mean_array = mean_array.write(
        time, update_result[4])

    covariance_array = covariance_array.write(
        time, update_result[5])

    log_prob = log_prob + update_result[6]

    time = time + 1

    return (
        measurements,
        mean,
        covariance,
        controls,
        dynamics_state_jacobian,
        dynamics_controls_jacobian,
        dynamics_covariance,
        measurement_jacobian,
        measurement_covariance,
        predicted_mean_array,
        predicted_covariance_array,
        innovation_array,
        innovation_covariance_array,
        mean_array,
        covariance_array,
        log_prob,
        time,
        horizon)


def kf_loop_condition(
        measurements,
        mean,
        covariance,
        controls,
        dynamics_state_jacobian,
        dynamics_controls_jacobian,
        dynamics_covariance,
        measurement_jacobian,
        measurement_covariance,
        predicted_mean_array,
        predicted_covariance_array,
        innovation_array,
        innovation_covariance_array,
        mean_array,
        covariance_array,
        log_prob,
        time,
        horizon
):
    """Inner condition of kalman filter update loop."""

    # exit the loop if time is greater than the horizon

    return tf.less(time, horizon)
