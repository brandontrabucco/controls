"""Author: Brandon Trabucco, Copyright 2019, MIT License"""


from controls.kf.update import kf_loop_body, kf_loop_condition
from controls.kf.smooth import kf_smooth_loop_body, kf_smooth_loop_condition
from controls.kf.em import kf_em_loop_body, kf_em_loop_condition
import tensorflow as tf


def kf(
        measurements,
        initial_mean,
        initial_covariance,
        controls,
        dynamics_state_jacobian,
        dynamics_controls_jacobian,
        dynamics_covariance,
        measurement_jacobian,
        measurement_covariance
):
    """Compute the bayes optimal estimates of a linear gaussian.

    Args:
    - measurements: a vector of measurements with shape [..., T, measurement_dim, 1].
    - initial_mean: the mean of the state estimate with shape [..., state_dim, 1].
    - initial_covariance: the covariance of the state estimate with shape [..., state_dim, state_dim].
    - controls: the controls that modify the state with shape [..., T, controls_dim, 1].

    - dynamics_state_jacobian: the jacobian of the dynamics wrt. the previous state with shape [..., T, state_dim, state_dim].
    - dynamics_controls_jacobian: the jacobian of the dynamics wrt. the controls with shape [..., T, state_dim, controls_dim].
    - dynamics_covariance: the covariance of the dynamics noise with shape [..., state_dim, state_dim].

    - measurement_jacobian: the jacobian of the measurement wrt. the state with shape [..., T, measurement_dim, state_dim].
    - measurement_covariance: the covariance of the measurement noise with shape [..., measurement_dim, measurement_dim].

    Returns:
    - predicted_mean: the predicted states without measurement.
    - predicted_covariance: the covariance of the predicted states without measurement.

    - innovation: the disagreement between the predicted and actual measurements.
    - innovation_covariance: the covariance of the disagreement.

    - mean: the mean of the bayes optimal estimate of the states.
    - covariance: the covariance of the bayes optimal states.
    - log_prob: the log likelihood of the measurements.

    - smooth_gain: the smooth gain used to estimate the smooth mean and covariance.
    - smooth_mean: the mean after measuring every step.
    - smooth_covariance: the covariance after measuring every step.

    - dynamics_covariance: the maximum likelihood estimate of the dynamics covariance.
    - measurement_covariance: the maximum likelihood estimate of the measurement covariance.
        """

    # record the batch shape, vector sizes and horizon length

    measurement_dim = tf.shape(measurements)[-2]
    state_dim = tf.shape(initial_mean)[-2]
    controls_dim = tf.shape(controls)[-2]
    batch_shape = tf.shape(measurements)[:-3]
    batch_dim = tf.reduce_prod(batch_shape)
    horizon = tf.shape(measurements)[-3]

    # make sure all inputs conform to the batch shape

    tf.debugging.assert_equal(
        batch_shape,
        tf.shape(initial_mean)[:-2],
        message="initial mean should have same batch shape as measurements")

    tf.debugging.assert_equal(
        batch_shape,
        tf.shape(initial_covariance)[:-2],
        message="initial covariance should have same batch shape as measurements")

    tf.debugging.assert_equal(
        batch_shape,
        tf.shape(controls)[:-3],
        message="controls should have same batch shape as measurements")

    tf.debugging.assert_equal(
        batch_shape,
        tf.shape(dynamics_state_jacobian)[:-3],
        message="dynamics state jacobian should have same batch shape as measurements")

    tf.debugging.assert_equal(
        batch_shape,
        tf.shape(dynamics_controls_jacobian)[:-3],
        message="dynamics controls jacobian should have same batch shape as measurements")

    tf.debugging.assert_equal(
        batch_shape,
        tf.shape(dynamics_covariance)[:-2],
        message="dynamics covariance should have same batch shape as measurements")

    tf.debugging.assert_equal(
        batch_shape,
        tf.shape(measurement_jacobian)[:-3],
        message="measurement jacobian should have same batch shape as measurements")

    tf.debugging.assert_equal(
        batch_shape,
        tf.shape(measurement_covariance)[:-2],
        message="measurement covariance should have same batch shape as measurements")

    tf.debugging.assert_equal(
        horizon,
        tf.shape(controls)[-3],
        message="controls should have same horizon as measurements")

    tf.debugging.assert_equal(
        horizon,
        tf.shape(dynamics_state_jacobian)[-3],
        message="dynamics state jacobian covariance should have same horizon as measurements")

    tf.debugging.assert_equal(
        horizon,
        tf.shape(dynamics_controls_jacobian)[-3],
        message="dynamics controls jacobian should have same horizon as measurements")

    tf.debugging.assert_equal(
        horizon,
        tf.shape(measurement_jacobian)[-3],
        message="measurement jacobian should have same horizon as measurements")

    # flatten the batch shape

    measurements = tf.reshape(
        measurements,
        [batch_dim, horizon, measurement_dim, 1])

    initial_mean = tf.reshape(
        initial_mean,
        [batch_dim, state_dim, 1])

    initial_covariance = tf.reshape(
        initial_covariance,
        [batch_dim, state_dim, state_dim])

    controls = tf.reshape(
        controls,
        [batch_dim, horizon, controls_dim, 1])

    dynamics_state_jacobian = tf.reshape(
        dynamics_state_jacobian,
        [batch_dim, horizon, state_dim, state_dim])

    dynamics_controls_jacobian = tf.reshape(
        dynamics_controls_jacobian,
        [batch_dim, horizon, state_dim, controls_dim])

    dynamics_covariance = tf.reshape(
        dynamics_covariance,
        [batch_dim, state_dim, state_dim])

    measurement_jacobian = tf.reshape(
        measurement_jacobian,
        [batch_dim, horizon, measurement_dim, state_dim])

    measurement_covariance = tf.reshape(
        measurement_covariance,
        [batch_dim, measurement_dim, measurement_dim])

    # create the loop variables

    predicted_mean_array = tf.TensorArray(horizon)

    predicted_covariance_array = tf.TensorArray(horizon)

    innovation_array = tf.TensorArray(horizon)

    innovation_covariance_array = tf.TensorArray(horizon)

    mean_array = tf.TensorArray(horizon)

    covariance_array = tf.TensorArray(horizon)

    log_prob = 0.0

    time = tf.constant(0)

    # run the filter forward through time

    filter_results = tf.while_loop(
        kf_loop_condition,
        kf_loop_body, (
            measurements,
            initial_mean,
            initial_covariance,
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
            horizon))

    # stack the sequence of predictions from the filter

    predicted_mean = tf.transpose(
        filter_results[9].stack(), [1, 0, 2, 3])

    predicted_covariance = tf.transpose(
        filter_results[10].stack(), [1, 0, 2, 3])

    innovation = tf.transpose(
        filter_results[11].stack(), [1, 0, 2, 3])

    innovation_covariance = tf.transpose(
        filter_results[12].stack(), [1, 0, 2, 3])

    mean = tf.transpose(
        filter_results[13].stack(), [1, 0, 2, 3])

    covariance = tf.transpose(
        filter_results[14].stack(), [1, 0, 2, 3])

    log_prob = tf.transpose(
        filter_results[15], [1, 0])

    # create loop variables for smoothing

    smooth_gain_array = tf.TensorArray(horizon - 1)

    smooth_mean_array = tf.TensorArray(horizon).write(
        horizon - 1, mean[:, -1, :, :])

    smooth_covariance_array = tf.TensorArray(horizon).write(
        horizon - 1, covariance[:, -1, :, :])

    time = horizon - 2

    # run the smoothing update backwards through time

    smooth_results = tf.while_loop(
        kf_smooth_loop_condition,
        kf_smooth_loop_body, (
            mean,
            covariance,
            predicted_mean,
            predicted_covariance,
            mean[:, -1, :, :],
            covariance[:, -1, :, :],
            dynamics_state_jacobian,
            smooth_gain_array,
            smooth_mean_array,
            smooth_covariance_array,
            time))

    # stack the sequence of predictions from the filter

    smooth_gain = tf.transpose(
        smooth_results[7].stack(), [1, 0, 2, 3])

    smooth_mean = tf.transpose(
        smooth_results[8].stack(), [1, 0, 2, 3])

    smooth_covariance = tf.transpose(
        smooth_results[9].stack(), [1, 0, 2, 3])

    # create loop variables for em

    dynamics_covariance_array = tf.TensorArray(horizon - 2)

    measurement_covariance_array = tf.TensorArray(horizon - 2)

    time = horizon - 3

    # run the smoothing update backwards through time

    em_results = tf.while_loop(
        kf_em_loop_condition,
        kf_em_loop_body, (
            measurements,
            mean,
            covariance,
            controls,
            predicted_mean,
            predicted_covariance,
            smooth_gain,
            smooth_mean,
            smooth_covariance,
            dynamics_state_jacobian,
            dynamics_controls_jacobian,
            measurement_jacobian,
            dynamics_covariance_array,
            measurement_covariance_array,
            time))

    # stack the sequence of predictions from the filter

    dynamics_covariance = tf.reduce_mean(
        em_results[12].stack(), 0)

    measurement_covariance = tf.reduce_mean(
        em_results[13].stack(), 0)

    # process the results from the kalman filter

    predicted_mean = tf.reshape(
        predicted_mean,
        tf.concat([batch_shape, [horizon, state_dim]], 0))

    predicted_covariance = tf.reshape(
        predicted_covariance,
        tf.concat([batch_shape, [horizon, state_dim, state_dim]], 0))

    innovation = tf.reshape(
        innovation,
        tf.concat([batch_shape, [horizon, measurement_dim]], 0))

    innovation_covariance = tf.reshape(
        innovation_covariance,
        tf.concat([batch_shape, [horizon, measurement_dim, measurement_dim]], 0))

    mean = tf.reshape(
        mean,
        tf.concat([batch_shape, [horizon, state_dim]], 0))

    covariance = tf.reshape(
        covariance,
        tf.concat([batch_shape, [horizon, state_dim, state_dim]], 0))

    log_prob = tf.reshape(
        log_prob,
        tf.concat([batch_shape, [horizon]], 0))

    smooth_gain = tf.reshape(
        smooth_gain,
        tf.concat([batch_shape, [horizon, state_dim, state_dim]], 0))

    smooth_mean = tf.reshape(
        smooth_mean,
        tf.concat([batch_shape, [horizon, state_dim]], 0))

    smooth_covariance = tf.reshape(
        smooth_covariance,
        tf.concat([batch_shape, [horizon, state_dim, state_dim]], 0))

    dynamics_covariance = tf.reshape(
        dynamics_covariance,
        tf.concat([batch_shape, [state_dim, state_dim]], 0))

    measurement_covariance = tf.reshape(
        measurement_covariance,
        tf.concat([batch_shape, [measurement_dim, measurement_dim]], 0))

    return (
        predicted_mean,
        predicted_covariance,
        innovation,
        innovation_covariance,
        mean,
        covariance,
        log_prob,
        smooth_gain,
        smooth_mean,
        smooth_covariance,
        dynamics_covariance,
        measurement_covariance)
