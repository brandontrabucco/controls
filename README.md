# Controls

This package implements differentiable optimal controls in TensorFlow 2.0. Have Fun! -Brandon

# Setup

Install this package directly from github using pip.

```
pip install git+git://github.com/brandontrabucco/controls.git
```

# Usage

Import the control algorithm you want to use.

```
from controls import cem
```

Collect a batch of initial states.

```
initial_states = tf.random.normal([1, 3, 1])
```

Define your dynamics model and cost model.

```
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

def dynamics_model(x):
    return A @ x[0] + B @ x[1]

def cost_model(x):
    return (tf.matmul(tf.matmul(x[0], Q, transpose_a=True), x[0]) + 
            tf.matmul(tf.matmul(x[1], R, transpose_a=True), x[1])) / 2.
```

Define your initial policy where the optimizer starts.

```
def controls_model(x):
    return tf.zeros([tf.shape(x[0])[0], 1, 1])
```

Launch the optimizer to get a new policy.

```
controls_model = cem(
    initial_states,
    controls_model,
    dynamics_model,
    cost_model,
    horizon=20,
    num_candidates=1000,
    num_iterations=100,
    top_k=100,
    exploration_noise_std=1.0)
```
