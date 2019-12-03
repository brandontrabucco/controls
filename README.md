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

def dynamics_model(time, inputs):
    return A @ inputs[0] + B @ inputs[1]

def cost_model(time, inputs):
    return (tf.matmul(tf.matmul(inputs[0], Q, transpose_a=True), inputs[0]) + 
            tf.matmul(tf.matmul(inputs[1], R, transpose_a=True), inputs[1])) / 2.
```

Define your initial policy where the optimizer starts.

```
def controls_model(time, inputs):
    return tf.zeros([1, 1, 1])
```

Launch the optimizer to get a new policy.

```
controls_model = cem(
    initial_states,
    controls_model,
    dynamics_model,
    cost_model,
    h=20,
    c=1000,
    n=100,
    k=100,
    s=0.5)
```
