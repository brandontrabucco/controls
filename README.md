# Controls

Controls implements differentiable optimal controls in TensorFlow 2.0. Have Fun! -Brandon

# Setup

Install this package using pip.

```
pip install git+git://github.com/brandontrabucco/controls.git
```

# Usage

Collect a batch of initial states to use for planning

```
initial_states = <...your code here...>
```

Create a dynamics model that predicts into the future.

```
A = tf.constant([[[-0.313, 56.7, 0.0],
                  [-0.0139, -0.426, 0.0],
                  [0.0, 56.7, 0.0]]])

B = tf.constant([[[0.232],
                  [0.0203],
                  [0.0]]])

dynamics_model = controls.Linear(0, [0, 0], [A, B])
```

Create a cost model that evaluates the cost of states and controls.

```
Q = tf.constant([[[0.0, 0.0, 0.0],
                  [0.0, 0.0, 0.0],
                  [0.0, 0.0, 1.0]]])

R = tf.constant([[[1.0]]])

cost_model = controls.Quadratic(0, [0, 0], [0, 0], [[Q, 0], [0, R]])
```

Define an initial policy where the optimizer starts.

```
controls_model = controls.Zeros(1)
```

Launch the optimizer to get a new policy.

```
controls_model = controls.iterative_lqr(
    initial_states,
    controls_model,
    dynamics_model,
    cost_model,
    h=20,
    n=10,
    a=0.1,
    random=False)
```
