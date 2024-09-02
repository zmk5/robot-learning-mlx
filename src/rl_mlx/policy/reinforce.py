"""REINFORCE model class for RL experiments with neural net function approx.

Written by: Zahi Kakish (zmk5)

"""
from __future__ import annotations

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim

import numpy as np


STATE = 0
ACTION = 1
REWARD = 2
NEXT_STATE = 3
NEXT_ACTION = 4
DONE = 5

 
class Reinforce():

    __slots__ = [
        "_n_vertices",
        "_n_states",
        "_n_actions",
        "_alpha",
        "_gamma",
        "_neural_net",
        "_optimizer",
        "_weights",
        "_hidden_layer_sizes",
        "_target_net",
        "_loss_function",
        "_gradients"
    ]

    def __init__(
        self,
        n_states: int,
        n_actions: int,
        alpha: float,
        gamma: float,
        hidden_layer_sizes: list[int]
    ) -> None:
        """Initialize the ModelREINFORCE class."""
        self._n_states = n_states
        self._n_actions = n_actions
        self._alpha = alpha
        self._gamma = gamma
        self._hidden_layer_sizes = hidden_layer_sizes

        # Check to make sure hidden layer sizes are correct.
        if len(hidden_layer_sizes) != 2:
            raise ValueError('Hidden layers must be a list of size 2!')

        # Set the neural net approximator
        self._neural_net = nn.Sequential([
            nn.Linear(
                self._n_states,
                hidden_layer_sizes[0],
            ),
            nn.ReLU(),
            nn.Linear(
                hidden_layer_sizes[0],
                hidden_layer_sizes[1],
            ),
            nn.ReLU(),
            nn.Linear(
                hidden_layer_sizes[1],
                self._n_actions
            ),
            nn.Softmax(),  # TODO figure out how to end with a softmax.
        ])

        # Set tensorflow loss function and optimizer
        self._optimizer = optim.Adam(learning_rate=self._alpha)
        self._loss_function = nn.losses.mse_loss

    def calculate_distribution_param(self, state) -> np.ndarray:
        """Calculate the probability distrbution parameters from neural net."""
        return self._neural_net(state, training=True)

    def calculate_returns(
        self,
        batch: tuple[mx.array],
        batch_size: int,
        previous_future_ret: float = 0.0,
    ) -> mx.array:
        """Calculate returns for each element in the sample batch."""
        ret_value = mx.array(np.zeros_like(batch[REWARD]))
        future_ret = previous_future_ret
        for t in reversed(range(batch_size + 1)):
            ret_value[t] = future_ret = batch[REWARD][t] + self._gamma * future_ret * (
                1 - batch[DONE][t]
            )

        return ret_value

    def calculate_policy_loss(
        self,
        batch: tuple[mx.array],
        dist_params: mx.array,
        ret_values: mx.array,
        batch_size: int,
    ) -> int | float:
        """Calculate the policy loss."""
        log_probs = mx.log(dist_params)
        idx = tf.Variable(
            np.append(
                np.arange(batch_size + 1).reshape(batch_size + 1, 1),
                batch[ACTION],
                axis=1,
            ),
            dtype=tf.int32,
        )
        act_log_probs = mx.reshape(tf.gather_nd(log_probs, idx), (batch_size + 1, 1))
        return -1 * self._alpha * tf.math.reduce_sum(act_log_probs * ret_values)

    def act(self, state: np.ndarray, epsilon: float | None = None) -> int | np.integer:
        """Apply the policy for a ROS inference service request."""
        _ = epsilon  # Unused by REINFORCE
        dist_parameters = self._neural_net(state)
        return mx.random.categorical(dist_parameters, 1)[0, 0].numpy()
    
    def train(self, batch: tuple[np.ndarray], batch_size: int = 16) -> None:
        """Train the policy based on a sample batch."""
        ret_values = self.calculate_returns(batch, batch_size)
        with tf.GradientTape() as tape:
            pd_params = self.calculate_distribution_param(batch[STATE])
            loss = self.calculate_policy_loss(batch, pd_params, ret_values, batch_size)

        # Calculate and apply graidents.
        self._gradients = tape.gradient(loss, self._neural_net.trainable_variables)
    
    def optimize(self, gradients: list[np.ndarray]) -> None:
        """Optimize global network policy."""
        self._optimizer.apply_gradients(
            zip(gradients, self._neural_net.trainable_variables))
