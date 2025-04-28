import tensorflow as tf
import numpy as np
import random
import os
from datetime import datetime

class DQN(tf.keras.Model):
    def __init__(self, input_dim, output_dim):
        super(DQN, self).__init__()
        self.fc1 = tf.keras.layers.Dense(512, activation='relu')
        self.fc2 = tf.keras.layers.Dense(256, activation='relu')
        self.fc3 = tf.keras.layers.Dense(128, activation='relu')
        self.fc4 = tf.keras.layers.Dense(64, activation='relu')
        self.fc5 = tf.keras.layers.Dense(output_dim, activation=None)
    
    def call(self, x):
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        x = self.fc4(x)
        return self.fc5(x)

class DQNAgent:
    def __init__(self, state_dim, action_dim, memory_size=10000, batch_size=64, gamma=0.99, lr=0.001, target_update=10):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.memory = PrioritizedReplayMemory(memory_size, state_dim)
        self.batch_size = batch_size
        self.gamma = gamma
        self.lr = lr
        self.target_update = target_update

        self.policy_net = DQN(state_dim, action_dim)
        self.target_net = DQN(state_dim, action_dim)
        self.target_net.set_weights(self.policy_net.get_weights())
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=lr)
        self.loss_fn = tf.keras.losses.Huber(reduction=tf.keras.losses.Reduction.NONE)
        self.steps_done = 0
        self.saved_models = []  # List to store saved model filenames

    def select_action(self, state, epsilon):
        if random.random() < epsilon:
            return random.randrange(self.action_dim)
        state = tf.convert_to_tensor(state, dtype=tf.float32)
        state = tf.expand_dims(state, axis=0)
        q_values = self.policy_net(state)
        return tf.argmax(q_values, axis=1).numpy()[0]

    def store_transition(self, state, action, reward, next_state, done):
        self.memory.push(state, action, reward, next_state, done)

    def update_target_net(self):
        self.target_net.set_weights(self.policy_net.get_weights())

        def optimize_model(self, beta=0.4):
            if len(self.memory) < self.batch_size:
                return None
            transitions, indices, weights = self.memory.sample(self.batch_size, beta)
            state_batch, action_batch, reward_batch, next_state_batch, done_batch = transitions

            with tf.GradientTape() as tape:
                state_action_values = tf.reduce_sum(
                    self.policy_net(state_batch) * tf.one_hot(action_batch, self.action_dim), axis=1, keepdims=True)

                next_state_values = tf.reduce_max(self.target_net(next_state_batch), axis=1, keepdims=True)
                expected_state_action_values = reward_batch + (1.0 - done_batch) * self.gamma * next_state_values

                td_errors = self.loss_fn(state_action_values, expected_state_action_values)
                loss = tf.reduce_mean(td_errors * weights)

            grads = tape.gradient(loss, self.policy_net.trainable_variables)
            self.optimizer.apply_gradients(zip(grads, self.policy_net.trainable_variables))

            priorities = tf.reshape(td_errors, [-1]).numpy() + 1e-5
            self.memory.update_priorities(indices, priorities)

            return loss.numpy()


    def save_model(self, episode):
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        filename = f"model_ep{episode}_{timestamp}.weights.h5"
        model_directory = "./models"
        if not os.path.exists(model_directory):
            os.makedirs(model_directory)
        filepath = os.path.join(model_directory, filename)
        self.policy_net.save_weights(filepath)
        self.saved_models.append(filepath)
        print(f"Model saved as {filename}")
        self.manage_saved_models()

    def manage_saved_models(self):
        if len(self.saved_models) > 3:
            oldest_model = self.saved_models.pop(0)
            if os.path.exists(oldest_model):
                os.remove(oldest_model)
                print(f"Removed oldest model: {oldest_model}")

    def load_model(self, filename):
        self.policy_net.load_weights(filename)
        self.update_target_net()
        print(f"Model loaded from {filename}")

class PrioritizedReplayMemory:
    def __init__(self, capacity, state_dim, alpha=0.6):
        self.capacity = capacity
        self.position = 0
        self.alpha = alpha

        self.states = np.zeros((capacity, state_dim), dtype=np.float32)
        self.actions = np.zeros((capacity, 1), dtype=np.int64)
        self.rewards = np.zeros((capacity, 1), dtype=np.float32)
        self.next_states = np.zeros((capacity, state_dim), dtype=np.float32)
        self.dones = np.zeros((capacity, 1), dtype=np.float32)
        self.priorities = np.zeros((capacity,), dtype=np.float32)

    def push(self, state, action, reward, next_state, done):
        max_priority = self.priorities.max() if self.position > 0 else 1.0
        self.states[self.position] = state
        self.actions[self.position] = action
        self.rewards[self.position] = reward
        self.next_states[self.position] = next_state
        self.dones[self.position] = done
        self.priorities[self.position] = max_priority
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size, beta=0.4):
        if self.position == self.capacity:
            priorities = self.priorities
        else:
            priorities = self.priorities[:self.position]
        probabilities = priorities ** self.alpha
        probabilities /= probabilities.sum()

        indices = np.random.choice(len(probabilities), batch_size, p=probabilities)
        weights = (len(probabilities) * probabilities[indices]) ** (-beta)
        weights /= weights.max()

        state_batch = tf.convert_to_tensor(self.states[indices], dtype=tf.float32)
        action_batch = tf.convert_to_tensor(self.actions[indices], dtype=tf.int64)
        reward_batch = tf.convert_to_tensor(self.rewards[indices], dtype=tf.float32)
        next_state_batch = tf.convert_to_tensor(self.next_states[indices], dtype=tf.float32)
        done_batch = tf.convert_to_tensor(self.dones[indices], dtype=tf.float32)

        return (state_batch, action_batch, reward_batch, next_state_batch, done_batch), indices, weights

    def update_priorities(self, batch_indices, batch_priorities):
        self.priorities[batch_indices] = batch_priorities

    def __len__(self):
        return min(self.position, self.capacity)

