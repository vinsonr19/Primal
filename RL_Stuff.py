import tensorflow as tf
# import tensorflow_probability as tfp
from tensorflow import keras
from tensorflow.keras import layers
from numpy.random import choice
import numpy as np
from constants import *

class ActorCriticModel(keras.Model):
    def __init__(self, state_shape, goal_shape, num_actions, min_std=1e-5):
        super(ActorCriticModel, self).__init__()

        self.global_step = tf.Variable(0, trainable=False)
        # boundaries = [SUPERVISED_CUTOFF]
        # values = [LEARNING_RATE, LEARNING_RATE/2]
        # learning_rate = tf.compat.v1.train.piecewise_constant(self.global_step, boundaries,
        # values)
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE)#learning_rate)


        # fancy input stuff
        self.convlayer1 = layers.Conv2D(filters=128, kernel_size=(3,3), strides=(1,1), padding='same', input_shape=(None, *state_shape), activation='swish')
        self.convlayer2 = layers.Conv2D(filters=128, kernel_size=(3,3), strides=(1,1), padding='same', activation='swish')
        self.pool1 = layers.MaxPooling2D(pool_size=(2,2))
        self.convlayer3 = layers.Conv2D(filters=128, kernel_size=(3,3), strides=(1,1), padding='same', activation='swish')
        self.convlayer4 = layers.Conv2D(filters=64, kernel_size=(3,3), strides=(1,1), padding='same', activation='swish')
        self.pool2 = layers.MaxPooling2D(pool_size=(2,2))
        self.convlayer5 = layers.Conv2D(filters=64, kernel_size=(3,3), strides=(1,1), padding='same', activation='swish')
        self.flattenlayer = layers.Flatten()

        self.dropout = layers.Dropout(0.1) # get rid of info to speed up our lives! and overfitting blah blah blah


        # goal input
        self.goal_layer1 = layers.Dense(12, input_shape=(None, *goal_shape), activation='swish')
        self.goal_layer2 = layers.Dense(12, activation='swish')


        # post concatenation
        self.dense1 = layers.Dense(128, activation='swish')
        self.dense2 = layers.Dense(128, activation='swish')
        # self.LSTM = layers.LSTM(128, activation='swish', recurrent_activation='swish', return_sequences=False)
        self.dense3 = layers.Dense(128, activation='swish')


        # policy output
        self.policy_dense1 = layers.Dense(128, activation='swish')
        self.policy_dense2 = layers.Dense(64, activation='swish')
        self.policy_output = layers.Dense(num_actions, activation='softmax')

        # value output
        self.value_dense1 = layers.Dense(128, activation='swish')
        self.value_dense2 = layers.Dense(64, activation='swish')
        self.value_output = layers.Dense(1, activation='linear')

        # on_goal output
        # self.goal_output = layers.Dense(1, activation='sigmoid')



    def call(self, inputs):
        conv_inputs = tf.convert_to_tensor(inputs[0])
        conv = self.convlayer1(conv_inputs)
        conv = self.convlayer2(conv)
        conv = self.pool1(conv)
        conv = self.convlayer3(conv)
        conv = self.convlayer4(conv)
        conv = self.pool2(conv)
        conv = self.convlayer5(conv)
        conv = self.flattenlayer(conv)
        conv = self.dropout(conv)

        goal_inputs = tf.convert_to_tensor(inputs[1])
        goal = self.goal_layer1(goal_inputs)
        goal = self.goal_layer2(goal)

        # print(conv.shape)
        # print(goal.shape)
        concatenated = layers.Concatenate(axis=1)([conv, goal])
        dense = self.dense1(concatenated)
        dense = self.dense2(dense)
        # dense = self.LSTM(dense)
        dense = self.dense3(dense)

        policy = self.policy_dense1(dense)
        policy = self.policy_dense2(policy)
        policy = self.policy_output(policy)

        value = self.value_dense1(dense)
        value = self.value_dense2(value)
        value = self.value_output(value)

        # goal = self.goal_output(dense)

        return policy, value#, goal

    # @tf.function
    def train(self, states, returns, advantages, is_valid, cur_goals):#, goals, goal_guesses):
        self.global_step = self.global_step + 1
        count_steps = 0
        # PPO EPOCHS is the number of times we will go through ALL the training data to make updates
        for _ in range(PPO_EPOCHS):
            # grabs random mini-batches several times until we have covered all data

            # for state, action, return_, advantage, is_valid_, value, goal, goal_guess in self._ppo_iterator(states, actions, returns, advantages, is_valid, values, goals, goal_guesses):
            for state, return_, advantage, is_valid_, cur_goal in self._ppo_iterator(states, returns, advantages, is_valid, cur_goals):

                with tf.GradientTape() as tape:
                    tape.watch(self.trainable_variables)   # keep track of the trainable variables (don't always need all of them)
                    loss = self._get_loss(state, return_, advantage, is_valid_, cur_goal)#, goal, goal_guess)
                    grads = tape.gradient(loss, self.trainable_variables)  # get_gradients (backprop) from losses

                # https://github.com/openai/baselines/blob/ea25b9e8b234e6ee1bca43083f8f3cf974143998/baselines/ppo2/model.py#L102-L108
                grads, _ = tf.clip_by_global_norm(grads, MAX_GRAD_NORM)
                self.optimizer.apply_gradients(zip(grads, self.trainable_variables))  # change weights based on gradients



    def sample(self, policy):
        # print(policy)
        return choice(num_actions, 1, p=policy.numpy().squeeze())



    def _ppo_iterator(self, states, returns, advantage, is_valid, cur_goals):#, goals, goal_guesses):
        batch_size = len(states)

        # generates random mini-batches until we have covered the full batch
        for _ in range(batch_size // MINI_BATCH_SIZE):
            rand_ids = tf.convert_to_tensor(np.random.randint(0, batch_size, MINI_BATCH_SIZE))
            yield tf.gather(tf.convert_to_tensor(states), rand_ids), \
                  tf.gather(tf.convert_to_tensor(returns), rand_ids), tf.gather(tf.convert_to_tensor(advantage), rand_ids), tf.gather(tf.convert_to_tensor(is_valid), rand_ids), \
                  tf.gather(tf.convert_to_tensor(cur_goals), rand_ids)
                  #\
#                  tf.gather(tf.convert_to_tensor(goals), rand_ids), tf.gather(tf.convert_to_tensor(goal_guesses), rand_ids)



    def _get_loss(self, state, return_, advantage, is_valid, cur_goal):#, on_goal, goal_guess):

        # breakpoint()

        state = tf.reshape(state, [32,30,30,7])
        cur_goal = tf.reshape(cur_goal, [32,2])

        policy, value = self.call([state, cur_goal])
        l_value = tf.reduce_mean(tf.pow(return_ - value, 2))

        l_entropy = -tf.reduce_mean(policy*tf.math.log(tf.clip_by_value(policy, 1e-10, 1)))

        l_policy = -tf.reduce_mean(tf.math.log(tf.clip_by_value(tf.convert_to_tensor(np.array([[np.max(i)] for i in policy])), 1e-10, 1.0)) * advantage)

        legal_policy = tf.cast(tf.convert_to_tensor(is_valid), tf.float32)
        valid_policys = tf.clip_by_value(policy, 1e-10, 1.0-1e-10)
        l_valid = -tf.reduce_mean(tf.math.log(valid_policys)*legal_policy + tf.math.log(1-valid_policys) * (1-legal_policy)) # cross entropy for illegal moves

        # l_goal = -tf.reduce_mean(on_goal*tf.math.log(tf.clip_by_value(goal_guess,1e-10,1.0))\
        #                          +(1-on_goal)*tf.math.log(tf.clip_by_value(1-goal_guess,1e-10,1.0)))

        loss = (CRITIC_DISCOUNT*l_value + l_policy + VALID_DISCOUNT*l_valid - ENTROPY_BETA*l_entropy)/10# + ON_GOAL_DISCOUNT*l_goal # + BLOCKING_DISCOUNT*l_blocking

        return loss


    def train_imitation(self, cur_states, cur_goals, cur_actions, returns):
        self.global_step = self.global_step + 1
        count_steps = 0

        # PPO EPOCHS is the number of times we will go through ALL the training data to make updates
        for _ in range(PPO_EPOCHS):
            # grabs random mini-batches several times until we have covered all data
            for state, cur_goal, correct_action, return_ in self._ppo_imitation_iterator(cur_states, cur_goals, cur_actions, returns):

                with tf.GradientTape() as tape:
                    tape.watch(self.trainable_variables)   # keep track of the trainable variables (don't always need all of them)
                    loss = self._get_imitation_loss(state, cur_goal, correct_action, return_)
                    grads = tape.gradient(loss, self.trainable_variables)  # get_gradients (backprop) from losses

                # https://github.com/openai/baselines/blob/ea25b9e8b234e6ee1bca43083f8f3cf974143998/baselines/ppo2/model.py#L102-L108
                grads, _ = tf.clip_by_global_norm(grads, MAX_GRAD_NORM)
                self.optimizer.apply_gradients(zip(grads, self.trainable_variables))  # change weights based on gradients


    def _get_imitation_loss(self, state, cur_goal, correct_action, return_):

        state = tf.reshape(state, [32,30,30,7])
        cur_goal = tf.reshape(cur_goal, [32,2])
        policy, value = self.call([state, cur_goal])

        useable_actions = np.array([[0 if j!=correct_action[i] else 1 for j in range(5)] for i in range(correct_action.shape[0])])
        l_imitation = tf.reduce_mean(keras.losses.categorical_crossentropy(useable_actions, policy))

        l_value = tf.reduce_mean(tf.pow(return_ - value, 2))

        return CRITIC_DISCOUNT*l_value + l_imitation


    def _ppo_imitation_iterator(self, cur_states, cur_goals, cur_actions, returns):
        batch_size = len(cur_states)
        # generates random mini-batches until we have covered the full batch
        for _ in range(batch_size // MINI_BATCH_SIZE):
            rand_ids = tf.convert_to_tensor(np.random.randint(0, batch_size, MINI_BATCH_SIZE))
            yield tf.gather(cur_states, rand_ids), tf.gather(cur_goals, rand_ids), \
                  tf.gather(cur_actions, rand_ids), tf.gather(returns, rand_ids)



def normalize(x):
    x -= np.mean(x)
    x /= (np.std(x) + 1e-8)
    return x

def compute_gae(next_value, rewards, masks, values, gamma=GAMMA, lam=GAE_LAMBDA):
    values = values + [next_value]
    gae = 0
    returns = []
    for step in reversed(range(len(rewards))):
        delta = rewards[step] + (gamma * values[step + 1] * masks[step] - values[step]) # r+td_error
        gae = delta + gamma * lam * masks[step] * gae
        # prepend to get correct order back
        returns.insert(0, gae + values[step])
    return returns
