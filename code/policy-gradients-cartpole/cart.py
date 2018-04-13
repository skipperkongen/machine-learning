import gym
import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
import sys
from collections import deque

class PGAgent:
    def __init__(self, state_size, action_size):
        """OK"""
        self.state_size = state_size
        self.action_size = action_size
        self.gamma = 0.99
        self.learning_rate = 1e-4
        self.xs = []
        self.gradients = []
        self.rewards = []
        self.probs = []
        self.model = self._build_model()
        self.model.summary()

    def _build_model(self):
        """OK"""
        model = Sequential()
        model.add(Dense(32, input_shape=(self.state_size,), activation='relu', kernel_initializer='he_uniform'))
        model.add(Dense(32, activation='relu', kernel_initializer='he_uniform'))
        model.add(Dense(self.action_size, activation='softmax'))
        opt = Adam(lr=self.learning_rate)
        model.compile(loss='categorical_crossentropy', optimizer=opt)
        return model

    def act(self, x):
        """OK"""
        # reshape to nested array
        x = x.reshape([1, x.size])
        # predict the actions probabilities with a forward propagation
        prob = self.model.predict(x, batch_size=1).flatten()
        # select an action with respect to the probabilities obtained
        action = np.random.choice(self.action_size, 1, p=prob)[0]
        return action, prob

    def discount_rewards(self, rewards):
        """OK"""
        discounted_rewards = np.zeros_like(rewards)
        running_add = 0
        for t in reversed(range(0, rewards.size)):
            running_add = running_add * self.gamma + rewards[t]
            discounted_rewards[t] = running_add
        return discounted_rewards

    # CHECKED

    def remember(self, x, action, prob, reward):
        # remember x
        self.xs.append(x)
        # remember gradient
        y = np.zeros([self.action_size])
        y[action] = 1
        self.gradients.append(y - prob)
        # remember probability
        self.probs.append(prob)
        # remember rewards
        self.rewards.append(reward)

    def train(self):
        gradients = np.vstack(self.gradients)
        rewards = np.vstack(self.rewards)
        # discount the rewards
        rewards = self.discount_rewards(rewards)
        # normalize the rewards
        rewards = rewards / np.std(rewards - np.mean(rewards))
        gradients *= rewards
        X = np.squeeze(np.vstack([self.xs]))
        Y = self.probs + self.learning_rate * np.squeeze(np.vstack([gradients]))
        self.model.train_on_batch(X, Y)
        self.xs, self.probs, self.gradients, self.rewards = [], [], [], []

    def load(self, name):
        self.model.load_weights(name)

    def save(self, name):
        self.model.save_weights(name)

if __name__ == "__main__":
    env = gym.make("CartPole-v0")
    x = env.reset()
    score = 0
    episode = 0

    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n
    agent = PGAgent(state_size, action_size)
    #agent.load('cart.h5')
    scores = deque(maxlen=100)
    while True:
        #env.render()
        action, prob = agent.act(x)
        #print(action, prob)
        x, reward, done, _ = env.step(action)
        if done:
            reward = 0
        score += reward
        agent.remember(x, action, prob, reward)
        if done:
            episode += 1
            agent.train()
            scores.append(score)
            print('Episode: %d - Score: %f - Average: %f.' % (episode, score, np.mean(scores)))
            score = 0
            x = env.reset()
            if episode > 1 and episode % 10 == 0:
                agent.save('cart.h5')
