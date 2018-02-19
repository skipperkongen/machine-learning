import gym
import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
import sys

class PGAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.gamma = 0.99
        self.learning_rate = 1e-4
        self.states = []
        self.gradients = []
        self.rewards = []
        self.probs = []
        self.model = self._build_model()
        self.model.summary()

    def _build_model(self):
        model = Sequential()
        model.add(Dense(8, input_shape=(self.state_size,), activation='relu', kernel_initializer='he_uniform'))
        model.add(Dense(self.action_size, activation='softmax'))
        opt = Adam(lr=self.learning_rate)
        model.compile(loss='categorical_crossentropy', optimizer=opt)
        return model

    def remember(self, state, action, prob, reward):
        y = np.zeros([self.action_size])
        # one-hot vector for the action taken
        y[action] = 1
        # compute the gradient with respect to probabilities in
        #   forward prop. and action taken
        self.gradients.append(np.array(y).astype('float32') - prob)
        # append current states and rewards
        self.states.append(state)
        self.rewards.append(reward)

    def act(self, state):
        # reshape from 6400 to (1, 6400)
        state = state.reshape([1, state.shape[0]])
        # predict the actions probabilities with a forward propagation
        aprob = self.model.predict(state, batch_size=1).flatten()
        self.probs.append(aprob)
        # select an action with respect to the probabilities obtained
        action = np.random.choice(self.action_size, 1, p=aprob)[0]
        return action, aprob

    def discount_rewards(self, rewards):
        discounted_rewards = np.zeros_like(rewards)
        running_add = 0
        for t in reversed(range(0, rewards.size)):
            if rewards[t] != 0:
                running_add = 0
            running_add = running_add * self.gamma + rewards[t]
            discounted_rewards[t] = running_add
        return discounted_rewards


    def train(self):
        gradients = np.vstack(self.gradients)
        rewards = np.vstack(self.rewards)
        # discount the rewards
        rewards = self.discount_rewards(rewards)
        # normalize the rewards
        rewards = (rewards - np.mean(rewards)) / (np.std(rewards) or 1.0)
        gradients *= rewards
        X = np.squeeze(np.vstack([self.states]))
        Y = self.probs + self.learning_rate * np.squeeze(np.vstack([gradients]))
        self.model.train_on_batch(X, Y)
        self.states, self.probs, self.gradients, self.rewards = [], [], [], []

    def load(self, name):
        self.model.load_weights(name)

    def save(self, name):
        self.model.save_weights(name)

if __name__ == "__main__":
    env = gym.make("CartPole-v0")
    state = env.reset()
    score = 0
    episode = 0

    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n
    agent = PGAgent(state_size, action_size)
    #agent.load('cart.h5')
    while True:
        env.render()

        action, prob = agent.act(state)
        #print(action, prob)
        state, reward, done, info = env.step(action)
        score += reward
        agent.remember(state, action, prob, reward)

        if done:
            episode += 1
            agent.train()
            print('Episode: %d - Score: %f.' % (episode, score))
            score = 0
            state = env.reset()
            if episode > 1 and episode % 10 == 0:
                agent.save('cart.h5')
            #sys.exit()
