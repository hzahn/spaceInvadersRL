import gym
import random
from collections import namedtuple
from time import time
import numpy as np
import mxnet as mx
from mxnet import gluon, autograd, nd
from mxnet.gluon import nn, rnn
from mxnet.gluon.loss import L2Loss
from PIL.Image import fromarray


# import matplotlib.pyplot as plt


MemRow = namedtuple('MemRow', ['action', 'observation', 'reward', 'done', 'next_observation'])


def create_model():
    net = nn.Sequential()
    with net.name_scope():
        net.add(
            nn.Conv2D(20, (3, 3), activation='relu'),
            #            nn.Dense(input_size, activation="relu"),
            # nn.Dense(256, activation="relu"),
            nn.Dense(256, activation="relu"),
            #            nn.Dense(256, activation="relu"),
            nn.Dense(6, activation='tanh')
        )
    net.initialize(init=mx.init.Uniform(scale=.1), force_reinit=True)
    return net


def run(env, model, games, batch_size, render=False):
    def get_expected_rewards(model, observation):
        reward_ = model(nd.ndarray.array(observation).swapaxes(2, 0).expand_dims(axis=0))[0]
        return reward_

    def get_best_action(model, observation, epsilon=0.1):
        expected_rewards = get_expected_rewards(model, observation)
        if np.random.random() <= epsilon or nd.sum(expected_rewards).asscalar() == 0:
            return np.random.randint(0, expected_rewards.shape[0]), True
        else:
            return int(expected_rewards.argmax(axis=0).asscalar()), False

    def calculate_q_values(action, rewards, observation, next_observations, dones, model, discount=0.9):
        actual_rewards = model(observation.swapaxes(3, 1))
        expected_rewards = model(next_observations.swapaxes(3, 1))
        best_reward = expected_rewards.max(axis=1)
        actual_rewards[:, action] = rewards + discount * best_reward * (1 - dones)
        return actual_rewards

    def train_model(model, memory_batch):
        observations = nd.ndarray.array([m.observation for m in memory_batch])
        next_observations = nd.ndarray.array([m.next_observation for m in memory_batch])
        rewards = nd.ndarray.array([m.reward for m in memory_batch])
        dones = nd.ndarray.array([1 * m.done for m in memory_batch])
        batch_size = len(memory_batch)
        actions = nd.ndarray.array([m.action for m in memory_batch])

        q_values = calculate_q_values(actions, rewards, observations, next_observations, dones, model)

        trainer = gluon.Trainer(model.collect_params(), 'sgd', {'learning_rate': 0.1})
        square_loss = L2Loss()

        # forward + backward
        with autograd.record():
            output = model(observations.swapaxes(3, 1))
            loss = square_loss(output, q_values)
        loss.backward()
        # update parameters
        trainer.step(batch_size)
        # calculate training metrics
        train_loss = loss.mean().asscalar()
        #        print(f'train batch: loss={train_loss}, time={time()-tic}')
        return train_loss

    def sample_batch(memory, batch_size):
        samples = min(batch_size, len(memory))
        return random.sample(memory, samples)

    def get_next_observations(n, env, action):
        next_observations = list()
        for i in range(n):
            next_observation, reward, done, info = env.step(action)
            next_observation = next_observation / 255


    def resize(obs, res):
        return np.array(fromarray(obs.astype('uint8')).resize(res))  # , PIL.Image.BILINEAR)

    def normalize_observation(observation):
        observation = observation / 255
        observation = resize(observation, (80, 80))
        return observation


    memory = []
    loss = []

    #    plt.ion()
    for i_episode in range(games):
        observation = normalize_observation(env.reset())
        t = 0
        rand_count = 0
        score = 0
        actions = []
        lives = 3
        while True:
            if render:
                env.render()
            action, rand_action = get_best_action(model, observation)
            actions.append(action)
            if rand_action:
                rand_count += 1

            (next_observations, rewards, dones, infos) = zip(*[env.step(action) for _ in range(3)])
            next_observation, reward, done, info = np.mean(next_observations, axis=0), sum(rewards), any(dones), infos[
                -1]
            # next_observation, reward, done, info = env.step(action)
            next_observation = normalize_observation(next_observation)
            score += reward
            reward = reward / 200
            if info['ale.lives'] < lives:
                lives = info['ale.lives']
                reward = -1
            memory.append(MemRow(observation=observation,
                                 reward=reward,
                                 done=done,
                                 next_observation=next_observation,
                                 action=action))
            observation = next_observation

            memory_batch = sample_batch(memory, batch_size)
            _loss = train_model(model, memory_batch)
            loss.append(_loss)

            if done:
                print("Episode {} finished after {} timesteps with reward {} and loss {} rnd {}".format(i_episode,
                                                                                                 t + 1,
                                                                                                 score,
                                                                                                 np.mean(loss), rand_count))
                # model.save_params('model_s.h5')
                break
            t += 1

        # plt.plot(loss, 'red')
        # plt.draw()
        # plt.pause(0.000001)

    env.close()
    # plt.show()


t = time()
env = gym.make('SpaceInvaders-v0')
observation_size = env.observation_space.shape[0]
model = create_model()
# model.load_params('model_s.h5')
run(env=env, model=model, games=150, batch_size=100, render=True)
# model.save_params('model_s.h5')
print(time() - t)
