from __future__ import division
import gym
from gym.envs.registration import register
import numpy as np
import random, math, time
import copy
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties

register(
    id          ='FrozenLakeNotSlippery-v0',
    entry_point ='gym.envs.toy_text:FrozenLakeEnv',
    kwargs      ={'map_name' : '8x8', 'is_slippery': False},
)

def running_mean(x, N=20):
    cumsum = np.cumsum(np.insert(x, 0, 0))
    return (cumsum[N:] - cumsum[:-N]) / float(N)

LEFT = 0
DOWN = 1
RIGHT = 2
UP = 3

class Agent:
    def __init__(self, env):
        #hyperparameters
        self.stateCnt      = env.observation_space.n
        self.actionCnt     = env.action_space.n
        self.learning_rate = 0.475
        self.gamma         = 0.95
        self.epsilon       = 0.1
        self.Q             = self._initialiseModel()

    def _initialiseModel(self):
        return np.zeros([self.stateCnt ,self.actionCnt ])

    def predict_value(self, s):
        values = np.zeros(4)
        values = self.Q[s,:]
        return values


    def update_value_Qlearning(self, s,a,r,s_next, terminal_state):
        values = self.predict_value(s)

        if terminal_state:
            self.Q[s,a] = values[a] + self.learning_rate*(r - values[a])
        else:
            self.Q[s,a] = values[a] + self.learning_rate*(r + self.gamma * np.max(self.predict_value(s_next)) - values[a])


    def update_value_SARSA(self, s,a,r,s_next, a_next, terminal_state):
        values = self.predict_value(s)
        next_values = self.predict_value(s_next)

        if terminal_state:
            self.Q[s,a] = values[a] + self.learning_rate*(r - values[a])
        else:
            self.Q[s,a] = values[a] + self.learning_rate*(r + self.gamma * (next_values[a_next]) - values[a])


    def choose_action(self, s):
        action = np.argmax(self.predict_value(s)*(1-self.epsilon) + np.random.randn(1,self.actionCnt)*self.epsilon)
        return action


    def updateEpsilon(self, totalEpisodes):
        self.epsilon = self.epsilon - 1/(2*totalEpisodes)


class World:
    def __init__(self, env):
        self.env = env
        print('Environment has %d states and %d actions.' % (self.env.observation_space.n, self.env.action_space.n))
        self.stateCnt           = env.observation_space.n
        self.actionCnt          = env.action_space.n
        self.maxStepsPerEpisode = 100
        self.q_Sinit_progress   = np.array([[0,0,0,0]])

    def run_episode_qlearning(self):
        s               = self.env.reset() # "reset" environment to start state
        r_total         = 0
        episodeStepsCnt = 0
        success         = False
        for i in range(self.maxStepsPerEpisode):
            a = agent.choose_action(s)
            s_next, r, terminal_state, info = self.env.step(a)
            r_total = r_total + r
            episodeStepsCnt += 1
            agent.update_value_Qlearning(s,a,r,s_next,terminal_state)
            if i==0:
                self.q_Sinit_progress = np.append(self.q_Sinit_progress, [agent.predict_value(s)],axis=0)
            s = s_next
            env.render()

            if terminal_state:
                break

        return r_total, episodeStepsCnt


    def run_episode_sarsa(self):
        s               = self.env.reset()
        r_total         = 0
        episodeStepsCnt = 0
        success         = False
        for i in range(self.maxStepsPerEpisode):
            a = agent.choose_action(s)
            s_next, r, terminal_state, info = self.env.step(a)
            a_next = agent.choose_action(s_next)
            r_total = r_total + r
            episodeStepsCnt += 1
            agent.update_value_SARSA(s,a,r,s_next,a_next,terminal_state)
            if i==0:
                self.q_Sinit_progress = np.append(self.q_Sinit_progress, [agent.predict_value(s)], axis=0)
            s = s_next
            env.render()

            if terminal_state:
                break

        return r_total, episodeStepsCnt


    def run_evaluation_episode(self):
        agent.epsilon = 0
        agent.gamma = 0
        s = self.env.reset()
        for i in range(self.maxStepsPerEpisode):
            s_next, r, success, info = self.env.step(agent.choose_action(s))
            np.append(self.q_Sinit_progress, [agent.predict_value(s)],axis=0)
            s = s_next
            if success:
                break



if __name__ == '__main__':
    env                      = gym.make('FrozenLakeNotSlippery-v0')
    world                    = World(env)
    agent                    = Agent(env)
    r_total_progress         = []
    episodeStepsCnt_progress = []
    nbOfTrainingEpisodes     = 10000
    for i in range(nbOfTrainingEpisodes):
        print('\n========================\n   Episode: {}\n========================'.format(i))
        r_total, episodeStepsCnt = world.run_episode_qlearning()
        r_total_progress.append(r_total)
        episodeStepsCnt_progress.append(episodeStepsCnt)
        #agent.updateEpsilon(nbOfTrainingEpisodes)

    # run_evaluation_episode
    world.run_evaluation_episode()

    ## Plots
    # 1) plot world.q_Sinit_progress
    fig1 = plt.figure(1)
    plt.ion()
    plt.plot(world.q_Sinit_progress[:,0], label='left',  color = 'r')
    plt.plot(world.q_Sinit_progress[:,1], label='down',  color = 'g')
    plt.plot(world.q_Sinit_progress[:,2], label='right', color = 'b')
    plt.plot(world.q_Sinit_progress[:,3], label='up',    color = 'y')
    fontP = FontProperties()
    fontP.set_size('small')
    plt.legend(prop = fontP, loc=1)
    plt.pause(0.001)

    # 2) plot the evolution of the number of steps per successful episode throughout training. A successful episode is an episode where the agent reached the goal (i.e. not any terminal state)
    fig2 = plt.figure(2)
    plt1 = plt.subplot(1,2,1)
    plt1.set_title("Number of steps per successful episode")
    plt.ion()
    plt.plot(episodeStepsCnt_progress)
    plt.pause(0.0001)
    # 3) plot the evolution of the total collected rewards per episode throughout training. you can use the running_mean function to smooth the plot
    plt2 = plt.subplot(1,2,2)
    plt2.set_title("Rewards collected per episode")
    plt.ion()
    r_total_progress = running_mean(r_total_progress)
    plt.plot(r_total_progress)
    plt.pause(0.0001)
