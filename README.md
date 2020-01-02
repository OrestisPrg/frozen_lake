# Reinforcement Learning: Frozen Lake Environment (OpenAI Gym)

This project uses the frozen lake environment from the OpenAI Gym library to explore and experiment with reinforcement learning and 2 reinforcement learning algorithms. The code includes implementations of both SARSA and Q-Learning algorithms and produces 3 graphs visualising the results.

The purpose of the project is to experiment with the different hyperparameters and compare the results that are produced by trying different values for each, in an attempt to determine the optimal value (or range of values) for each parameter.

*The sample code for plotting the graphs was provided by the University of Leeds.*

In order to run the file *frozen_lake.py* you need install **gym** library. You can do this in a virtual environment with the following steps:

1. Create a new environment with `virtualenv env`
2. Activate the environment with `source env/bin/activate`
3. Install gym `pip install gym`
4. Install matplotlib `pip install matplotlib`

# RL Hyperparameters:
* Exploration parameter (`epsilon`)
* Discount factor (`gamma`)
* Learning rate (`learning_rate`)
