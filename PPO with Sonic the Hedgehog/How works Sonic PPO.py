#!/usr/bin/env python
# coding: utf-8

# # Understand PPO implementation playing Sonic the Hedgehog 2 and 3 🦔
# 
# This Notebook explains the Proximal Policy Optimization implementation.<br>
# The [repository link]() 
# 
# ### Acknowledgements 👏
# This implementation is based on 2 repositories:
# - OpenAI [Baselines PPO2](https://github.com/openai/baselines/blob/24fe3d6576dd8f4cdd5f017805be689d6fa6be8c/baselines/ppo2/ppo2.py)
# - Alexandre Borghi [retro_contest_agent](https://github.com/aborghi)

# <img src="assets/PPO.png" alt="PPO"/>

# # This is a notebook from [Deep Reinforcement Learning Course with Tensorflow](https://simoninithomas.github.io/Deep_reinforcement_learning_Course/)
# <img src="https://raw.githubusercontent.com/simoninithomas/Deep_reinforcement_learning_Course/master/docs/assets/img/DRLC%20Environments.png" alt="Deep Reinforcement Course"/>
# <br>
# <p>  Deep Reinforcement Learning Course is a free series of articles and videos tutorials 🆕 about Deep Reinforcement Learning, where **we'll learn the main algorithms (Q-learning, Deep Q Nets, Dueling Deep Q Nets, Policy Gradients, A2C, Proximal Policy Gradients…), and how to implement them with Tensorflow.**
# <br><br>
#     
# 📜The articles explain the architectures from the big picture to the mathematical details behind them.
# <br>
# 📹 The videos explain how to build the agents with Tensorflow </b></p>
# <br>
# This course will give you a **solid foundation for understanding and implementing the future state of the art algorithms**. And, you'll build a strong professional portfolio by creating **agents that learn to play awesome environments**: Doom© 👹, Space invaders 👾, Outrun, Sonic the Hedgehog©, Michael Jackson’s Moonwalker, agents that will be able to navigate in 3D environments with DeepMindLab (Quake) and able to walk with Mujoco. 
# <br><br>
# </p> 
# 
# ## 📚 The complete [Syllabus HERE](https://simoninithomas.github.io/Deep_reinforcement_learning_Course/)
# 
# 
# ## Any questions 👨‍💻
# <p> If you have any questions, feel free to ask me: </p>
# <p> 📧: <a href="mailto:hello@simoninithomas.com">hello@simoninithomas.com</a>  </p>
# <p> Github: https://github.com/simoninithomas/Deep_reinforcement_learning_Course </p>
# <p> 🌐 : https://simoninithomas.github.io/Deep_reinforcement_learning_Course/ </p>
# <p> Twitter: <a href="https://twitter.com/ThomasSimonini">@ThomasSimonini</a> </p>
# <p> Don't forget to <b> follow me on <a href="https://twitter.com/ThomasSimonini">twitter</a>, <a href="https://github.com/simoninithomas/Deep_reinforcement_learning_Course">github</a> and <a href="https://medium.com/@thomassimonini">Medium</a> to be alerted of the new articles that I publish </b></p>
#     
# ## How to help  🙌
# 3 ways:
# - **Clap our articles and like our videos a lot**:Clapping in Medium means that you really like our articles. And the more claps we have, the more our article is shared Liking our videos help them to be much more visible to the deep learning community.
# - **Share and speak about our articles and videos**: By sharing our articles and videos you help us to spread the word. 
# - **Improve our notebooks**: if you found a bug or **a better implementation** you can send a pull request.
# <br>
# 
# ## Important note 🤔
# <b> You can run it on your computer but it's better to run it on GPU based services</b>, personally I use Microsoft Azure and their Deep Learning Virtual Machine (they offer 170$)
# https://azuremarketplace.microsoft.com/en-us/marketplace/apps/microsoft-ads.dsvm-deep-learning
# <br>
# ⚠️ I don't have any business relations with them. I just loved their excellent customer service.
# 
# If you have some troubles to use Microsoft Azure follow the explainations of this excellent article here (without last the part fast.ai): https://medium.com/@manikantayadunanda/setting-up-deeplearning-machine-and-fast-ai-on-azure-a22eb6bd6429

# ## How to use it? 📖
# ⚠️ First you need to follow Step 1 (Download Sonic the Hedgehog series)
# ### Watch our agent playing 👀
# - **Modify the environment**: change `env.make_train_1`with the env you want in `env= DummyVecEnv([env.make_train_1]))` in play.py
# - **See the agent playing**: run `python play.py`
# 
# ### Continue to train 🏃‍♂️
# ⚠️ There is a big risk **of overfitting**
# - Run `python agent.py`

# # This notebook is a companion of the PPO implementation in [Deep Reinforcement Learning Course's repo]() where each part of the code is explained in the comments

# ## Step 1: Download Sonic the Hedgehog 2 and 3💻

# 1. The first step is to download the games, to make it works on retro **you need to buy them legally on Steam**
# <br><br>
# [Sonic the Hedgehog 2](https://store.steampowered.com/app/71163/Sonic_The_Hedgehog_2/)<br>
# [Sonic 3 & Knuckles](https://store.steampowered.com/app/71162/Sonic_3__Knuckles/)<br>
# 
# <br>
# <img src="https://steamcdn-a.akamaihd.net/steam/apps/71162/header.jpg?t=1522076247" alt="Sonic the Hedgehog 3"/>
# <br>
# 
# 2. Then follow the **Quickstart part** of this [website](https://contest.openai.com/2018-1/details/)
# 

# ## Step 2: Build all elements we need for our environement in sonic_env.py 🖼️

# - `PreprocessFrame(gym.ObservationWrapper)` : in this class we will **preprocess our environment**
#     - Set frame to gray 
#     - Resize the frame to 96x96x1
# <br>
# <br>
# - `ActionsDiscretizer(gym.ActionWrapper)` : in this class we **limit the possibles actions in our environment** (make it discrete)
# 
# In fact you'll see that for each action in actions:
#     Create an array of 12 False (12 = nb of buttons)
#         For each button in action: (for instance ['LEFT']) we need to make that left button index = True
#             Then the button index = LEFT = True
# 
# --> In fact at the end we will have an array where each array is an action and **each elements True of this array are the buttons clicked.**
# For instance LEFT action = [F, F, F, F, F, F, T, F, F, F, F, F]
# <br><br>
# - `RewardScaler(gym.RewardWrapper)`: We **scale the rewards** to reasonable scale (useful in PPO).
# <br><br>
# - `AllowBacktracking(gym.Wrapper)`: **Allow the agent to go backward without being discourage** (avoid our agent to be stuck on a wall during the game).
# <br><br>
# 
# - `make_env(env_idx)` : **Build an environement** (and stack 4 frames together using `FrameStack`
# <br><br>
# The idea is that we'll build multiple instances of the environment, different environments each times (different level) **to avoid overfitting and helping our agent to generalize better at playing sonic** 
# <br><br>
# To handle these multiple environements we'll use `SubprocVecEnv` that **creates a vector of n environments to run them simultaneously.**

# ## Step 3: Build the PPO architecture in architecture.py 🧠

# - `from baselines.common.distributions import make_pdtype`: This function selects the **probability distribution over actions**
# <br><br>
# 
# - First, we create two functions that will help us to avoid to call conv and fc each time.
#     - `conv`: function to create a convolutional layer.
#     - `fc`: function to create a fully connected layer.
# <br><br>
# - Then, we create `PPOPolicy`, the object that **contains the architecture**<br>
# 3 CNN for spatial dependencies<br>
# Temporal dependencies is handle by stacking frames<br>
# (Something funny nobody use LSTM in OpenAI Retro contest)<br>
# 1 common FC<br>
# 1 FC for policy<br>
# 1 FC for value
# <br><br>
# - `self.pdtype = make_pdtype(action_space)`: Based on the action space, will **select what probability distribution typewe will use to distribute action in our stochastic policy** (in our case DiagGaussianPdType aka Diagonal Gaussian, multivariate normal distribution
# <br><br>
# - `self.pdtype.pdfromlatent` : return a returns a probability distribution over actions (self.pd) and our pi logits (self.pi).
# <br><br>
# 
# - We create also 3 useful functions in PPOPolicy
#     - `def step(state_in, *_args, **_kwargs)`: Function use to take a step returns **action to take, V(s) and neglogprob**
#     - `def value(state_in, *_args, **_kwargs)`: Function that calculates **only the V(s)**
#     -  `def select_action(state_in, *_args, **_kwargs)`: Function that output **only the action to take**

# ## Step 4: Build the Model in model.py 🏗️

# We use Model object to:
# __init__:
# - `policy(sess, ob_space, action_space, nenvs, 1, reuse=False)`: Creates the step_model (used for sampling)
# - `policy(sess, ob_space, action_space, nenvs*nsteps, nsteps, reuse=True)`: Creates the train_model (used for training)
# 
# - `def train(states_in, actions, returns, values, lr)`: Make the training part (calculate advantage and feedforward and retropropagation of gradients)
# 
# First we create the placeholders:
# - `oldneglopac_`: keep track of the old policy
# - `oldvpred_`: keep track of the old value
# - `cliprange_`: keep track of the cliprange
# 
# save/load():
# - `def save(save_path)`:Save the weights.
# - `def load(load_path)`:Load the weights

# ## Step 5: Build the Runner in model.py 🏃
# Runner will be used to make a mini batch of experiences
# - Each environement send 1 timestep (4 frames stacked) (`self.obs`)
# - This goes through `step_model`
#     - Returns actions, values.
# - Append `mb_obs`, `mb_actions`, `mb_values`, `mb_dones`, `mb_neglopacs`.
# - Take actions in environments and watch the consequences
#     - return `obs`, `rewards`, `dones`, `infos` (which contains a ton of useful informations such as the number of rings, the player position etc...)
#     
# - We need to calculate advantage to do that we use General Advantage Estimation

# ## Step 6: Build the learn function in model.py
# The learn function can be seen as **the gathering of all the logic of our A2C**
# - Instantiate the model object (that creates step_model and train_model)
# - Instantiate the runner object
# - Train always in two phases:
#     - Run to get a batch of experiences
#     - Train that batch of experiences
# <br><br>
# 
# We use explained_variance which is a **really important parameter**:
# <br><br>
# `ev = 1 - Variance[y - ypredicted] / Variance [y]`
# <br><br>
# In fact this calculates **if value function is a good predictor of the returns or if it's just worse than predicting nothing.**
# ev=0  =>  might as well have predicted zero
# ev<0  =>  worse than just predicting zero so you're overfitting (need to tune some hyperparameters)
# ev=1  =>  perfect prediction
# 
# --> The goal is that **ev goes closer and closer to 1.**

# ## Step 7: Build the play function in model.py
# - This function will be use to play an environment using the trained model.

# ## Step 8: Build the agent.py

# - `config.gpu_options.allow_growth = True` : This creates a GPU session
# -  `model.learn(...)` : Here we just call the learn function that contains all the elements needed to train our PPO agent
# 
