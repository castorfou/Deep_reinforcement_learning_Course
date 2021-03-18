#!/usr/bin/env python
# coding: utf-8

# # Doom-Health: REINFORCE Monte Carlo Policy gradients 🕹️

# In this notebook we'll implement an agent <b>that try to survive in Doom environment by using a Policy Gradient architecture.</b> <br>
# Our agent playing Doom:
# 
# <img src="assets/projectw4.gif" style="max-width: 600px;" alt="Policy Gradient with Doom"/>

# # You can follow this notebook with this video tutorial 📹 that will helps you to understand each step:

# In[1]:


from IPython.display import HTML
HTML('<iframe width="560" height="315" src="https://www.youtube.com/embed/wLTQRuizVyE?showinfo=0" frameborder="0" allow="autoplay; encrypted-media" allowfullscreen></iframe>')


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

# ## Prerequisites 🏗️
# Before diving on the notebook **you need to understand**:
# - The foundations of Reinforcement learning (MC, TD, Rewards hypothesis...) [Article](https://medium.freecodecamp.org/an-introduction-to-reinforcement-learning-4339519de419)
# - Policy gradients [Article](https://medium.freecodecamp.org/an-introduction-to-policy-gradients-with-cartpole-and-doom-495b5ef2207f)
# - We made a [tutorial video](https://youtu.be/wLTQRuizVyE) where we implement a Policy Gradient agent with Tensorflow that learns to play Doom 👹🔫 in a Deathmatch environment.

# ## Step 1: Import the libraries 📚

# In[2]:


import tensorflow as tf      # Deep Learning library
import numpy as np           # Handle matrices
from vizdoom import *        # Doom Environment
import random                # Handling random number generation
import time                  # Handling time calculation
from skimage import transform# Help us to preprocess the frames

from collections import deque# Ordered collection with ends
import matplotlib.pyplot as plt # Display graphs

import warnings # This ignore all the warning messages that are normally printed during the training because of skiimage
warnings.filterwarnings('ignore')


# In[3]:


config = tf.ConfigProto()
config.gpu_options.allow_growth=True


# ## Step 2: Create our environment 🎮
# - Now that we imported the libraries/dependencies, we will create our environment.
# - Doom environment takes:
#     - A `configuration file` that **handle all the options** (size of the frame, possible actions...)
#     - A `scenario file`: that **generates the correct scenario** (in our case basic **but you're invited to try other scenarios**).
# - Note: We have 3 possible actions `[[0,0,1], [1,0,0], [0,1,0]]` so we don't need to do one hot encoding (thanks to < a href="https://stackoverflow.com/users/2237916/silgon">silgon</a> for figuring out. 
# 
# ### Our environment
# <img src="assets/health_doom.jpg" style="max-width:500px;" alt="Doom health"/>
# 
# The purpose of this scenario is to teach the agent **how to survive without knowing what makes him survive.** Agent know only that life is precious and death is bad so he must learn what prolongs his existence and that his health is connected with it.
# 
# Map is a rectangle with green, acidic floor which hurts the player periodically. Initially there are some medkits spread uniformly over the map. A new medkit falls from the skies every now and then. **Medkits heal some portions of player's health - to survive agent needs to pick them up. Episode finishes after player's death or on timeout.**
# 
# Further configuration:
# 
# - living_reward = 1
# - 3 available buttons: turn left, turn right, move forward
# - 1 available game variable: HEALTH
# - death penalty = 100
# <br><br>

# In[4]:


"""
Here we create our environment
"""
def create_environment():
    game = DoomGame()
    
    # Load the correct configuration
    game.load_config("health_gathering.cfg")
    
    # Load the correct scenario (in our case defend_the_center scenario)
    game.set_doom_scenario_path("health_gathering.wad")
    
    game.init()
    
    # Here our possible actions
    # [[1,0,0],[0,1,0],[0,0,1]]
    possible_actions  = np.identity(3,dtype=int).tolist()
    
    return game, possible_actions


# In[5]:


game, possible_actions = create_environment()


# ## Step 3: Define the preprocessing functions ⚙️
# ### preprocess_frame 🖼️
# Preprocessing is an important step, <b>because we want to reduce the complexity of our states to reduce the computation time needed for training.</b>
# <br><br>
# Our steps:
# - Grayscale each of our frames (because <b> color does not add important information </b>). But this is already done by the config file.
# - Crop the screen (in our case we remove the roof because it contains no information)
# - We normalize pixel values
# - Finally we resize the preprocessed frame

# In[6]:


"""
    preprocess_frame:
    Take a frame.
    Resize it.
        __________________
        |                 |
        |                 |
        |                 |
        |                 |
        |_________________|
        
        to
        _____________
        |            |
        |            |
        |            |
        |____________|
    Normalize it.
    
    return preprocessed_frame
    
    """
def preprocess_frame(frame):
    # Greyscale frame already done in our vizdoom config
    # x = np.mean(frame,-1)
    
    # Crop the screen (remove the roof because it contains no information)
    # [Up: Down, Left: right]
    cropped_frame = frame[80:,:]
    
    # Normalize Pixel Values
    normalized_frame = cropped_frame/255.0
    
    # Resize
    preprocessed_frame = transform.resize(normalized_frame, [84,84])
    
    return preprocessed_frame


# ### stack_frames
# 👏 This part was made possible thanks to help of <a href="https://github.com/Miffyli">Anssi</a><br>
# 
# As explained in this really <a href="https://danieltakeshi.github.io/2016/11/25/frame-skipping-and-preprocessing-for-deep-q-networks-on-atari-2600-games/">  good article </a> we stack frames.
# 
# Stacking frames is really important because it helps us to **give have a sense of motion to our Neural Network.**
# 
# - First we preprocess frame
# - Then we append the frame to the deque that automatically **removes the oldest frame**
# - Finally we **build the stacked state**
# 
# This is how work stack:
# - For the first frame, we feed 4 frames
# - At each timestep, **we add the new frame to deque and then we stack them to form a new stacked frame**
# - And so on
# <img src="https://raw.githubusercontent.com/simoninithomas/Deep_reinforcement_learning_Course/master/DQN/Space%20Invaders/assets/stack_frames.png" alt="stack">
# - If we're done, **we create a new stack with 4 new frames (because we are in a new episode)**.

# In[7]:


stack_size = 4 # We stack 4 frames

# Initialize deque with zero-images one array for each image
stacked_frames  =  deque([np.zeros((84,84), dtype=np.int) for i in range(stack_size)], maxlen=4) 

def stack_frames(stacked_frames, state, is_new_episode):
    # Preprocess frame
    frame = preprocess_frame(state)
    
    if is_new_episode:
        # Clear our stacked_frames
        stacked_frames = deque([np.zeros((84,84), dtype=np.int) for i in range(stack_size)], maxlen=4)
        
        # Because we're in a new episode, copy the same frame 4x
        stacked_frames.append(frame)
        stacked_frames.append(frame)
        stacked_frames.append(frame)
        stacked_frames.append(frame)
        
        # Stack the frames
        stacked_state = np.stack(stacked_frames, axis=2)

    else:
        # Append frame to deque, automatically removes the oldest frame
        stacked_frames.append(frame)

        # Build the stacked state (first dimension specifies different frames)
        stacked_state = np.stack(stacked_frames, axis=2) 
    
    return stacked_state, stacked_frames


# ### discount_and_normalize_rewards 💰
# This function is important, because we are in a Monte Carlo situation. <br>
# 
# We need to **discount the rewards at the end of the episode**. This function takes, the reward discount it, and **then normalize them** (to avoid a big variability in rewards).

# In[8]:


def discount_and_normalize_rewards(episode_rewards):
    discounted_episode_rewards = np.zeros_like(episode_rewards)
    cumulative = 0.0
    for i in reversed(range(len(episode_rewards))):
        cumulative = cumulative * gamma + episode_rewards[i]
        discounted_episode_rewards[i] = cumulative
    
    mean = np.mean(discounted_episode_rewards)
    std = np.std(discounted_episode_rewards)
    discounted_episode_rewards = (discounted_episode_rewards - mean) / (std)

    return discounted_episode_rewards


# ## Step 4: Set up our hyperparameters ⚗️
# In this part we'll set up our different hyperparameters. But when you implement a Neural Network by yourself you will **not implement hyperparamaters at once but progressively**.
# 
# - First, you begin by defining the neural networks hyperparameters when you implement the model.
# - Then, you'll add the training hyperparameters when you implement the training algorithm.

# In[9]:


### ENVIRONMENT HYPERPARAMETERS
state_size = [84,84,4] # Our input is a stack of 4 frames hence 84x84x4 (Width, height, channels) 
action_size = game.get_available_buttons_size() # 3 possible actions: turn left, turn right, move forward
stack_size = 4 # Defines how many frames are stacked together

## TRAINING HYPERPARAMETERS
learning_rate = 0.002
num_epochs = 500 # Total epochs for training 

batch_size = 1000 # Each 1 is a timestep (NOT AN EPISODE) # YOU CAN CHANGE TO 5000 if you have GPU
gamma = 0.95 # Discounting rate

### MODIFY THIS TO FALSE IF YOU JUST WANT TO SEE THE TRAINED AGENT
training = True


# Quick note: Policy gradient methods like reinforce **are on-policy method which can not be updated from experience replay.**

# ## Step 5: Create our Policy Gradient Neural Network model 🧠

# <img src="assets/doomPG.png" alt="Doom PG"/>

# In[10]:


class PGNetwork:
    def __init__(self, state_size, action_size, learning_rate, name='PGNetwork'):
        self.state_size = state_size
        self.action_size = action_size
        self.learning_rate = learning_rate
        
        with tf.variable_scope(name):
            with tf.name_scope("inputs"):
                # We create the placeholders
                # *state_size means that we take each elements of state_size in tuple hence is like if we wrote
                # [None, 84, 84, 4]
                self.inputs_= tf.placeholder(tf.float32, [None, *state_size], name="inputs_")
                self.actions = tf.placeholder(tf.int32, [None, action_size], name="actions")
                self.discounted_episode_rewards_ = tf.placeholder(tf.float32, [None, ], name="discounted_episode_rewards_")
            
                
                # Add this placeholder for having this variable in tensorboard
                self.mean_reward_ = tf.placeholder(tf.float32, name="mean_reward")
                
            with tf.name_scope("conv1"):
                """
                First convnet:
                CNN
                BatchNormalization
                ELU
                """
                # Input is 84x84x4
                self.conv1 = tf.layers.conv2d(inputs = self.inputs_,
                                             filters = 32,
                                             kernel_size = [8,8],
                                             strides = [4,4],
                                             padding = "VALID",
                                              kernel_initializer=tf.contrib.layers.xavier_initializer_conv2d(),
                                             name = "conv1")

                self.conv1_batchnorm = tf.layers.batch_normalization(self.conv1,
                                                       training = True,
                                                       epsilon = 1e-5,
                                                         name = 'batch_norm1')

                self.conv1_out = tf.nn.elu(self.conv1_batchnorm, name="conv1_out")
                ## --> [20, 20, 32]
            
            with tf.name_scope("conv2"):
                """
                Second convnet:
                CNN
                BatchNormalization
                ELU
                """
                self.conv2 = tf.layers.conv2d(inputs = self.conv1_out,
                                     filters = 64,
                                     kernel_size = [4,4],
                                     strides = [2,2],
                                     padding = "VALID",
                                    kernel_initializer=tf.contrib.layers.xavier_initializer_conv2d(),
                                     name = "conv2")

                self.conv2_batchnorm = tf.layers.batch_normalization(self.conv2,
                                                       training = True,
                                                       epsilon = 1e-5,
                                                         name = 'batch_norm2')

                self.conv2_out = tf.nn.elu(self.conv2_batchnorm, name="conv2_out")
                ## --> [9, 9, 64]
            
            with tf.name_scope("conv3"):
                """
                Third convnet:
                CNN
                BatchNormalization
                ELU
                """
                self.conv3 = tf.layers.conv2d(inputs = self.conv2_out,
                                     filters = 128,
                                     kernel_size = [4,4],
                                     strides = [2,2],
                                     padding = "VALID",
                                    kernel_initializer=tf.contrib.layers.xavier_initializer_conv2d(),
                                     name = "conv3")

                self.conv3_batchnorm = tf.layers.batch_normalization(self.conv3,
                                                       training = True,
                                                       epsilon = 1e-5,
                                                         name = 'batch_norm3')

                self.conv3_out = tf.nn.elu(self.conv3_batchnorm, name="conv3_out")
                ## --> [3, 3, 128]
            
            with tf.name_scope("flatten"):
                self.flatten = tf.layers.flatten(self.conv3_out)
                ## --> [1152]
            
            with tf.name_scope("fc1"):
                self.fc = tf.layers.dense(inputs = self.flatten,
                                      units = 512,
                                      activation = tf.nn.elu,
                                           kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                    name="fc1")
            
            with tf.name_scope("logits"):
                self.logits = tf.layers.dense(inputs = self.fc, 
                                               kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                              units = 3, 
                                            activation=None)
            
            with tf.name_scope("softmax"):
                self.action_distribution = tf.nn.softmax(self.logits)
                

            with tf.name_scope("loss"):
                # tf.nn.softmax_cross_entropy_with_logits computes the cross entropy of the result after applying the softmax function
                # If you have single-class labels, where an object can only belong to one class, you might now consider using 
                # tf.nn.sparse_softmax_cross_entropy_with_logits so that you don't have to convert your labels to a dense one-hot array. 
                self.neg_log_prob = tf.nn.softmax_cross_entropy_with_logits_v2(logits = self.logits, labels = self.actions)
                self.loss = tf.reduce_mean(self.neg_log_prob * self.discounted_episode_rewards_) 
        
    
            with tf.name_scope("train"):
                self.train_opt = tf.train.RMSPropOptimizer(self.learning_rate).minimize(self.loss)


# In[11]:


# Reset the graph
tf.reset_default_graph()

# Instantiate the PGNetwork
PGNetwork = PGNetwork(state_size, action_size, learning_rate)

# Initialize Session
# sess = tf.Session()
sess = tf.Session(config=config)

init = tf.global_variables_initializer()
sess.run(init)


# ## Step 6: Set up Tensorboard 📊
# For more information about tensorboard, please watch this <a href="https://www.youtube.com/embed/eBbEDRsCmv4">excellent 30min tutorial</a> <br><br>
# To launch tensorboard : `tensorboard --logdir=./tensorboard/pg/test`

# In[12]:


# Setup TensorBoard Writer
writer = tf.summary.FileWriter("./tensorboard/pg/test")

## Losses
tf.summary.scalar("Loss", PGNetwork.loss)

## Reward mean
tf.summary.scalar("Reward_mean", PGNetwork.mean_reward_ )

write_op = tf.summary.merge_all()


# ## Step 7: Train our Agent 🏃‍♂️

# Here we'll create batches.<br>
# These batches contains episodes **(their number depends on how many rewards we collect**: for instance if we have episodes with only 10 rewards we can put batch_size/10 episodes
# <br>
# * Make a batch
#     * For each step:
#         * Choose action a
#         * Perform action a
#         * Store s, a, r
#         * **If** done:
#             * Calculate sum reward
#             * Calculate gamma Gt

# In[13]:


def make_batch(batch_size, stacked_frames):
    # Initialize lists: states, actions, rewards_of_episode, rewards_of_batch, discounted_rewards
    states, actions, rewards_of_episode, rewards_of_batch, discounted_rewards = [], [], [], [], []
    
    # Reward of batch is also a trick to keep track of how many timestep we made.
    # We use to to verify at the end of each episode if > batch_size or not.
    
    # Keep track of how many episodes in our batch (useful when we'll need to calculate the average reward per episode)
    episode_num  = 1
    
    # Launch a new episode
    game.new_episode()
        
    # Get a new state
    state = game.get_state().screen_buffer
    state, stacked_frames = stack_frames(stacked_frames, state, True)

    while True:
        # Run State Through Policy & Calculate Action
        action_probability_distribution = sess.run(PGNetwork.action_distribution, 
                                                   feed_dict={PGNetwork.inputs_: state.reshape(1, *state_size)})
        
        # REMEMBER THAT WE ARE IN A STOCHASTIC POLICY SO WE DON'T ALWAYS TAKE THE ACTION WITH THE HIGHEST PROBABILITY
        # (For instance if the action with the best probability for state S is a1 with 70% chances, there is
        #30% chance that we take action a2)
        action = np.random.choice(range(action_probability_distribution.shape[1]), 
                                  p=action_probability_distribution.ravel())  # select action w.r.t the actions prob
        action = possible_actions[action]

        # Perform action
        reward = game.make_action(action)
        done = game.is_episode_finished()

        # Store results
        states.append(state)
        actions.append(action)
        rewards_of_episode.append(reward)
        
        if done:
            # The episode ends so no next state
            next_state = np.zeros((84, 84), dtype=np.int)
            next_state, stacked_frames = stack_frames(stacked_frames, next_state, False)
            
            # Append the rewards_of_batch to reward_of_episode
            rewards_of_batch.append(rewards_of_episode)
            
            # Calculate gamma Gt
            discounted_rewards.append(discount_and_normalize_rewards(rewards_of_episode))
           
            # If the number of rewards_of_batch > batch_size stop the minibatch creation
            # (Because we have sufficient number of episode mb)
            # Remember that we put this condition here, because we want entire episode (Monte Carlo)
            # so we can't check that condition for each step but only if an episode is finished
            if len(np.concatenate(rewards_of_batch)) > batch_size:
                break
                
            # Reset the transition stores
            rewards_of_episode = []
            
            # Add episode
            episode_num += 1
            
            # Start a new episode
            game.new_episode()

            # First we need a state
            state = game.get_state().screen_buffer

            # Stack the frames
            state, stacked_frames = stack_frames(stacked_frames, state, True)
         
        else:
            # If not done, the next_state become the current state
            next_state = game.get_state().screen_buffer
            next_state, stacked_frames = stack_frames(stacked_frames, next_state, False)
            state = next_state
                         
    return np.stack(np.array(states)), np.stack(np.array(actions)), np.concatenate(rewards_of_batch), np.concatenate(discounted_rewards), episode_num


# * Create the Neural Network
# * Initialize the weights
# * Init the environment
# * maxReward = 0 # Keep track of maximum reward
# * **For** epochs in range(num_epochs):
#     * Get batches
#     * Optimize

# In[14]:


# Keep track of all rewards total for each batch
allRewards = []

total_rewards = 0
maximumRewardRecorded = 0
mean_reward_total = []
epoch = 1
average_reward = []

# Saver
saver = tf.train.Saver()

if training:
    # Load the model
    #saver.restore(sess, "./models/model.ckpt")

    while epoch < num_epochs + 1:
        # Gather training data
        states_mb, actions_mb, rewards_of_batch, discounted_rewards_mb, nb_episodes_mb = make_batch(batch_size, stacked_frames)

        ### These part is used for analytics
        # Calculate the total reward ot the batch
        total_reward_of_that_batch = np.sum(rewards_of_batch)
        allRewards.append(total_reward_of_that_batch)

        # Calculate the mean reward of the batch
        # Total rewards of batch / nb episodes in that batch
        mean_reward_of_that_batch = np.divide(total_reward_of_that_batch, nb_episodes_mb)
        mean_reward_total.append(mean_reward_of_that_batch)

        # Calculate the average reward of all training
        # mean_reward_of_that_batch / epoch
        average_reward_of_all_training = np.divide(np.sum(mean_reward_total), epoch)

        # Calculate maximum reward recorded 
        maximumRewardRecorded = np.amax(allRewards)

        print("==========================================")
        print("Epoch: ", epoch, "/", num_epochs)
        print("-----------")
        print("Number of training episodes: {}".format(nb_episodes_mb))
        print("Total reward: {}".format(total_reward_of_that_batch, nb_episodes_mb))
        print("Mean Reward of that batch {}".format(mean_reward_of_that_batch))
        print("Average Reward of all training: {}".format(average_reward_of_all_training))
        print("Max reward for a batch so far: {}".format(maximumRewardRecorded))

        # Feedforward, gradient and backpropagation
        loss_, _ = sess.run([PGNetwork.loss, PGNetwork.train_opt], feed_dict={PGNetwork.inputs_: states_mb.reshape((len(states_mb), 84,84,4)),
                                                            PGNetwork.actions: actions_mb,
                                                                     PGNetwork.discounted_episode_rewards_: discounted_rewards_mb 
                                                                    })

        print("Training Loss: {}".format(loss_))

        # Write TF Summaries
        summary = sess.run(write_op, feed_dict={PGNetwork.inputs_: states_mb.reshape((len(states_mb), 84,84,4)),
                                                            PGNetwork.actions: actions_mb,
                                                                     PGNetwork.discounted_episode_rewards_: discounted_rewards_mb,
                                                                    PGNetwork.mean_reward_: mean_reward_of_that_batch
                                                                    })

        #summary = sess.run(write_op, feed_dict={x: s_.reshape(len(s_),84,84,1), y:a_, d_r: d_r_, r: r_, n: n_})
        writer.add_summary(summary, epoch)
        writer.flush()

        # Save Model
        if epoch % 10 == 0:
            saver.save(sess, "./models/model.ckpt")
            print("Model saved")
        epoch += 1


# ## Step 8: Watch our Agent play 👀
# Now that we trained our agent, we can test it

# In[15]:


# Saver
saver = tf.train.Saver()

with tf.Session() as sess:
    game = DoomGame()

    # Load the correct configuration 
    game.load_config("health_gathering.cfg")
    
    # Load the correct scenario (in our case basic scenario)
    game.set_doom_scenario_path("health_gathering.wad")
    
    # Load the model
    saver.restore(sess, "./models/model.ckpt")
    game.init()
    
    for i in range(10):
        
        # Launch a new episode
        game.new_episode()

        # Get a new state
        state = game.get_state().screen_buffer
        state, stacked_frames = stack_frames(stacked_frames, state, True)

        while not game.is_episode_finished():
        
            # Run State Through Policy & Calculate Action
            action_probability_distribution = sess.run(PGNetwork.action_distribution, 
                                                       feed_dict={PGNetwork.inputs_: state.reshape(1, *state_size)})

            # REMEMBER THAT WE ARE IN A STOCHASTIC POLICY SO WE DON'T ALWAYS TAKE THE ACTION WITH THE HIGHEST PROBABILITY
            # (For instance if the action with the best probability for state S is a1 with 70% chances, there is
            #30% chance that we take action a2)
            action = np.random.choice(range(action_probability_distribution.shape[1]), 
                                      p=action_probability_distribution.ravel())  # select action w.r.t the actions prob
            action = possible_actions[action]

            # Perform action
            reward = game.make_action(action)
            done = game.is_episode_finished()
            
            if done:
                break
            else:
                # If not done, the next_state become the current state
                next_state = game.get_state().screen_buffer
                next_state, stacked_frames = stack_frames(stacked_frames, next_state, False)
                state = next_state
        

        print("Score for episode ", i, " :", game.get_total_reward())
    game.close()

