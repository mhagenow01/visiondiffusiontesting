import sys
import os

project_path = os.path.dirname(os.path.realpath(__file__))

from envs.gymuncerpentine import Uncerpentine
from envs.robosuite_tending import MachineTending
from envs.ur5e_tending import UR5ETending

from policies.kdtree_stochastic import KDTreeStochastic
from policies.time_dependent import TimeDependent
from policies.diffusion import Diffusion
from policies.sticky_filter import StickyFilter
from policies.traded_teleop import TradedTeleop

from assistance.no_assist import NoAssist
from assistance.teleop import Teleoperation
from assistance.discrete import Discrete
from assistance.corrections import Corrections

from human_assist.assistance_blend_sm import AssistanceBlendSM
from human_assist.uncertainty_teleop import UncertainTeleopSM

import sys
import pygame
import time

import rospy
from std_msgs.msg import String

from pynput import keyboard

import plotly
import plotly.graph_objs as go

import numpy as np
import ipdb
import argparse
import matplotlib.pyplot as plt
import matplotlib

import pickle



def plotSingleforecast(forecast):
    data = []
    
    mins = [-1,-1,0]
    maxes = [1, 1, 2]
    
    for nn in range(np.shape(forecast)[0]):

        tmp_color = 'blue'
        
        # Configure the trace.
        data.append(go.Scatter3d(
            x=forecast[nn,:,0].flatten(),  # <-- Put your data instead
            y=forecast[nn,:,1].flatten(),  # <-- Put your data instead
            z=forecast[nn,:,2].flatten(),  # <-- Put your data instead
            mode='markers',
            marker={
                'size': 0.8,
                'opacity': 0.8,
                'color': tmp_color
            }
        ))


        # Configure the layout.
        layout = go.Layout(
            margin={'l': 0, 'r': 0, 'b': 0, 't': 0},
        )

    # Bounds configuring with invisible dots

    trace3 = go.Scatter3d(
        x=[mins[0], maxes[0]],  # <-- Put your data instead
        y=[mins[1], maxes[1]],  # <-- Put your data instead
        z=[mins[2], maxes[2]],  # <-- Put your data instead
        mode='markers',
        marker={
            'size': 0.0,
            'opacity': 0.0,
            'color': 'blue'
        }
    )
    
    data.append(trace3)

    plot_figure = go.Figure(data=data, layout=layout)

    plot_figure.update_layout(yaxis_range=[-10,10])


    # Render the plot.
    plotly.offline.iplot(plot_figure)

    fig, ax = plt.subplots(7,1)
    for state in range(np.shape(forecast)[2]):
        for nn in range(np.shape(forecast)[0]):
            ax[state].plot(forecast[nn,:,state])
    plt.savefig('rollouts.png')



class RunAssistance():
    def __init__(self, savelocation, policy, env, assisters, likelihoodprocessor, rate, action_chunking, partid, method, args) -> None:
        self.policy = policy
        self.env = env
        self.assisters = assisters
        self.args = args
        self.human_assist = AssistanceBlendSM(self.assisters)
        # self.human_assist = UncertainTeleopSM()

        self.action_chunking = action_chunking
        self.curr_action_chunk = 0
        self.cumulative_reward = 0
        self.iter = 0
        self.rate = rate
        self.last_t = None

        self.episode_likelihoods = []

        self.assist_names = []
        for assister in self.assisters:
            self.assist_names.append(assister.__class__.__name__)

        self.likelihoodprocessor = likelihoodprocessor

        self.partid = partid
        if self.partid is not None:
            self.method = method
            try:
                rospy.init_node('assistance_runner')
            except Exception as e:
                pass
            
            self.loggingpub = rospy.Publisher("/trialbagger", String, queue_size =1, latch = True)
            time.sleep(1)


    def likelihoodplotter(self,episode_likelihoods,assist_names):
        fig, ax = plt.subplots(2,1)
        num_pts = np.shape(episode_likelihoods)[0]
        for ii in range(np.shape(episode_likelihoods)[1]):
            ax[0].plot(np.arange(num_pts),np.array(episode_likelihoods)[:,ii])
        ax[0].legend(assist_names)
        ax[1].set_xlabel("Sample")
        ax[0].set_ylabel("Likelihood")
        
        # temporal plot of most likely assistance
        best = np.argmax(episode_likelihoods,axis=1)
        colors=['#1f77b4','#ff7f0e','#2ca02c','#d62728','#9467bd','#8c564b','#e377c2','#7f7f7f','#bcbd22','#17becf']
        for tt in range(num_pts):
            ax[1].bar(tt,1,1,color=colors[best[tt]])    
        plt.savefig("likelihoods.png")

    def processKeyPress(self,key):
        try:
            # if called by pygame it is a string, otherwise it is a Key() instance
            if isinstance(key, str):
                keychar = key
            else:
                keychar = key.char

            if keychar == 's':  # start over
                if not self.episode_running:
                    self.episode_over = False
                    self.obs = self.env.reset()
                    self.episode_likelihoods = []
                    self.pausemode = False
                    self.episode_running = True
                    self.cumulative_reward = 0
                    self.iter = 0
                    self.policy.reset()
                    self.likelihoodprocessor.reset()
                    self.change_counter = 0
                    self.prevbest = None
                    self.curr_action_chunk = 0
                    if self.partid is not None:
                        self.loggingpub.publish(String("start_"+str(self.partid)+"_"+str(self.method)))
            if keychar == 'p':  # pause mode
                if not self.pausemode:
                        if not self.episode_running:
                            self.obs = self.env.reset()
                        self.pausemode = True
                        self.episode_running = False
                if self.pausemode:
                    self.advanceFrame = True
            if keychar == 'r':  # resume from pause mode
                if self.pausemode:
                    self.pausemode = False
                    self.episode_running = True
            if keychar == 'e':  # end episode
                self.episode_over = True
                if self.episode_running:
                    if self.partid is not None:
                        self.loggingpub.publish(String("stop"))
            if keychar == 'v': # visualize current forecast
                 # put in pause mode
                if not self.pausemode:
                    if not self.episode_running:
                        self.obs = self.env.reset()
                    self.pausemode = True
                    self.episode_running = False

                last_forecast = self.forecasts[-1]
                plotSingleforecast(last_forecast)
            if keychar == 'w':
                self.prevbest = 0 # force takeover
            if keychar == 'q': # quit
                sys.exit()
        except Exception as err:
            pass
            # print(err)

    def run(self):
        self.running = True
        self.episode_running = False
        self.pausemode = False
        self.advanceFrame = False # for pausemode
        self.curr_action_chunk = 0
        self.env.reset()

        if not self.env.pygameKeyboard():
            listener = keyboard.Listener(on_press=self.processKeyPress)
            listener.start()

        print("------------------------")
        print(" Keyboard controls: ")
        print(" ... s to start")
        print(" ... p to pause/advance")
        print(" ... r to resume (paused)")
        print(" ... q to quit")
        print("------------------------")

        meta = dict()

        self.forecasts = []

        while self.running:    
            # issues with using pygame with mujoco and pyinput with pygame so thus
            # two sets of controls
            if self.env.pygameKeyboard():
                pygame.event.pump()
                keys = pygame.key.get_pressed()
                if keys[pygame.K_s]:
                    self.processKeyPress('s')
                if keys[pygame.K_p]:
                    self.processKeyPress('p')
                if keys[pygame.K_r]:
                    self.processKeyPress('r')
                if keys[pygame.K_e]:
                    self.processKeyPress('e')
                if keys[pygame.K_q]:
                    self.processKeyPress('q')
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        self.processKeyPress('q')

            if self.episode_running or self.advanceFrame:
                
              
                ######### Assistance Value Estimation ###############

                # if self.iter == 220:
                #     print('hi')
                forecast = np.array(self.policy.forecastAction(self.obs,50,12,self.env))
                self.forecasts.append(forecast)

                likelihoods = []
                for assister in self.assisters:
                    likelihoods.append(assister.getPenalizedLikelihood(forecast))
                    # likelihoods.append(assister.getSumEntropy(forecast))

                

                if self.args.plotlikelihood:
                    self.episode_likelihoods.append(likelihoods)

               
                likelihoods = self.likelihoodprocessor.getLikelihood(likelihoods=likelihoods)
                print("iter: ",self.iter," LIK:",np.round(likelihoods,5)," ",self.change_counter)
                likelihoods_raw = likelihoods.copy()

                # a few in a row
                if self.prevbest is None:
                    self.prevbest = np.argmax(likelihoods)
                if np.argmax(likelihoods)!=self.prevbest:
                    if self.change_counter>=0:
                        self.prevbest = np.argmax(likelihoods)
                        self.change_counter = 0
                    else:
                        self.change_counter+=1
                likelihoods = [0]*len(self.assisters)
                likelihoods = np.array(likelihoods)
                likelihoods[self.prevbest] = 1

                meta['assist_names'] = self.assist_names
                meta['assist_likelihoods'] = likelihoods
                meta['likelihoods_raw'] = likelihoods_raw

                if self.args.showforecast:
                    meta['forecasted_actions'] = forecast
              
                ######## Get Action #################
                if forecast is None:
                    action = self.policy.getAction(self.obs) # call either way to make sure internal state tracking updates
                elif self.args.optimal:
                    action = self.env.getOptimalAction() # use environment optimal instead 
                else:
                    # pull from forecasts with action chunking
                    if self.curr_action_chunk == 0:
                        self.temp_forecast = forecast.copy()
                        action = self.temp_forecast[0,0,:] # pull from forecast (avoid second call to gpu)
                    else:
                        action = self.temp_forecast[0,self.curr_action_chunk,:]
                    # print("AC:",self.curr_action_chunk)

                    self.curr_action_chunk +=1
                    
                    if self.curr_action_chunk >= self.action_chunking:
                        self.curr_action_chunk = 0

                if not self.args.optimal:
                    action, force, teleoperation_active, discrete_active, likelihoods, sm_state = self.human_assist.getModifiedAction(action,self.obs,likelihoods,forecast)
                    meta['teleoperation_active'] = teleoperation_active
                    meta['discrete_active'] = discrete_active
                    meta['sm_state'] = sm_state
                    if teleoperation_active or discrete_active: # reset while active in teleop or discrete so it isn't a sterile prediction
                        self.curr_action_chunk = 0
                # print(np.round(action,3))

                ######### Environment Step and Store Data ###########
                data_tmp = self.env.step(action, meta=meta) # obs, reward, terminated, truncated, info
                # note: some gym environments don't seem to return 5 (just 4)
                self.obs = data_tmp[0]
                reward_tmp = data_tmp[1]
                terminated = data_tmp[2]
                self.cumulative_reward+=reward_tmp
                self.iter+=1
                
                self.env.render()
                
                # sleep based on rate
                if self.last_t is not None:
                    sleep_t = np.maximum(0,1.0/self.rate-(time.time()-self.last_t))
                    time.sleep(sleep_t)
                    # print("iter ",self.iter, " ",np.round(time.time()-self.last_t,2)," (s)")
                else:
                    print("iter ",self.iter)
                self.last_t = time.time()

                self.advanceFrame = False
                if terminated or self.episode_over:
                    print("\n\nEpisode Terminated. Reward: ",self.cumulative_reward)
                    self.episode_running = False
                    self.pausemode = False
                    self.episode_over = False

                    if self.args.plotlikelihood:
                        self.likelihoodplotter(self.episode_likelihoods,self.assist_names)

                    file = open('forecasts.pkl', 'wb')

                    pickle.dump(self.forecasts,file)
                    file.close()

if __name__ == '__main__':
    savelocation = project_path+'/saved_policies/diffusion_ur5e_4-22-24_NO_ACTION_NOISE.pkl'
    
    parser = argparse.ArgumentParser()
    parser.add_argument('-p', '--plotlikelihood', action='store_true', help='plot likelihoods')
    parser.add_argument('-o', '--optimal', action='store_true', help='use optimal action for policy (but not forecast)')
    parser.add_argument('-f', '--showforecast', action='store_true', help='send forecast to rviz')

    args = parser.parse_args()

    plot_likelihoods_after_episode = False

    # policy = KDTreeStochastic()
    # policy = TimeDependent()
    policy = Diffusion()

    # env = Uncerpentine()
    # env = MachineTending()
    env = UR5ETending()

    rate = 10 # Hz
    policy.load(savelocation)

    beta = 0.001**2.0
    sigmoid = 0.001
    filter_alpha = 1.0
    sticky = 0.0000
    delta = 0.046 #0.005
    alpha = [1.0, 1.0-3*delta,  1.0-1.0*delta,  1.0-2.5*delta]
    estimate_horizon = 12
    action_chunking = 8

    na = NoAssist(beta=beta, sigmoid=sigmoid, alpha=alpha[0], num_actions=policy.action_dim, horizon=estimate_horizon, mins=policy.action_mins, maxs=policy.action_maxs)
    tele = Teleoperation(beta=beta, sigmoid=sigmoid, alpha=alpha[1], num_actions=policy.action_dim, horizon=estimate_horizon, mins=policy.action_mins, maxs=policy.action_maxs)
    disc = Discrete(k=2, beta=beta, sigmoid=sigmoid,alpha=alpha[2], num_actions=policy.action_dim, horizon=estimate_horizon, mins=policy.action_mins, maxs=policy.action_maxs)
    corr = Corrections(k=1, beta=beta, sigmoid=sigmoid,alpha=alpha[3], num_actions=policy.action_dim, horizon=estimate_horizon, mins=policy.action_mins, maxs=policy.action_maxs)

    assisters = [na, tele, disc, corr]

    likelihoodprocessor = StickyFilter(filter_alpha=filter_alpha, sticky=sticky)

    # thres = 0.646
    # beta = 0.001**2.0
    # sigmoid = 0.001
    # alpha = [1.0]*4
    # assisters = [NoAssist(beta=beta,sigmoid=sigmoid,alpha=alpha[0]), Teleoperation(beta=beta,sigmoid=sigmoid,alpha=alpha[1]), Discrete(k=2,beta=beta,sigmoid=sigmoid,alpha=alpha[2]), Corrections(k=1,beta=beta,sigmoid=sigmoid,alpha=alpha[3])]
    # likelihoodprocessor = TradedTeleop(assisters=assisters,thres=thres)

    partid = None
    method = None
    assistanceRunner = RunAssistance(savelocation, policy, env, assisters, likelihoodprocessor, rate, action_chunking, partid, method, args)
    assistanceRunner.run()
    
