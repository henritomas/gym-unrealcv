#!/usr/bin/env python

import os
import gym
import matplotlib
import matplotlib.pyplot as plt
import time
import itertools
import sys
import argparse
import numpy as np
from scipy.interpolate import pchip
from gym.wrappers import monitor as monitoring

class MultiLivePlot(object):
    def __init__(self, outdir, data_key='episode_rewards', line_color='blue'):
        """
        Liveplot renders a graph of either episode_rewards or episode_lengths
        Args:
            outdir (outdir): Monitor output file location used to populate the graph
            data_key (Optional[str]): The key in the json to graph (episode_rewards or episode_lengths).
            line_color (Optional[dict]): Color of the plot.
        """
        #data_key can be set to 'episode_lengths'
        self.outdir = outdir
        self._last_data = None
        self.data_key = data_key
        self.line_color = line_color

        #styling options
        matplotlib.rcParams['toolbar'] = 'None'
        plt.style.use('ggplot')
        plt.xlabel("steps")
        plt.ylabel("cumulated episode rewards")
        fig = plt.gcf().canvas.set_window_title('averaged_simulation_graph')
        matplotlib.rcParams.update({'font.size': 15})

    def plot(self, full=True, dots=False, average=0, interpolated=0):

        colors = ['blue', 'green', 'red', 'yellow', 'magenta']

        for idx, path in enumerate(self.outdir):
            print(path)
            results = monitoring.load_results(path)

            data =  results[self.data_key]
            steps = results['episode_lengths']
            #print steps
            count_steps = 0
            for i in range(len(steps)):
                count_steps += steps[i]
                steps[i] = count_steps

            avg_data = []

            if full:
                plt.plot(steps,data, color='blue')
            if dots:
                plt.plot(steps,data, '.', color='black')
            if average > 0:
                average = int(average)
                for i, val in enumerate(data):
                    '''if i%average==0:
                        if (i+average) < len(data)+average:
                            avg =  sum(data[i:i+average])/average
                            avg_data.append(avg)'''
                    if i < average:
                        avg = np.array(data[:average]).mean()
                    else:
                        avg = np.array(data[(i-average):i]).mean()
                    avg_data.append(avg)
                #new_data = expand(avg_data,average)
                plt.plot(steps,avg_data, color=colors[idx], linewidth=1)
                highest_avg = np.max(avg_data)
                print('\n{} HIGHEST AVG: {:.2f}'.format(path, highest_avg))

                for s_idx, s in enumerate(avg_data):
                    if s > highest_avg - 0.05:
                        print('Converged at {} timesteps'.format(steps[s_idx]))
                        break

        # pause so matplotlib will display
        # may want to figure out matplotlib animation or use a different library in the future
        plt.legend(['VGG-Lite + Augm', 
        'ViT-64 + Augm',
        ])

        plt.pause(0.000001)


def expand(lst, n):
    lst = [[i]*n for i in lst]
    lst = list(itertools.chain.from_iterable(lst))
    return lst

def pause():
    programPause = input("Press the <ENTER> key to finish...")

if __name__ == '__main__':

    #print args.path
    all_paths = [
        '/home/eee198/Documents/gym-unrealcv/example/dqn_torch/saved_logs/tmp_24_vgglite_aap',
        '/home/eee198/Documents/gym-unrealcv/example/dqn_torch/saved_logs/tmp_22_vit64_aap_3e-5',
    ]
    plotter = MultiLivePlot(all_paths)
    plotter.plot(full=False, dots=False, average=200, interpolated=0)

    pause()
