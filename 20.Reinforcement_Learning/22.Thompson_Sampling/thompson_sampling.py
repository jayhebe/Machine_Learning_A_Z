# -*- coding: utf-8 -*-
"""
Created on Sun Mar 20 13:12:36 2022

@author: Jay
"""

import pandas as pd
import matplotlib.pyplot as plt

import random

ads_data = pd.read_csv(
    "./20.Reinforcement_Learning/22.Thompson_Sampling/Ads_CTR_Optimisation.csv"
)

# Implementing Thompson
N = 10000
d = 10
numbers_of_rewards_1 = [0] * d
numbers_of_rewards_0 = [0] * d
ads_selected = []
total_reward = 0
for n in range(0, N):
    ad = 0
    max_random = 0
    for i in range(0, d):
        random_beta = random.betavariate(
            numbers_of_rewards_1[i] + 1,
            numbers_of_rewards_0[i] + 1
        )
        if random_beta > max_random:
            max_random = random_beta
            ad = i
    ads_selected.append(ad)
    reward = ads_data.values[n, ad]
    if reward == 1:
        numbers_of_rewards_1[ad] += 1
    else:
        numbers_of_rewards_0[ad] += 1
    total_reward += reward

# Visualising the results
plt.hist(ads_selected)
plt.title("Histogram of ads selections")
plt.xlabel("Ads")
plt.ylabel("Number of times eatch ad was selected")
plt.show()
