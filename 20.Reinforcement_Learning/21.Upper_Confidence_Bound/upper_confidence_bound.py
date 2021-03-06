# -*- coding: utf-8 -*-
"""
Created on Sun Mar 20 11:08:16 2022

@author: Jay
"""

import pandas as pd
import matplotlib.pyplot as plt

import math


ads_data = pd.read_csv(
    "./20.Reinforcement_Learning/21.Upper_Confidence_Bound/Ads_CTR_Optimisation.csv"
)

# Implementing UCB
N = 10000
d = 10
numbers_of_selections = [0] * d
sums_of_rewards = [0] * d
ads_selected = []
total_reward = 0
for n in range(0, N):
    ad = 0
    max_upper_bound = 0
    for i in range(0, d):
        if (numbers_of_selections[i] > 0):
            average_reward = sums_of_rewards[i] / numbers_of_selections[i]
            delta_i = math.sqrt(3 / 2 * math.log(n + 1) / numbers_of_selections[i])
            upper_bound = average_reward + delta_i
        else:
            upper_bound = 1e400
        if upper_bound > max_upper_bound:
            max_upper_bound = upper_bound
            ad = i
    ads_selected.append(ad)
    reward = ads_data.values[n, ad]
    numbers_of_selections[ad] += 1
    sums_of_rewards[ad] += reward
    total_reward += reward

# Visualising the results
plt.hist(ads_selected)
plt.title("Histogram of ads selections")
plt.xlabel("Ads")
plt.ylabel("Number of times eatch ad was selected")
plt.show()
