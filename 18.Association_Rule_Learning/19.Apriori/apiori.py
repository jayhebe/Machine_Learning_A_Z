# -*- coding: utf-8 -*-
"""
Created on Fri Mar 18 15:49:22 2022

@author: Jay
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


market_data = pd.read_csv(
    "./18.Association_Rule_Learning/19.Apriori/Market_Basket_Optimisation.csv",
    header=None
)
transactions = []
for i in range(0, market_data.shape[0]):
    transactions.append(
        [str(market_data.iloc[i, j]) for j in range(0, market_data.shape[1])]
    )


from apyori import apriori

rules = apriori(
    transactions,
    min_support=0.003,
    min_confidence=0.2,
    min_lift=3,
    min_length=2
)

# Visualising the results
results = list(rules)
my_results = [list(x) for x in results]
