from re import I
import numpy as np
from utils.data_utils import *


data = super_average_data(process_data())

GPT2_measures = []
regressions = []
for clause_type in data:
    for item in data[clause_type]:
        for region_type in data[clause_type][item]:
            GPT2_measures.append(data[clause_type][item][region_type]['GPT2'])
            regressions.append(data[clause_type][item][region_type]['ro'])

