#encoding: utf-8
import random
import numpy as np
from public import pareto,NDsort


def init_designparams(particals,in_min,in_max):
    in_dim = len(in_max)     #输入参数维度

    in_temp = np.random.uniform(0,1,(particals,in_dim))*(in_max-in_min)+in_min

    return in_temp

def init_v(particals,v_max,v_min):
    v_dim = len(v_max)     #输入参数维度
    # v_ = np.random.uniform(0,1,(particals,v_dim))*(v_max-v_min)+v_min

    v_ = np.zeros((particals,v_dim))
    return v_

def init_pbest(in_,fitness_):
    return in_,fitness_

def init_archive(in_,fitness_):

    FrontValue_1_index = NDsort.NDSort(fitness_, in_.shape[0])[0]==1
    FrontValue_1_index = np.reshape(FrontValue_1_index,(-1,))

    curr_archiving_in=in_[FrontValue_1_index]
    curr_archiving_fit=fitness_[FrontValue_1_index]

    # pareto_c = pareto.Pareto_(in_,fitness_)
    # curr_archiving_in_,curr_archiving_fit_ = pareto_c.pareto()
    return curr_archiving_in,curr_archiving_fit


