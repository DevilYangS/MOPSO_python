#encoding: utf-8
import numpy as np
import random
from  public import NDsort

def update_v(v_,v_min,v_max,in_,in_pbest,in_gbest):
    #更新速度ٶ�ֵ

    w = 0.4
    N,D = v_.shape
    r1 = np.tile(np.random.rand(N,1),(1,D))
    r2 = np.tile(np.random.rand(N,1),(1,D))

    v_temp = w*v_ + r1*(in_pbest-in_) + r2*(in_gbest-in_)

    #速度边界处理ֵ
    Upper = np.tile(v_max,(N,1))
    Lower = np.tile(v_min,(N,1))
    v_temp = np.maximum(np.minimum(Upper,v_temp), Lower) # v不存在上下限，因此是否有必要进行限制
    return v_temp


def update_in(in_,v_,in_min,in_max):
    N, D = in_.shape
    #更新位置
    in_temp = in_ + v_
    #越界处理ֵ
    Upper = np.tile(in_max, (N, 1))
    Lower = np.tile(in_min, (N, 1))
    in_temp = np.maximum(np.minimum(Upper,in_temp), Lower)
    return in_temp


def update_pbest(in_,fitness_,in_pbest,out_pbest):
    temp = out_pbest - fitness_
    Dominate = np.int64(np.any(temp< 0, axis=1)) - np.int64(np.any(temp> 0, axis=1))

    remained_1 = Dominate==-1
    out_pbest[remained_1] = fitness_[remained_1]
    in_pbest[remained_1] = in_[remained_1]

    remained_2 = Dominate == 0
    remained_temp_rand = np.random.rand(len(Dominate),)<0.5
    remained_final = remained_2 & remained_temp_rand
    out_pbest[remained_final] = fitness_[remained_final]
    in_pbest[remained_final] = in_[remained_final]
    return in_pbest,out_pbest



def update_archive_1(in_,fitness_,archive_in,archive_fitness,thresh,mesh_div):
    ##首先，计算当前粒子群的pareto边界，将边界粒子加入到存档archiving中
    total_Pop = np.vstack((archive_in,in_))
    total_Func = np.vstack((archive_fitness,fitness_))

    FrontValue_1_index = NDsort.NDSort(total_Func, total_Pop.shape[0])[0]==1
    FrontValue_1_index = np.reshape(FrontValue_1_index,(-1,))
    archive_in =total_Pop[FrontValue_1_index]
    archive_fitness = total_Func[FrontValue_1_index]

    if archive_in.shape[0] > thresh:

        Del_index = Delete(archive_fitness,archive_in.shape[0]-thresh,mesh_div)
        archive_in  = np.delete(archive_in,Del_index,0)
        archive_fitness = np.delete(archive_fitness,Del_index,0)
    return archive_in,archive_fitness

def Delete(archiving_fit,K,mesh_div):
    Nop, num_obj = archiving_fit.shape

    # %% Calculate the grid location of each solution
    fmax = np.max(archiving_fit, axis=0)
    fmin = np.min(archiving_fit, axis=0)
    d = (fmax - fmin) / mesh_div
    fmin = np.tile(fmin, (Nop, 1))
    d = np.tile(d, (Nop, 1))
    Gloc = np.floor((archiving_fit - fmin) / d)
    Gloc[Gloc >= mesh_div] = mesh_div - 1
    Gloc[np.isnan(Gloc)] = 0

    # Detect the grid of each solution belongs to
    _, _, Site = np.unique(Gloc, return_index=True, return_inverse=True, axis=0)


    # Calculate the crowd degree of each grid
    CrowdG = np.histogram(Site, np.max(Site)+1)[0]
    CrowdG_ =CrowdG.copy()

    Del_index = np.zeros(Nop,)==1

    while np.sum(Del_index)<K:
        maxGrid = np.where(CrowdG == max(CrowdG))[0]
        Temp = np.random.randint(0,len(maxGrid))
        Grid = maxGrid[Temp]

        InGrid = np.where(Site==Grid)[0]


        Temp = np.random.randint(0,len(InGrid))
        p = InGrid[Temp]
        Del_index[p] = True
        Site[p] = -100
        CrowdG[Grid] = CrowdG[Grid] -1

    return np.where(Del_index==1)[0]







def update_gbest_1(archiving_in,archiving_fit,mesh_div,particals):
    Nop,num_obj =  archiving_fit.shape

    # %% Calculate the grid location of each solution
    fmax = np.max(archiving_fit,axis=0)
    fmin = np.min(archiving_fit,axis=0)
    d = (fmax-fmin)/mesh_div
    fmin = np.tile(fmin,(Nop,1))
    d = np.tile(d,(Nop,1))
    Gloc = np.floor((archiving_fit-fmin)/d)
    Gloc[Gloc>=mesh_div] = mesh_div-1
    Gloc[np.isnan(Gloc)] = 0

    #Detect the grid of each solution belongs to
    _,_,Site = np.unique(Gloc, return_index=True,return_inverse=True,axis=0)

    #Calculate the crowd degree of each grid
    CrowdG =  np.histogram(Site,np.max(Site)+1)[0]

    #  Roulette-wheel 1/Fitnessselection
    TheGrid = RouletteWheelSelection(particals,CrowdG)

    ReP = np.zeros(particals,)
    for i in range(particals):
        InGrid = np.where(Site==TheGrid[i])[0]
        Temp = np.random.randint(0,len(InGrid))
        ReP[i] = InGrid[Temp]
    ReP = np.int64(ReP)
    return archiving_in[ReP],archiving_fit[ReP]

def RouletteWheelSelection(N,Fitness):

    Fitness = np.reshape(Fitness,(-1,))
    Fitness  = Fitness + np.minimum(np.min(Fitness),0)
    Fitness = np.cumsum(1/Fitness)
    Fitness = Fitness/np.max(Fitness)
    index = np.sum(np.int64(~(np.random.rand(N,1)<Fitness)), axis=1)

    return index







