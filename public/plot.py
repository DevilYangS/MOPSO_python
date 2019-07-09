#encoding: utf-8
import numpy as np
import os
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
#from mpl_toolkits.mplot3d import Axes3D
import time
class Plot_pareto:
    def __init__(self):
        self.start_time = time.time()

    def show(self,in_,fitness_,archive_in,archive_fitness,i):
        #共3个子图，第1、2/子图绘制输入坐标与适应值关系，第3图展示pareto边界的形成过程
        fig = plt.figure('第' + str(i + 1) + '次迭代')
        # fig = plt.figure('第'+str(i+1)+'次迭代',figsize = (17,5))
        # ax1 = fig.add_subplot(131, projection='3d')
        # ax1.set_xlabel('input_x1')
        # ax1.set_ylabel('input_x2')
        # ax1.set_zlabel('fitness_y1')
        # ax1.plot_surface(self.x1,self.x2,self.y1,alpha = 0.6)
        # ax1.scatter(in_[:,0],in_[:,1],fitness_[:,0],s=20, c='blue', marker=".")
        # ax1.scatter(archive_in[:,0],archive_in[:,1],archive_fitness[:,0],s=50, c='red', marker=".")
        # ax2 = fig.add_subplot(132, projection='3d')
        # ax2.set_xlabel('input_x1')
        # ax2.set_ylabel('input_x2')
        # ax2.set_zlabel('fitness_y2')
        # ax2.plot_surface(self.x1,self.x2,self.y2,alpha = 0.6)
        # ax2.scatter(in_[:,0],in_[:,1],fitness_[:,1],s=20, c='blue', marker=".")
        # ax2.scatter(archive_in[:,0],archive_in[:,1],archive_fitness[:,1],s=50, c='red', marker=".")

        ax3 = fig.add_subplot(111)#133
        # ax3.set_xlim((0,1))
        # ax3.set_ylim((0,1))
        ax3.set_xlabel('fitness_y1')
        ax3.set_ylabel('fitness_y2')
        ax3.scatter(fitness_[:,0],fitness_[:,1],s=10, c='blue', marker=".")
        ax3.scatter(archive_fitness[:,0],archive_fitness[:,1],s=30, c='red', marker=".",alpha = 1.0)
        plt.show()
        # plt.savefig('./img_txt/'+str(i+1)+'.png')
        # print ('第'+str(i+1)+'次迭代的图片保存于 img_txt 文件夹')
        # print ('第'+str(i+1)+'次迭代, time consuming: ',np.round(time.time() - self.start_time, 2), "s")
        plt.ion()
        # plt.close()
