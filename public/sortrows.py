import numpy as np

#Programmed By Yang Shang shang  Email：yangshang0308@gmail.com  GitHub: https://github.com/DevilYangS/codes
def sortrows(Matrix, order = "ascend"):
    # 默认先以第一列值大小  对  行  进行排序，若第一列值相同，则按照第二列 值，以此类推,返回排序结果 及对应 索引 （Reason: list.sort() 仅仅返回 排序后的结果， np.argsort() 需要多次 排序，其中、
    #  np.lexsort()的操作对象 等同于 sortcols ，先排以最后一行 对  列  进行排序，然后以倒数第二列，以此类推. np.lexsort((d,c,b,a)来对[a,b,c,d]进行排序、其中 a 为一列向量 ）
    Matrix_temp = Matrix[:, ::-1] #因为np.lexsort() 默认从最后一行 对  列  开始排序，需要将matrix 反向 并 转置
    Matrix_row = Matrix_temp.T
    if order == "ascend":
        rank = np.lexsort(Matrix_row)
    elif order == "descend":
        rank = np.lexsort(-Matrix_row)
    Sorted_Matrix = Matrix[rank,:] # Matrix[rank] 也可以
    return Sorted_Matrix, rank