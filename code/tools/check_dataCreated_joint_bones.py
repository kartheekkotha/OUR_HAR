import os
import numpy as np

sets = {
    'val'
}

datasets = {
    'xview'
}

chunk_size = 1000  # Define your chunk size here

def check_similar(data1, data2):
    return np.allclose(data1, data2, rtol=1e-3, atol=1e-3)

for dataset in datasets:
    for set in sets:
        print(dataset, set)
        data_jpt = np.load('../../data/ntu/{}/{}_data_joint.npy'.format(dataset, set), mmap_mode='r')
        print(len(data_jpt))
        data_bone = np.load('../../data/ntu/{}/{}_data_bone.npy'.format(dataset, set), mmap_mode='r')
        print(len(data_bone))
        data_joint_bone = np.load('../../data/ntu/{}/{}_data_joint_bones.npy'.format(dataset, set), mmap_mode='r')
        N, C, T, V, M = data_jpt.shape
        print('The shape of data_joint ',N, C, T, V, M )
        N1, C1, T1, V1, M1 = data_bone.shape
        print('The shape of data_bone',N1, C1, T1, V1, M1 )
        N2,C2, T2, V2, M2 = data_joint_bone.shape
        print('The shape of data_joint_bone ',N2, C2, T2, V2, M2 )
        # print(data_joint_bone[0])
        # print(data_joint_bone[1000])

