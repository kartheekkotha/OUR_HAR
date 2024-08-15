# import os
# import numpy as np

# '''
# Function adapted from: https://github.com/kenziyuliu/Unofficial-DGNN-PyTorch 
# '''


# sets = {
#     'val', 'train'
# }

# # datasets= {'kinetics'} if kinetics is used
# datasets = {
#     'xsub', 'xview'
# }

# for dataset in datasets:
#     for set in sets:
#         print(dataset, set)
#         data_jpt = np.load('../../data/ntu/{}/{}_data_joint.npy'.format(dataset, set))
#         print(len(data_jpt))
#         data_bone = np.load('../../data/ntu/{}/{}_data_bone.npy'.format(dataset, set))
#         print(len(data_bone))
#         N, C, T, V, M = data_jpt.shape
#         data_jpt_bone = np.concatenate((data_jpt, data_bone), axis=1)
#         np.save('../../data/ntu/{}/{}_data_joint_bones.npy'.format(dataset, set), data_jpt_bone)

import os
import numpy as np
import math

sets = {
    'val', 'train'
}

# datasets= {'kinetics'} if kinetics is used
# datasets = {
#     'xsub', 'xview'
# }
datasets = {
    'xsub', 'xset'
}


# sets = {
#     'val'
# }

# datasets = {
#     'xview'
# }

chunk_size = 1000  # Define your chunk size here

for dataset in datasets:
    for set in sets:
        print(dataset, set)
        data_jpt = np.load('../new_data_processed/{}/{}_joint_120.npy'.format(dataset, set), mmap_mode='r')
        print(len(data_jpt))
        data_bone = np.load('../new_data_processed_bones/{}/{}_data_bone.npy'.format(dataset, set), mmap_mode='r')
        print(len(data_bone))
        N, C, T, V, M = data_jpt.shape
        num_chunks = math.ceil(N / chunk_size)  # Use math.ceil to round up

        output_file = '../new_data_processed_bones/{}/{}_data_joint_bones.npy'.format(dataset, set)

        # Create a memory-mapped array for the output data
        output_data = np.lib.format.open_memmap(output_file, mode='w+', shape=(N, C*2, T, V, M), dtype=data_jpt.dtype)

        # Iterate over chunks and save to the memory-mapped array
        for i in range(num_chunks):
            start = i * chunk_size
            end = min((i + 1) * chunk_size, N)

            # Concatenate and save
            output_data[start:end] = np.concatenate((data_jpt[start:end], data_bone[start:end]), axis=1)

        # Close the memory-mapped array
        del output_data
