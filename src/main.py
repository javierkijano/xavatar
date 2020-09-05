


import sys, os, shutil
import numpy as np
from matplotlib import pyplot as plt
import cv2
import lib.video_utils as vu
from skimage import io
from lib.optical_flow_utils import video_interpolation_OF
from lib.optical_flow_utils import estimate_framesPair_OF
from lib.optical_flow_utils import sample_representative_frames
from itertools import combinations, permutations
from lib.astar import astar_planner
from skimage.color import rgb2gray


print('... stating program')

if False:
    VIDEOS_FILE_NAME = ['seq_0.mp4', 'seq_1.mp4', 'seq_2.mp4']
    frames = vu.videos2frames('data/test_videos/test_videos_b', 'data/test_videos_b', VIDEOS_FILE_NAME, frameRate=10, rescaleFactor=0.25, save=True)


# select video frames
SEQUENCE_NUM_FRAMES_LENGTH = 10
SEQUENCE_NUM_FRAMES_OFFSET = 1
SELECTED_FRAMES_IDS = [0, 9]

# reading seq_init
SEQUENCE_NUM_FRAMES_LENGTH = 65
SEQUENCE_NUM_FRAMES_OFFSET = 0
SELECTED_FRAMES_IDS = [0, 64]
sequence_frames_ids = [id + SEQUENCE_NUM_FRAMES_OFFSET for id in list(range(SEQUENCE_NUM_FRAMES_LENGTH))]
frames_rgb_fileNames = ['data/test_videos_b/seq_0/frame_%d.jpg' % sequence_frame_id for sequence_frame_id in sequence_frames_ids]
frames_rgb_seq_init = io.imread_collection(frames_rgb_fileNames, conserve_memory=True)
# reading seq_end
SEQUENCE_NUM_FRAMES_LENGTH = 81
SEQUENCE_NUM_FRAMES_OFFSET = 0
SELECTED_FRAMES_IDS = [0, 80]
sequence_frames_ids = [id + SEQUENCE_NUM_FRAMES_OFFSET for id in list(range(SEQUENCE_NUM_FRAMES_LENGTH))]
frames_rgb_fileNames = ['data/test_videos_b/seq_1/frame_%d.jpg' % sequence_frame_id for sequence_frame_id in sequence_frames_ids]
frames_rgb_seq_end = io.imread_collection(frames_rgb_fileNames, conserve_memory=True)
# reading seq_link
SEQUENCE_NUM_FRAMES_LENGTH = 148
SEQUENCE_NUM_FRAMES_OFFSET = 0
SELECTED_FRAMES_IDS = [0, 147]
sequence_frames_ids = [id + SEQUENCE_NUM_FRAMES_OFFSET for id in list(range(SEQUENCE_NUM_FRAMES_LENGTH))]
frames_rgb_fileNames = ['data/test_videos_b/seq_2/frame_%d.jpg' % sequence_frame_id for sequence_frame_id in sequence_frames_ids]
frames_rgb_seq_link = io.imread_collection(frames_rgb_fileNames, conserve_memory=True)


a = sample_representative_frames(frames_rgb_seq_link, frames_rgb_seq_init[-1], frames_rgb_seq_end[0])
a=0
if True:
    selected_frames = [frames_rgb[selected_frame_id] for selected_frame_id in SELECTED_FRAMES_IDS]
    io.imshow_collection(selected_frames)
a=0

frames = [rgb2gray(frame) for frame in frames_rgb]
# estimate correc frames order
astar_pl = astar_planner(frames)
f_heuristic = estimate_framesPair_OF
path = astar_pl.search(0, SEQUENCE_NUM_FRAMES_LENGTH-1, f_heuristic, aproximate_frames_number=10)
print(astar_pl.costMatrix_g_inc)
print(astar_pl.costMatrix_g)
print(astar_pl.costMatrix_h)
print(astar_pl.costMatrix_f_hat)
io.imshow_collection([frames_rgb[image_i] for image_i in path])










io.imshow_collection(frames_rgb)

# calculate best sequence
frames_pairs = list(combinations(list(range(len(frames))), 2))
frames_pairs_OP = estimate_OF(frames, frames_pairs)

a = video_interpolation_OF(frames[0], frames[1])

frames_sequences = list(permutations(list(range(len(frames)))))
for frames_sequence in frames_sequences:
    frames_sequence_pairs = [(frames_sequence[frames_sequence_i], frames_sequence[frames_sequence_i+1]) for frames_sequence_i, _ in enumerate(frames_sequence[:-1])]
    frames_sequence_pairs = [frames_sequence_pair if frames_sequence_pair[0] <= frames_sequence_pair[1] else frames_sequence_pair[::-1] for frames_sequence_pair in frames_sequence_pairs]
    frames_sequence_sumPO = np.array(frames_pairs_OP)[[frames_pairs.index(frames_sequence_pair) for frames_sequence_pair in frames_sequence_pairs]].sum()
    print('%s --> %f'%(str(frames_sequence), frames_sequence_sumPO))

# interpolate video using optical flow
# a = video_interpolation_OF(frames[0], frames[1])



a=0
#io.imshow(frame_0)
#plt.xticks([]), plt.yticks([])  # to hide tick values on X and Y axis
#plt.show()

# frame_1 = io.imread('data/seq_1/frame_21.jpg')
# io.imshow(frame_0)
# plt.xticks([]), plt.yticks([])  # to hide tick values on X and Y axis
# plt.show()




#ptical_flow_estimation(image0, image1):


a=0
    # if False:
    #     io.imshow(frame_rgb)
    #     plt.xticks([]), plt.yticks([])  # to hide tick values on X and Y axis
    #     plt.show()

#TODO: para ordenar  de una secuancia de imagenes desde una imagen I0 a una imagen It, puedo plantearlo
# como un problema de planificacion donde creo un arbol en el que en cada nodo se crean branches con todas las
# imagenes candidatas y en cada uno de estas nuevas branlches con las que quedan por seleccionar. el camino elegido
# es el path que minimiza la suma sqaure optical flows.  Las branches crecen hasta que el optical flow con la imgaen
# final está por debajo de un determinado threshold.