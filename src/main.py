


import sys, os, shutil
import numpy as np
from matplotlib import pyplot as plt
import cv2
import lib.video_utils as vu

from skimage import io
from lib.optical_flow_utils import video_interpolation_OF
from lib.optical_flow_utils import estimate_OF

from itertools import combinations, permutations


#frames = vu.videos2frames('data/seq_videos', 'data', VIDEOS_FILE_NAME, frameRate=10, rescaleFactor=0.25, save=True)


frames_fileName = ['data/seq_1/frame_%d.jpg'%frame for frame in list(range(4))]
frames = io.imread_collection(frames_fileName, conserve_memory=True)
#io.imshow_collection(frames_ic)

# calculate best sequence
frames_pairs = list(combinations(list(range(len(frames))), 2))
frames_pairs_OP = estimate_OF(frames, frames_pairs)

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