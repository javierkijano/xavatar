

# get individual frames from video at an specific framerate
def get_frames(vidcap, frameRate=30, rescaleFactor=1.0):

    import cv2
    from skimage.transform import rescale
    sec = 0
    frames = []
    while True:
        vidcap.set(cv2.CAP_PROP_POS_MSEC, sec * 1000)
        success, frame = vidcap.read()
        if success:
            if rescaleFactor != 1.0:
                frame = rescale(frame, rescaleFactor, preserve_range=True, multichannel=True)
                frame = frame.astype('uint8')
            frames.append(frame)
            sec = sec + 1/frameRate
        else:
            return frames

def video2frames(
        dir_inputVideo, dir_outputVideo, videoFileName,
        ext_outputImage='jpg', frameRate=30, rescaleFactor=1.0, save=True):

    import os
    import shutil
    import cv2
    from skimage import io

    if os.path.exists(dir_outputVideo):
        shutil.rmtree(dir_outputVideo)
    os.makedirs(dir_outputVideo)
    vidcap = cv2.VideoCapture(dir_inputVideo + '/' + videoFileName)
    print('... getting frames from video ...')
    frames = get_frames(vidcap, frameRate=frameRate, rescaleFactor=rescaleFactor)
    if save:
        for frame_i, frame in enumerate(frames):
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            io.imsave(dir_outputVideo + "/frame_" + str(frame_i) + '.' + ext_outputImage, frame_rgb)
            print('saving %s/%d' % (videoFileName, frame_i))
    return frames

def videos2frames(
        dir_inputVideos, dir_outputVideos, videosFilesName,
        ext_outputImage='jpg', frameRate=30, rescaleFactor=1.0, save=True):

    import os
    frames = {}
    for videoFileName in videosFilesName:
        dir_outputVideo = dir_outputVideos + '/' + os.path.splitext(videoFileName)[0]
        frames[videoFileName] = \
            video2frames(dir_inputVideos, dir_outputVideo, videoFileName,
                         ext_outputImage=ext_outputImage, frameRate=frameRate, rescaleFactor=rescaleFactor, save=save)


def frames2video(dir, ext='.jpg', frameRate=10):

    import glob
    import cv2
    import os
    img_array = []
    for filename in sorted(glob.glob(dir + '/*' + ext), key=os.path.getmtime):
        img = cv2.imread(filename)
        height, width, layers = img.shape
        size = (width, height)
        img_array.append(img)

    outputFile = dir + '/' + os.path.basename(os.path.normpath(dir)) + '_' + str(frameRate) + '.avi'
    videocap = cv2.VideoWriter(outputFile, cv2.VideoWriter_fourcc(*'DIVX'), frameRate, size)

    for i in range(len(img_array)):
        videocap.write(img_array[i])
    videocap.release()
    return videocap







