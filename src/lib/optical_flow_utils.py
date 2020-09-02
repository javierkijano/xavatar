def estimate_OF(images, images_pairs):

    from skimage.color import rgb2gray
    from skimage.registration import optical_flow_tvl1
    import numpy as np
    from skimage.color import rgb2gray

    #from joblib import Parallel, delayed

    images = [rgb2gray(image) for image in images]
    images_pairs_OP = []
    for images_pair in images_pairs:
        image_0 = images[images_pair[0]]
        image_1 = images[images_pair[1]]
        v, u = optical_flow_tvl1(image_0, image_1)
        images_pairs_OP.append(np.mean(np.sqrt(np.power(u,2)+np.power(v,2))))
    return images_pairs_OP



def video_interpolation_OF(image0_rgb, image1_rgb):

    import numpy as np
    from matplotlib import pyplot as plt
    from skimage.color import rgb2gray
    from skimage.data import stereo_motorcycle
    from skimage.transform import warp
    from skimage.registration import optical_flow_tvl1
    from skimage import img_as_float


    # Convert the images to gray level: color is not supported.
    image0 = rgb2gray(image0_rgb)
    image1 = rgb2gray(image1_rgb)

    # --- Compute the optical flow
    v, u = optical_flow_tvl1(image0, image1)

    # --- Use the estimated optical flow for registration

    nr, nc = image0.shape

    row_coords, col_coords = np.meshgrid(np.arange(nr), np.arange(nc),
                                         indexing='ij')

    image1_warp_grey = warp(image1, np.array([row_coords + v, col_coords + u]),
                       mode='nearest')
    image1_warp_r = warp(img_as_float(image1_rgb[..., 0]), np.array([row_coords + v, col_coords + u]),
                         mode='nearest')
    image1_warp_g = warp(img_as_float(image1_rgb[..., 1]), np.array([row_coords + v, col_coords + u]),
                         mode='nearest')
    image1_warp_b = warp(img_as_float(image1_rgb[..., 2]), np.array([row_coords + v, col_coords + u]),
                         mode='nearest')

    a=0

    # build an RGB image with the unregistered sequence
    seq_im = np.zeros((nr, nc, 3))
    seq_im[..., 0] = image1
    seq_im[..., 1] = image0
    seq_im[..., 2] = image0

    # build an RGB image with the registered sequence
    reg_im = np.zeros((nr, nc, 3))
    reg_im[..., 0] = image1_warp_grey
    reg_im[..., 1] = image0
    reg_im[..., 2] = image0

    # build an RGB image with the all three independent channels
    reg_im_rgb = np.zeros((nr, nc, 3))
    reg_im_rgb[..., 0] = image1_warp_r
    reg_im_rgb[..., 1] = image1_warp_g
    reg_im_rgb[..., 2] = image1_warp_b

    # build an RGB image with the registered sequence
    target_im = np.zeros((nr, nc, 3))
    target_im[..., 0] = image0
    target_im[..., 1] = image0
    target_im[..., 2] = image0

    # --- Show the result

    fig, ((ax0, ax1, ax2), (ax3, ax4, ax5)) = plt.subplots(2, 3, figsize=(5, 10))

    ax0.imshow(image0_rgb)
    ax0.set_title("Image t0", fontSize=10)
    ax0.set_axis_off()

    ax1.imshow(image1_rgb)
    ax1.set_title("Image t1", fontSize=10)
    ax1.set_axis_off()

    ax2.imshow(seq_im)
    ax2.set_title("Unregistered sequence", fontSize=10)
    ax2.set_axis_off()

    ax3.imshow(reg_im)
    ax3.set_title("Registered seq", fontSize=10)
    ax3.set_axis_off()

    ax4.imshow(reg_im_rgb)
    ax4.set_title("Registered seq RGB", fontSize=10)
    ax4.set_axis_off()

    fig.tight_layout()
    plt.show()