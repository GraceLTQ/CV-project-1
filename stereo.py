import numpy as np
# ==============No additional imports allowed ================================#


def get_ncc_descriptors(img, patchsize):
    '''
    Prepare normalized patch vectors for normalized cross
    correlation.

    Input:
        img -- height x width x channels image of type float32
        patchsize -- integer width and height of NCC patch region.
    Output:
        normalized -- height* width *(channels * patchsize**2) array

    For every pixel (i,j) in the image, your code should:
    (1) take a patchsize x patchsize window around the pixel,
    (2) compute and subtract the mean for every channel
    (3) flatten it into a single vector
    (4) normalize the vector by dividing by its L2 norm
    (5) store it in the (i,j)th location in the output

    If the window extends past the image boundary, zero out the descriptor

    If the norm of the vector is <1e-6 before normalizing, zero out the vector.

    '''
    # height, width, channels = img.shape
    # normalized = np.zeros(
    #     (height, width, channels * patchsize**2), dtype=np.float32)
    # half_patch = patchsize // 2

    # for i in range(height):
    #     for j in range(width):
    #         patch = np.zeros((patchsize, patchsize, channels),
    #                          dtype=np.float32)
    #         for k in range(channels):
    #             top = max(i - half_patch, 0)
    #             bottom = min(i + half_patch + 1, height)
    #             left = max(j - half_patch, 0)
    #             right = min(j + half_patch + 1, width)
    #             patch_slice = img[top: bottom, left: right, k] - \
    #                 img[top:bottom, left:right, k].mean()
    #             patch[top - i + half_patch: bottom - i + half_patch, left -
    #                   j + half_patch: right - j + half_patch, k] = patch_slice
    #         patch_vector = patch.flatten()
    #         norm = np.linalg.norm(patch_vector)
    #         if norm < 1e-6:
    #             patch_vector = np.zeros_like(patch_vector)
    #         else:
    #             patch_vector /= norm
    #         normalized[i, j] = patch_vector
    # return normalized

    height, width, channels = img.shape
    descriptor_dim = channels * patchsize ** 2
    normalized = np.zeros((height, width, descriptor_dim), dtype=np.float32)

    half_patchsize = patchsize // 2

    for i in range(height):
        for j in range(width):
            # Initialize patch with zeros for cases where the window extends past the image boundary
            patch = np.zeros((patchsize, patchsize, channels),
                             dtype=np.float32)

            # Calculate the coordinates of the patch in the original image
            # and ensure they are within the image boundaries
            top = max(i - half_patchsize, 0)
            bottom = min(i + half_patchsize + 1, height)
            left = max(j - half_patchsize, 0)
            right = min(j + half_patchsize + 1, width)

            # Calculate the coordinates of the patch in the patch array
            patch_top = max(half_patchsize - i, 0)
            patch_bottom = patch_top + (bottom - top)
            patch_left = max(half_patchsize - j, 0)
            patch_right = patch_left + (right - left)

            # Copy the valid part of the window into the patch array
            patch[patch_top:patch_bottom, patch_left:patch_right,
                  :] = img[top:bottom, left:right, :]

            # Subtract the mean for every channel
            mean_subtracted_patch = patch - \
                np.mean(patch, axis=(0, 1), keepdims=True)

            # Flatten the patch into a single vector
            vector = mean_subtracted_patch.flatten()

            # Compute L2 norm of the vector
            norm = np.linalg.norm(vector)

            # Normalize the vector by its L2 norm if norm is >= 1e-6, else zero out
            normalized_vector = vector / \
                norm if norm >= 1e-6 else np.zeros(
                    descriptor_dim, dtype=np.float32)

            # Store the normalized vector in the output
            normalized[i, j, :] = normalized_vector

    return normalized


def compute_ncc_vol(img_right, img_left, patchsize, dmax):
    '''
    Compute the NCC-based cost volume
    Input:
        img_right: the right image, H x W x C
        img_left: the left image, H x W x C
        patchsize: the patchsize for NCC, integer
        dmax: maximum disparity
    Output:
        ncc_vol: A dmax x H x W tensor of scores.

    ncc_vol(d,i,j) should give a score for the (i,j)th pixel for disparity d. 
    This score should be obtained by computing the similarity (dot product)
    between the patch centered at (i,j) in the right image and the patch centered
    at (i, j+d) in the left image.

    Your code should call get_ncc_descriptors to compute the descriptors once.
    '''
    # height, width, _ = img_right.shape
    # # Precompute the NCC descriptors for both images
    # ncc_descriptors_right = get_ncc_descriptors(img_right, patchsize)
    # ncc_descriptors_left = get_ncc_descriptors(img_left, patchsize)

    # # Initialize the NCC volume with zeros
    # ncc_vol = np.zeros((height, width, dmax), dtype=np.float32)

    # for d in range(dmax):
    #     # Shift the right image's descriptors by d
    #     shifted_descriptors = np.roll(ncc_descriptors_right, d, axis=1)
    #     # Zero out the descriptors that have wrapped around
    #     if d > 0:
    #         shifted_descriptors[:, :d, :] = 0

    #     # Compute the dot product between left descriptors and shifted right descriptors for each disparity
    #     for i in range(height):
    #         for j in range(width):
    #             vector_left = ncc_descriptors_left[i, j]
    #             vector_right = shifted_descriptors[i, j]
    #             # Compute NCC using dot product since vectors are already normalized
    #             ncc_vol[i, j, d] = np.dot(vector_left, vector_right)

    # return ncc_vol

    # Compute NCC descriptors for both images
    descriptors_right = get_ncc_descriptors(img_right, patchsize)
    descriptors_left = get_ncc_descriptors(img_left, patchsize)

    # Get image dimensions
    height, width, _ = img_right.shape

    # Initialize the NCC volume
    ncc_vol = np.zeros((dmax, height, width))

    for d in range(dmax):
        for i in range(height):
            for j in range(width):
                # Ensure that the left image index (j + d) does not exceed image width
                if (j + d) < width:
                    # Dot product between descriptors to compute similarity
                    ncc_vol[d, i, j] = np.dot(
                        descriptors_right[i, j, :], descriptors_left[i, j + d, :])
                else:
                    # If the index exceeds the width, assign a score of 0 (indicating no match)
                    ncc_vol[d, i, j] = 0

    return ncc_vol


def get_disparity(ncc_vol):
    '''
    Get disparity from the NCC-based cost volume
    Input: 
        ncc_vol: A dmax X H X W tensor of scores
    Output:
        disparity: A H x W array that gives the disparity for each pixel. 

    the chosen disparity for each pixel should be the one with the largest score for that pixel
    '''
    disparity_map = np.argmax(ncc_vol, axis=2)
    return disparity_map
    # disparity = np.argmax(ncc_vol, axis=0)
    # return disparity
