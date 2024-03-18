import numpy as np
##======================== No additional imports allowed ====================================##


def photometric_stereo_singlechannel(I, L):
    #L is 3 x k
    #I is k x n
    G = np.linalg.inv(L @ L.T) @ L @ I
    # G is  3 x n 
    albedo = np.sqrt(np.sum(G*G, axis=0))

    normals = G/(albedo.reshape((1,-1)) + (albedo==0).astype(float).reshape((1,-1)))
    return albedo, normals


def normalize(normals):
    '''
    Normalizes the vectors in normals
    Input:
        normals: H x W x 3 array of vectors
    
    '''
    h,w,_ = normals.shape
    for r in range(h):
        for c in range(w):
            norm = np.linalg.norm(normals[r,c,:])
            normals[r,c,:] = normals[r,c,:]/norm
    

def photometric_stereo(images, lights):
    '''
        Use photometric stereo to compute albedos and normals
        Input:
            images: A list of N images, each a numpy float array of size H x W x 3
            lights: 3 x N array of lighting directions. 
        Output:
            albedo, normals
            albedo: H x W x 3 array of albedo for each pixel
            normals: H x W x 3 array of normal vectors for each pixel

        Assume light intensity is 1.
        Compute the albedo and normals for red, green and blue channels separately.
        The normals should be approximately the same for all channels, so average the three sets
        and renormalize so that they are unit norm

    '''
    num_lights = len(images) #number of lights, which is the same as number of images
    h,w,_ = images[0].shape   
    num_pixels=h*w #total number of pixels in an image

    albedo = np.zeros((h,w,3))
    normals = np.zeros((3,num_pixels))
    
    for c in range(3): #for each channel
        I = np.zeros((num_lights,num_pixels))
        for row in range(h):
            for col in range(w):
                for img in range(num_lights):
                    I[img,row*w+col] = images[img][row,col,c]
        a,nor = photometric_stereo_singlechannel(I,lights)
        albedo[:,:,c] = a.reshape((h,w))
        normals = normals + nor

    normals = normals/3
    normals = np.reshape(normals.T,(h,w,3))
    normalize(normals)
 
    return albedo,normals



