import numpy as np
from PIL import Image
import math

def circshift(c_A, c_shift):
    for i in range(c_shift.size):
        c_A = np.roll(c_A, (c_shift[i]).astype(int), axis=i)
    return c_A
    
def psf_prep(p_psf, p_outshape = None, p_dtype = None):
    if not p_dtype:
        p_dtype = np.float32
    p_psf = np.float32(p_psf)
    p_size =np.int32(p_psf.shape)
    if not p_outshape:
        p_outshape = p_size
    p_n_psf = np.zeros(p_outshape, dtype=p_dtype)
    p_n_psf[:p_size[0], :p_size[1]] = p_psf[:,:]
    p_psf = p_n_psf
   #pdb.set_trace()
    p_shift = -(p_size/2)
    p_psf = circshift(p_psf,p_shift)
    return p_psf
def psf2otf(po_psf, po_outshape=None):
    po_dat = psf_prep(po_psf, po_outshape)
    po_otf = np.fft.fftn(po_dat)
    return np.complex64(po_otf)

def l0_smooth(l_image, l_kappa=None, l_lambda=None):
    l_image = l_image.astype("float32")
    l_image/=255.
    l_N,l_M,l_D = l_image.shape
    l_kappa = l_kappa if l_kappa is not None else 2
    l_lambda = l_lambda if l_lambda is not None else .02
    
    l_outshape = [l_N,l_M]
    l_colorgradx = np.int32([[1, -1]])
    l_colorgrady = np.int32([[1],[ -1]])
    l_otf_x = psf2otf(l_colorgradx,l_outshape)
    l_otf_y = psf2otf(l_colorgrady,l_outshape)

    l_FI = np.complex64(np.zeros((l_N,l_M,l_D)))
    l_FI[:,:,0] = np.fft.fft2(l_image[:,:,0])
    l_FI[:,:,1] = np.fft.fft2(l_image[:,:,1])
    l_FI[:,:,2] = np.fft.fft2(l_image[:,:,2])

    l_MTF = np.power(np.abs(l_otf_x), 2) + np.power(np.abs(l_otf_y), 2)
    l_MTF = np.tile(l_MTF[:, :, np.newaxis], (1, 1, l_D))

    l_h = np.float32(np.zeros((l_N, l_M, l_D)))
    l_v = np.float32(np.zeros((l_N, l_M, l_D)))
    l_dxhp = np.float32(np.zeros((l_N, l_M, l_D)))
    l_dyvp = np.float32(np.zeros((l_N, l_M, l_D)))
    l_FS = np.complex64(np.zeros((l_N, l_M, l_D)))

    # Iteration settings
    l_beta_max = 1e5;
    l_beta = 2 * l_lambda
    l_iteration = 0
    
    while l_beta < l_beta_max:
        # compute dxSp
        l_h[:,0:l_M-1,:] = np.diff(l_image, 1, 1)
        l_h[:,l_M-1:l_M,:] = l_image[:,0:1,:] - l_image[:,l_M-1:l_M,:]

        # compute dySp
        l_v[0:l_N-1,:,:] = np.diff(l_image, 1, 0)
        l_v[l_N-1:l_N,:,:] = l_image[0:1,:,:] - l_image[l_N-1:l_N,:,:]

        # compute minimum energy E = dxSp^2 + dySp^2 <= _lambda/beta
        l_t = np.sum(np.power(l_h, 2) + np.power(l_v, 2), axis=2) < l_lambda / l_beta
        l_t = np.tile(l_t[:, :, np.newaxis], (1, 1, 3))
        l_h[l_t] = 0
        l_v[l_t] = 0

         # compute dxhp + dyvp
        l_dxhp[:,0:1,:] = l_h[:,l_M-1:l_M,:] - l_h[:,0:1,:]
        l_dxhp[:,1:l_M,:] = -(np.diff(l_h, 1, 1))
        l_dyvp[0:1,:,:] = l_v[l_N-1:l_N,:,:] - l_v[0:1,:,:]
        l_dyvp[1:l_N,:,:] = -(np.diff(l_v, 1, 0))
        l_normin = l_dxhp + l_dyvp

        l_FS[:,:,0] = np.fft.fft2(l_normin[:,:,0])
        l_FS[:,:,1] = np.fft.fft2(l_normin[:,:,1])
        l_FS[:,:,2] = np.fft.fft2(l_normin[:,:,2])

        # solve for S + 1 in Fourier domain
        l_denorm = 1 + l_beta * l_MTF;
        l_FS[:,:,:] = (l_FI + l_beta * l_FS) / l_denorm

        # inverse FFT to compute S + 1
        l_image[:,:,0] = np.float32((np.fft.ifft2(l_FS[:,:,0])).real)
        l_image[:,:,1] = np.float32((np.fft.ifft2(l_FS[:,:,1])).real)
        l_image[:,:,2] = np.float32((np.fft.ifft2(l_FS[:,:,2])).real)

        l_beta *= l_kappa
        l_iteration +=1
    
    return (l_image*255).astype("uint8")