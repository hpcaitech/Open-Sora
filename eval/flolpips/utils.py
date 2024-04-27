import numpy as np
import cv2
import torch


def normalize_tensor(in_feat,eps=1e-10):
    norm_factor = torch.sqrt(torch.sum(in_feat**2,dim=1,keepdim=True))
    return in_feat/(norm_factor+eps)

def l2(p0, p1, range=255.):
    return .5*np.mean((p0 / range - p1 / range)**2)

def dssim(p0, p1, range=255.):
    from skimage.measure import compare_ssim
    return (1 - compare_ssim(p0, p1, data_range=range, multichannel=True)) / 2.

def tensor2im(image_tensor, imtype=np.uint8, cent=1., factor=255./2.):
    image_numpy = image_tensor[0].cpu().float().numpy()
    image_numpy = (np.transpose(image_numpy, (1, 2, 0)) + cent) * factor
    return image_numpy.astype(imtype)

def tensor2np(tensor_obj):
    # change dimension of a tensor object into a numpy array
    return tensor_obj[0].cpu().float().numpy().transpose((1,2,0))

def np2tensor(np_obj):
     # change dimenion of np array into tensor array
    return torch.Tensor(np_obj[:, :, :, np.newaxis].transpose((3, 2, 0, 1)))

def tensor2tensorlab(image_tensor,to_norm=True,mc_only=False):
    # image tensor to lab tensor
    from skimage import color

    img = tensor2im(image_tensor)
    img_lab = color.rgb2lab(img)
    if(mc_only):
        img_lab[:,:,0] = img_lab[:,:,0]-50
    if(to_norm and not mc_only):
        img_lab[:,:,0] = img_lab[:,:,0]-50
        img_lab = img_lab/100.

    return np2tensor(img_lab)

def read_frame_yuv2rgb(stream, width, height, iFrame, bit_depth, pix_fmt='420'):
    if pix_fmt == '420':
        multiplier = 1
        uv_factor = 2
    elif pix_fmt == '444':
        multiplier = 2
        uv_factor = 1
    else:
        print('Pixel format {} is not supported'.format(pix_fmt))
        return

    if bit_depth == 8:
        datatype = np.uint8
        stream.seek(iFrame*1.5*width*height*multiplier)
        Y = np.fromfile(stream, dtype=datatype, count=width*height).reshape((height, width))
        
        # read chroma samples and upsample since original is 4:2:0 sampling
        U = np.fromfile(stream, dtype=datatype, count=(width//uv_factor)*(height//uv_factor)).\
                                reshape((height//uv_factor, width//uv_factor))
        V = np.fromfile(stream, dtype=datatype, count=(width//uv_factor)*(height//uv_factor)).\
                                reshape((height//uv_factor, width//uv_factor))

    else:
        datatype = np.uint16
        stream.seek(iFrame*3*width*height*multiplier)
        Y = np.fromfile(stream, dtype=datatype, count=width*height).reshape((height, width))
                
        U = np.fromfile(stream, dtype=datatype, count=(width//uv_factor)*(height//uv_factor)).\
                                reshape((height//uv_factor, width//uv_factor))
        V = np.fromfile(stream, dtype=datatype, count=(width//uv_factor)*(height//uv_factor)).\
                                reshape((height//uv_factor, width//uv_factor))

    if pix_fmt == '420':
        yuv = np.empty((height*3//2, width), dtype=datatype)
        yuv[0:height,:] = Y

        yuv[height:height+height//4,:] = U.reshape(-1, width)
        yuv[height+height//4:,:] = V.reshape(-1, width)

        if bit_depth != 8:
            yuv = (yuv/(2**bit_depth-1)*255).astype(np.uint8)

        #convert to rgb
        rgb = cv2.cvtColor(yuv, cv2.COLOR_YUV2RGB_I420)
    
    else:
        yvu = np.stack([Y,V,U],axis=2)
        if bit_depth != 8:
            yvu = (yvu/(2**bit_depth-1)*255).astype(np.uint8)
        rgb = cv2.cvtColor(yvu, cv2.COLOR_YCrCb2RGB)

    return rgb
