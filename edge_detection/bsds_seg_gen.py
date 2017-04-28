import numpy as np
from scipy.signal import convolve2d as conv2d
from PIL import Image
from scipy.io import loadmat

#Generates segments for BSDS mat files. pass in a mat file loaded into a variable

def seg2edge(s_seg):
    s_conv = np.array([[1.,1.,1.],[1.,1.,1.],[1.,1.,1.]])
    s_res  = conv2d(np.pad(s_seg,1,"edge"),s_conv,mode="same")
    s_resShape = s_res.shape
    s_res = s_res[1:s_res.shape[0]-1,1:s_res.shape[1]-1]
    s_res = np.abs(s_res-9.*s_seg)
    s_res = (1.-np.ceil(s_res/np.max(s_res)))
    return s_res

#pass in a mat file and this will load it and stuff
#returns PIL.Image formatted variable
def seg_generator(f_file, f_sparse):
    f_trF = loadmat(f_file)
    f_res,edgeSt = [],[]
    f_res.append((seg2edge(f_trF['groundTruth'][0,0][0,0][0].astype("float"))))
    edgeSt.append(np.sum(1.-f_res[-1]))
    for i in range(1,f_trF['groundTruth'].shape[1]):
        f_res.append(seg2edge(f_trF['groundTruth'][0,i][0,0][0].astype("float")))
        edgeSt.append(np.sum(1.-f_res[-1]))
    sortedEdgeSet = [i[0] for i in sorted(enumerate(edgeSt), key=lambda x:x[1])]
    if  len(edgeSt) > 1:
        f_res = f_res[sortedEdgeSet[0]]+f_res[sortedEdgeSet[1]] if f_sparse else f_res[sortedEdgeSet[len(edgeSt)-1]]+f_res[sortedEdgeSet[len(edgeSt)-2]]
    else: 
        f_res = f_res[0]
    return (255*(1-np.ceil((1-(f_res/np.max(f_res))))).astype("uint8"))