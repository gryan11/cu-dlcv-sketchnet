import numpy as np
import matplotlib.pyplot as plt
from subprocess import call
import glob, re, random

def show_sample(X, y, prediction=None):
    im = X
    cmap = None
    if (im.shape[2] == 1):
        im = np.squeeze(im, (2,))
        cmap = 'gray'
    plt.imshow(im, interpolation='none', cmap = cmap)
    if prediction != None:
        plt.title("Class = %s, Predict = %s" % (str(y), str(prediction)))
    else:
        plt.title("Class = %s" % (str(y)))

    plt.axis('on')
    plt.show()
    
def plots(ims, figsize=(12,6), rows=1, interp=False, titles=None):
    cmap = None
    if type(ims[0]) is np.ndarray:
        ims = np.array(ims).astype(np.uint8)
        if (ims.shape[-1] == 1):
            ims = np.squeeze(ims, (3,))
            cmap = 'gray'
    f = plt.figure(figsize=figsize)
    for i in range(len(ims)):
        sp = f.add_subplot(rows, len(ims)//rows, i+1)
        sp.axis('Off')
        if titles is not None:
            sp.set_title(titles[i], fontsize=16)
        plt.imshow(ims[i], interpolation=None if interp else 'none', cmap=cmap)
        
def take_sample(orig_dir, sample_dir, num_classes):
    # make sample dirs
    call(['rm', '-r', sample_dir])
    
    call(['mkdir', sample_dir])
    call(['mkdir', sample_dir+'/train'])
    call(['mkdir', sample_dir+'/test'])
    
    
    # get list of classes and pick ones to sample
    train = sorted(glob.glob(orig_dir+'/train/*'))
    test = sorted(glob.glob(orig_dir+'/test/*'))
    
    def getclass(c):
        return re.search(r"\/([\w\-_]+)$", c).group(1)
    
    classes = sorted(list(map(getclass, train)))
    
    random.seed(0)
    sample_classes = random.sample(classes, num_classes)
    
    # copy to sample dirs
    for sample_class in sample_classes:        
        train_src = next(c for c in train if sample_class in c)
        test_src = next(c for c in test if sample_class in c)
        
        call(['cp', '-r', train_src, sample_dir+'/train/'])
        call(['cp', '-r', test_src, sample_dir+'/test/'])
