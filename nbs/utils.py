import numpy as np
import matplotlib.pyplot as plt
from subprocess import call
import glob, re, random

from keras.preprocessing.image import ImageDataGenerator
from keras import backend as K
from keras.callbacks import TensorBoard, ModelCheckpoint

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
        
class ModelTrainer():
    
    def __init__(self,
                 model,
                 model_name,
                 opt,
                 train_generator,
                 validation_generator,
                 train_steps,
                 val_steps):
        self.total_epochs = 0
        self.opt = opt
        self.model = model
        self.train_generator = train_generator
        self.validation_generator = validation_generator
        self.train_steps = train_steps
        self.val_steps = val_steps
        
        call(['mkdir', '-p', '../logs/'+model_name])
        call(['mkdir', '-p', '../models/'+model_name])

        with open('../models/'+model_name+'/model.json', 'w') as f:
            f.write(model.to_json())

        self.log_cb =\
            TensorBoard(log_dir='../logs/'+model_name+'/', 
                        histogram_freq=0, 
                        write_graph=False, write_images=False)
        self.best_model_cb =\
            ModelCheckpoint('../models/'+model_name+'/best.h5', 
                            monitor='val_acc', verbose=0, 
                            save_best_only=True, 
                            mode='auto', period=1)
        self.latest_model_cb =\
            ModelCheckpoint('../models/'+model_name+'/latest.h5', 
                            monitor='val_acc', verbose=0, 
                            period=1)
            
    def train(self, epochs, lr):
        K.set_value(self.opt.lr, lr)

        hist = self.model.fit_generator(
                self.train_generator,
                steps_per_epoch = self.train_steps,
                epochs = epochs + self.total_epochs,
                validation_data = self.validation_generator,
                validation_steps = self.val_steps,
                verbose = 1,
                initial_epoch = self.total_epochs,
                callbacks=[self.log_cb, 
                           self.best_model_cb,
                           self.latest_model_cb]
        )
        self.total_epochs += epochs
        
        
        
        
        
        
        
