from __future__ import print_function
import numpy as np
import glob
import os.path
import skimage.io as io
from skimage.transform import resize

DATA_PATH = "VOCdevkit/VOC2012/"
SEGS_DIR = DATA_PATH + "SegmentationClass/"
IMGS_DIR = DATA_PATH + "JPEGImages/"

WIDTH = 224
HEIGHT = 224
CHANNEL = 3

def seg_to_img(seg, imgs_dir):
    dirname, basename = os.path.split(seg)
    imgname = basename.replace(".png", ".jpg")
    return os.path.join(imgs_dir, imgname)

def seg_to_mask(seg):
    mask = np.ndarray((HEIGHT, WIDTH, 1))
    for i in range(len(seg)):
        for j in range(len(seg[0])):
            if seg[i][j].any() != 0:
                mask[i][j][0] = 1
            else:
                mask[i][j][0] = 0
    return mask

def generate_data():
    seg_names = [seg for seg in glob.glob(SEGS_DIR + "*")]
    img_names = [seg_to_img(seg, IMGS_DIR) for seg in seg_names]
    
    num_of_imgs = len(seg_names)
    
    print('-'*30)
    print('There are ', num_of_imgs, ' pairs of images and masks to load.')
    
    imgs = np.ndarray((num_of_imgs, HEIGHT, WIDTH, CHANNEL), dtype=np.uint8)
    masks = np.ndarray((num_of_imgs, HEIGHT, WIDTH, 1), dtype=np.uint8)
    
    print('-'*30)
    print('Creating training images and masks...')
    print('-'*30)
    
    for i in range(num_of_imgs):
        seg =resize(io.imread(seg_names[i]), (HEIGHT, WIDTH, CHANNEL), preserve_range=True)
        masks[i] = seg_to_mask(seg)
        img = resize(io.imread(img_names[i]), (HEIGHT, WIDTH, CHANNEL), preserve_range=True)
        imgs[i] = img
        if i % 200 == 0:
            print(i, "pairs of images and masks are loaded.")
        
    print('Finished loading all ', num_of_imgs, " pairs of images and masks.")

    imgs = imgs.astype('float32')
    mean = np.mean(imgs)
    std = np.std(imgs)
    imgs -= mean
    imgs /= std
    
    np.save(DATA_PATH + "imgs.npy", imgs)
    np.save(DATA_PATH + "masks.npy", masks)
    
    print('Saved images and masks to .npy files')
    

def load_data():
    imgs = np.load(DATA_PATH + 'imgs.npy')
    masks = np.load(DATA_PATH + 'masks.npy')
    return imgs, masks
    
    
if __name__ == '__main__':
    generate_data()