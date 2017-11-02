import numpy as np
import skimage.io as io
from skimage.transform import resize

WIDTH = 224
HEIGHT = 224
CHANNEL = 3

def predict(model, img_path):

    img = io.imread(img_path)
    img =resize(img, (WIDTH, HEIGHT, CHANNEL))
    io.imshow(img)
    io.show()
    img = np.expand_dims(img, 0)
    pred = model.predict(img, batch_size=1)
    pred = (pred*255.0).astype(np.uint8)
    pred = np.squeeze(pred)
    io.imshow(pred)
    io.show()
 
    return pred