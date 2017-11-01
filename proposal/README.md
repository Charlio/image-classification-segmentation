## Data
    - Train data is downloaded here: http://host.robots.ox.ac.uk/pascal/VOC/voc2012/#devkit
    - I split the data into train and validation
    - For test purpose, I will use any image containing common object, generate and draw the mask image. One can tell how well the model is visually from the pictures.
    - The original segmentation images in the folder SegmentationClasses are more complex than what we need in the project, so I will transfer them into the form I will use by the following function:
def seg_to_mask(img):
    mask = np.ndarray((224, 224))
    for i in range(len(img)):
        for j in range(len(img[0])):
            if img[i][j].any() != 0:
                mask[i][j] = 1
            else:
                mask[i][j] = 0
    return mask
    - The above function transfer an image of size (224, 224, 3) into size (224, 224). I simply change a non-black pixel value into 1 by the function.
