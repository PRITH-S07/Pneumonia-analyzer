## NOTE THAT THIS CODE IS IN CASE WE ARE NOT USING THE PRE-TRAINED MODEL

import cv2,pathlib
def resize_fn(fp: str) -> np.ndarray:
    """ Resize an image maintaining its proportions
    Args:
        fp (str): Path argument to image file
        scale (Union[float, int]): Percent as whole number of original image. eg. 53
    Returns:
        image (np.ndarray): Scaled image
    """    
    _scale = lambda dim, s: int(dim * s / 100)
    im: np.ndarray = cv2.imread(fp)
    #plt.imshow(im)
    width, height, channels = im.shape
    scale_1=(200*100)/width
    scale_2=(200*100)/height
    print("{},{},{}".format(width,height,channels))
    new_width: int = _scale(width, scale_1)
    new_height: int = _scale(height, scale_2)
    new_dim: tuple = (new_width, new_height)
    return cv2.resize(src=im, dsize=new_dim, interpolation=cv2.INTER_LINEAR)
  
img=cv2.imread('/content/drive/MyDrive/test (2)/NORMAL (1)/IM-0009-0001 (1).jpeg') # ANY IMAGE WE WANT
resized = resize_fn('/content/drive/MyDrive/test (2)/NORMAL (1)/IM-0009-0001 (1).jpeg') # RESIZING
print(resized.shape)
plt.imshow(resized)
image = resized.reshape((200,200,3))  # RESHAPING
X = np.zeros((1,200,200, 3), dtype=np.float32)
X[0]=image
model.predict(X)


