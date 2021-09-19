# I SUGGEST TO USE THE WEIGHTS TITLED 'Pneumonia_classification.h5' as they are the ones for which we got the best validation/ testing accuracy.
from keras.models import load_model
model_n = load_model('/content/Pneumonia_classification.h5')
img=cv2.imread('/content/drive/MyDrive/test (2)/NORMAL (1)/IM-0009-0001 (1).jpeg')
print(img.shape)
resized = resize_fn('/content/drive/MyDrive/test (2)/NORMAL (1)/IM-0009-0001 (1).jpeg')
print(resized.shape)
plt.imshow(resized)
image = resized.reshape((200,200,3))
X = np.zeros((1,200,200, 3), dtype=np.float32)
X[0]=image
model_n.predict(X)
