import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

model = tf.keras.models.load_model("mnist.model")

img = cv.imread("test_data/Untitled.png")[:, :, 0]
img = np.invert(np.array([img]))
prediction = model.predict(img)
print(np.argmax(prediction))
plt.imshow(img[0], cmap = plt.cm.binary)