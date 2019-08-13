from model import color_net
import numpy as np

from PIL import Image
from skimage import transform


def load(filename):
    np_image = Image.open(filename)
    np_image = np.array(np_image).astype('float32')/255
    np_image = transform.resize(np_image, (224, 224, 3))
    np_image = np.expand_dims(np_image, axis=0)
    return np_image


model = color_net(8)
model.load_weights('color_model.h5')

