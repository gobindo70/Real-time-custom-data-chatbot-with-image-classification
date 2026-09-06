import sys
import tensorflow as tf
from tensorflow import keras
import numpy as np

IMG_SIZE = (160, 160)
CLASS_NAMES = ['cats', 'dogs']

model = keras.models.load_model("catdog_model.keras")

def predict(img_path):
    img = keras.utils.load_img(img_path, target_size=IMG_SIZE)
    arr = keras.utils.img_to_array(img)
    arr = np.expand_dims(arr, axis=0)
    pred = model.predict(arr, verbose=0)[0][0]
    label = CLASS_NAMES[1] if pred > 0.5 else CLASS_NAMES[0]
    confidence = pred if pred > 0.5 else 1 - pred
    print(f"{img_path}: {label} ({confidence:.1%} confident)")

if __name__ == "__main__":
    for path in sys.argv[1:]:
        predict(path)
