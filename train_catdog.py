import os
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

IMG_SIZE = (160, 160)
BATCH = 32
DATA_DIR = "catdog_small_dataset/catdog_small"
WEIGHTS_PATH = os.path.expanduser("~/.keras/models/mobilenet_v2_weights_tf_dim_ordering_tf_kernels_1.0_224_no_top.h5")

# 1) Data load
train_ds = keras.utils.image_dataset_from_directory(
    f"{DATA_DIR}/train", image_size=IMG_SIZE, batch_size=BATCH, label_mode="binary"
)
val_ds = keras.utils.image_dataset_from_directory(
    f"{DATA_DIR}/val", image_size=IMG_SIZE, batch_size=BATCH, label_mode="binary"
)
class_names = train_ds.class_names
print("Classes:", class_names)

AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_ds.prefetch(AUTOTUNE)
val_ds = val_ds.prefetch(AUTOTUNE)

# 2) Data augmentation (small dataset -> augmentation helps a lot)
data_augmentation = keras.Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.1),
    layers.RandomZoom(0.1),
])

# 3) Pretrained base model (MobileNetV2, ImageNet weights) - transfer learning
base_model = keras.applications.MobileNetV2(
    input_shape=IMG_SIZE + (3,), include_top=False, weights=None
)
base_model.load_weights(WEIGHTS_PATH, by_name=False)
base_model.trainable = False  # freeze pretrained layers

# 4) Build full model
inputs = keras.Input(shape=IMG_SIZE + (3,))
x = data_augmentation(inputs)
x = keras.applications.mobilenet_v2.preprocess_input(x)
x = base_model(x, training=False)
x = layers.GlobalAveragePooling2D()(x)
x = layers.Dropout(0.2)(x)
outputs = layers.Dense(1, activation="sigmoid")(x)
model = keras.Model(inputs, outputs)

model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=1e-3),
    loss="binary_crossentropy",
    metrics=["accuracy"],
)
model.summary()

# 5) Train (only the new top layers - base is frozen)
EPOCHS = 8
history = model.fit(train_ds, validation_data=val_ds, epochs=EPOCHS)

# 6) Save model
model.save("catdog_model.keras")
print("\nFinal validation accuracy: {:.2%}".format(history.history["val_accuracy"][-1]))

# 7) Save history for plotting
import json
with open("history.json", "w") as f:
    json.dump(history.history, f)
