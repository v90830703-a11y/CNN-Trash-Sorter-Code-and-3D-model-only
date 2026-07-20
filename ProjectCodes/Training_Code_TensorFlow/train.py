import tensorflow as tf
import json
from model import build_model

# ---------------- CONFIG ----------------
IMG_SIZE = (32, 32)
BATCH_SIZE = 16
EPOCHS = 20
DATA_PATH = "Training_Code_TensorFlow/Data_Splitted"
NUM_CLASSES = 4

# ---------------- DATA ----------------
train_ds = tf.keras.utils.image_dataset_from_directory(
    f"{DATA_PATH}/train",
    image_size=IMG_SIZE,
    color_mode="grayscale",
    batch_size=BATCH_SIZE,
    label_mode="int"
)

val_ds = tf.keras.utils.image_dataset_from_directory(
    f"{DATA_PATH}/val",
    image_size=IMG_SIZE,
    color_mode="grayscale",
    batch_size=BATCH_SIZE,
    label_mode="int"
)

test_ds = tf.keras.utils.image_dataset_from_directory(
    f"{DATA_PATH}/test",
    image_size=IMG_SIZE,
    color_mode="grayscale",
    batch_size=BATCH_SIZE,
    label_mode="int",
    shuffle=False
)

class_names = train_ds.class_names
print("Classes:", class_names)

# ---------------- NORMALIZATION ----------------
rescale = tf.keras.layers.Rescaling(1./255)

train_ds = train_ds.map(lambda x, y: (rescale(x), y))
val_ds = val_ds.map(lambda x, y: (rescale(x), y))
test_ds = test_ds.map(lambda x, y: (rescale(x), y))

AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_ds.cache().shuffle(1000).prefetch(AUTOTUNE)
val_ds = val_ds.cache().prefetch(AUTOTUNE)
test_ds = test_ds.cache().prefetch(AUTOTUNE)

# ---------------- MODEL ----------------
#choose one of these two options:

#firt time training
#model = build_model(input_shape=(32, 32, 1), num_classes=NUM_CLASSES)

#subsequent training
model = tf.keras.models.load_model("Training_Code_TensorFlow/best_model.keras")

#change learning rate for subsequent training
model.compile(
    optimizer=tf.keras.optimizers.Adam(0.00001),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

model.summary()

# ---------------- TRAIN ----------------
checkpoint = tf.keras.callbacks.ModelCheckpoint(
    "Training_Code_TensorFlow/best_model.keras",
    save_best_only=True,
    monitor="val_accuracy",
    mode="max"
)

history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS,
    callbacks=[checkpoint]
)

# ---------------- TEST ----------------
test_loss, test_acc = model.evaluate(test_ds)
print("FINAL TEST ACCURACY:", test_acc)

# ---------------- SAVE MODEL ----------------
model.save("Training_Code_TensorFlow/saved_model")

# save class names
with open("Training_Code_TensorFlow/class_names.json", "w") as f:
    json.dump(class_names, f)

print("Saved model + classes")