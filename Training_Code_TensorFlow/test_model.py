import tensorflow as tf

# -----------------------
# CONFIG
# -----------------------
IMG_SIZE = (32, 32)
BATCH_SIZE = 16
DATA_PATH = "Training_Code_TensorFlow/Data_Splitted"

# -----------------------
# LOAD TEST DATA
# -----------------------
test_ds = tf.keras.utils.image_dataset_from_directory(
    f"{DATA_PATH}/test",
    image_size=IMG_SIZE,
    color_mode="grayscale",
    batch_size=BATCH_SIZE,
    label_mode="int",
    shuffle=False
)

# Normalize (same as training!)
rescale = tf.keras.layers.Rescaling(1./255)
test_ds = test_ds.map(lambda x, y: (rescale(x), y))

AUTOTUNE = tf.data.AUTOTUNE
test_ds = test_ds.prefetch(AUTOTUNE)

# -----------------------
# LOAD MODEL
# -----------------------
model = tf.keras.models.load_model("Training_Code_TensorFlow/best_model.keras")

# -----------------------
# EVALUATE
# -----------------------
loss, acc = model.evaluate(test_ds)

print("\n======================")
print("TEST LOSS:", loss)
print("TEST ACCURACY:", acc)
print("======================")