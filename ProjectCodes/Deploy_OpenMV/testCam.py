import sensor, image, time, tf
import json

sensor.reset()
sensor.set_pixformat(sensor.GRAYSCALE)
sensor.set_framesize(sensor.QQVGA)
sensor.set_windowing((32, 32))
sensor.skip_frames(time=2000)

clock = time.clock()

# ✅ FULL REQUIRED CONSTRUCTOR
net = tf.Model(
    "model_int8.tflite",
    load_to_fb=True,
    postprocess="softmax"
)
print(net)
help(net.predict)

with open("class_names.json") as f:
    labels = json.load(f)

while True:
    clock.tick()

    img = sensor.snapshot()
    # print([img])
    # print(type(img))
    dummy = [0] * 1024
    print(len(dummy))

    print(net.predict([dummy]))
    # print(dir(net))
    # print(net.input_shape())
    # print(net.predict((img,)))

    if result:
        scores = result[0]

        idx = scores.index(max(scores))

        print("Detected:", labels[idx], "| FPS:", clock.fps())
