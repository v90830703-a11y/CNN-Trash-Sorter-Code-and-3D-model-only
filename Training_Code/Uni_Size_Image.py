#a image processing function that resizes an image to QVGA 
#while maintaining the aspect ratio and adding padding if necessary 
import cv2

QVGA_WIDTH = 320
QVGA_HEIGHT = 240
QVGA_RATIO = QVGA_WIDTH / QVGA_HEIGHT


def resize_pad_and_replace_image(file_path):
    pad_height, pad_width = 0, 0

    image = cv2.imread(file_path)
    height, width, channels = image.shape
    # print(f"Width: {width} px")
    # print(f"Height: {height} px\n converted to \nQVGA: {QVGA_WIDTH} x {QVGA_HEIGHT} px")


    if width / height > QVGA_RATIO:
        new_width = QVGA_WIDTH
        new_height = int(QVGA_WIDTH / width * height)
        pad_height = int((QVGA_HEIGHT - new_height) / 2)

    elif width / height < QVGA_RATIO:
        new_width = int(QVGA_HEIGHT / height * width)
        new_height = QVGA_HEIGHT
        pad_width = int ((QVGA_WIDTH - new_width) / 2)

    else:
        new_width = QVGA_WIDTH
        new_height = QVGA_HEIGHT

    resized = cv2.resize(image, (new_width, new_height))
    padded = cv2.copyMakeBorder(resized, int(pad_height), int(pad_height),
                                int(pad_width), int(pad_width),
                                cv2.BORDER_CONSTANT, value=[0, 0, 0])

    cv2.imwrite(file_path, padded)
