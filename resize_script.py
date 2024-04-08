import os
import cv2


def resizeAllImagesInDir(pathToDir):
    for filename in os.listdir(pathToDir):
        abs_filename = os.path.join(pathToDir, filename)
        if os.path.isfile(abs_filename):
            # Process the file here
            img = cv2.imread(abs_filename)
            if img is not None:
                img = cv2.resize(img, (224, 224))
                cv2.imwrite(abs_filename, img)
            else:
                os.remove(abs_filename)


resizeAllImagesInDir("./FastFood/train/Burger")
# resizeAllImagesInDir("./FastFood/train/HotDog")
# resizeAllImagesInDir("./FastFood/val/Burger")
# resizeAllImagesInDir("./FastFood/val/HotDog")

# resizeAllImagesInDir("./test_fastfood")
