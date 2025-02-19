import os

path = "C:/Users/dahan/PycharmProjects/insect_classification/data/insects_dataset/butterfly/google0.jpg"
dirname = os.path.dirname(path)

for dirname in os.listdir("."):
    if os.path.isdir(dirname):
        for i, filename in enumerate(os.listdir(dirname)):
            os.rename(dirname + "/" + filename, dirname + "/" + str(i) + ".bmp")