import argparse
import tensorflow as tf
import json
import numpy as np
import tensorflow_hub as hub
from PIL import Image

# helping methods (taken from previous notebook):
def process_image(image):
    new_image = tf.convert_to_tensor(image, dtype = tf.float32)
    new_image = tf.image.resize(new_image, (224, 224))
    new_image /= 255
    return new_image.numpy()

# method to get the flowers name instead of label number.
def getNames(classes):
    classes_names = []
    for i in classes:
        classes_names.append(class_names[str(i)])
    return classes_names

def labelPlus1(label):
    return label + 1

def predict(image_path, model, top_k):
    # open image.
    im = Image.open(image_path)
    # process the image.
    procesed_image = process_image(np.asarray(im))
    # expand the image shape from (224, 224, 3) to (1, 224, 224, 3).
    expanded_image = np.expand_dims(procesed_image, axis=0)
    # make a predict.
    imagePredicts = model.predict(expanded_image)
    imagePredicts = imagePredicts.tolist()

    probs, classes = tf.math.top_k(imagePredicts, k=top_k)

    classes = tf.map_fn(labelPlus1, classes)

    probs = probs.numpy().tolist()[0]
    classes = classes.numpy().tolist()[0]
    return probs, classes

# print the class name with its probabilities.
def print_top_classes(prob, classes_with_nam, top_k):
    for i in range(top_k):
        print(i+1)
        print(classes_with_name[i])
        print(prob[i])
        print('-------------------------')


parser = argparse.ArgumentParser(description='Predict the type of the flower in the photo.')

parser.add_argument('--input', action='store', dest='inputPath', default='./test_images/orange_dahlia.jpg')
parser.add_argument('--model', action='store', dest='modelPath', default='savedModel.h5')
parser.add_argument('--top_k', type=int, action='store', dest='top_k', default=5)
parser.add_argument('--category_names', action='store', dest="categoryPath", default='./label_map.json')

args = parser.parse_args()

if __name__ == '__main__':

    model = tf.keras.models.load_model(args.modelPath)

    with open(args.categoryPath, 'r') as f:
        class_names = json.load(f)

    prob, classes = predict(args.inputPath, model, args.top_k)
    classes_with_name = getNames(classes)

    print_top_classes(prob, classes_with_name, args.top_k)

'''
sources: 
https://docs.python.org/3/howto/argparse.html
https://www.youtube.com/watch?v=cdblJqEUDNo
'''

