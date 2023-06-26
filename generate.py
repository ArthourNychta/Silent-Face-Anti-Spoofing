from ast import dump
import json
import os
import cv2
import numpy as np
import argparse
import warnings
import time
from src.anti_spoof_predict import AntiSpoofPredict
from src.generate_patches import CropImage
from src.utility import parse_model_name
warnings.filterwarnings('ignore')

DEFAULT_IMAGE_PATH = "./images/sample/" #Change it to where the pictures are located

#Checking whether the image is 3:4 or not
def check_image(image):
    height, width, channel = image.shape
    if width/height != 3/4:
        print("Image is not appropriate!!!\nHeight/Width should be 4/3.")
        return False
    else:
        return True

#Predict image spoofing
def predict(image_name, model_dir, device_id):
    model_test = AntiSpoofPredict(device_id)
    image_cropper = CropImage()
    image = cv2.imread(DEFAULT_IMAGE_PATH + image_name)
    result = check_image(image)
    if result is False:
        return
    image_bbox = model_test.get_bbox(image)
    prediction = np.zeros((1, 3))
    test_speed = 0
    # sum the prediction from single model's result
    for model_name in os.listdir(model_dir):
        h_input, w_input, model_type, scale = parse_model_name(model_name)
        param = {
            "org_img": image,
            "bbox": image_bbox,
            "scale": scale,
            "out_w": w_input,
            "out_h": h_input,
            "crop": True,
        }
        if scale is None:
            param["crop"] = False
        img = image_cropper.crop(**param)
        start = time.time()
        prediction += model_test.predict(img, os.path.join(model_dir, model_name))
        test_speed += time.time()-start
    label = np.argmax(prediction)
    value = prediction[0][label]/2
    if label == 1:
        generate("Real", "{:.2f}".format(value), "{:.2f}".format(test_speed))
    else:
        generate("Fake", "{:.2f}".format(value), "{:.2f}".format(test_speed))

def generate(result, value, speed):
    image_name = args.image_name.split(".")[0]
    data = {
    "Result": result,
    "Value": value,
    "PredictionSpeed": speed,
    }
    with open("{}.json".format(image_name), "w") as write: json.dump(data, write)
    return print("Data generated as {}.json!".format(image_name))


if __name__ == "__main__":
    desc = "predict"
    parser = argparse.ArgumentParser(description=desc)
    parser.add_argument(
        "--device_id",
        type=int,
        default=0,
        help="which gpu id, [0/1/2/3]")
    parser.add_argument(
        "--model_dir",
        type=str,
        default="./resources/anti_spoof_models",
        help="model_lib used to predict")
    parser.add_argument(
        "--image_name",
        type=str,
        default="image_F1.jpg", #Can change the default image's name to anything
        help="image to predict")
    args = parser.parse_args()
    predict(args.image_name, args.model_dir, args.device_id)