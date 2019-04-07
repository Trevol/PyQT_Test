import numpy as np
import cv2
from cv_named_window import CvNamedWindow as Wnd, cvWaitKeys as waitKeys
from video_capture import VideoCapture
from video_controller import VideoController
import video_sources
import utils
import time

import keras
import tensorflow as tf
from keras_retinanet import models
from keras_retinanet.utils.image import read_image_bgr, preprocess_image, resize_image
from keras_retinanet.utils.visualization import draw_box, draw_caption


def main():
    def roiSelectedEvent(roi, imageRoi, wnd, image):
        if roi is None:
            return
        predictions = predict_on_image(model, None, imageRoi, 0.95)
        cv2.imshow('roi', predictions)

    model = getModel()
    video = VideoCapture(video_sources.video_2)
    vc = VideoController(10, 'pause')
    wnd = Wnd('video', roiSelectedEvent=roiSelectedEvent)
    for frame in video.frames():
        cv2.destroyWindow('roi')
        wnd.imshow(frame)
        if vc.wait_key() == 27: break


def getModel():
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    session = tf.Session(config=config)

    keras.backend.tensorflow_backend.set_session(session)
    model_path = './retinaNet/resnet50_pins_n_solder_14_inference.h5'
    return models.load_model(model_path, backbone_name='resnet50')


def predict_on_image(model, labels_to_names, image, thresh):
    # copy to draw on
    draw = image.copy()
    # draw = cv2.cvtColor(draw, cv2.COLOR_BGR2RGB, dst=draw)

    # preprocess image for network
    image = preprocess_image(image)
    # image, scale = resize_image(image)
    image, scale = image, 1

    # process image
    start = time.time()
    boxes, scores, labels = model.predict_on_batch(np.expand_dims(image, axis=0))
    print("processing time: ", time.time() - start)

    # correct for image scale
    boxes /= scale

    # visualize detections
    for box, score, label in zip(boxes[0], scores[0], labels[0]):
        # scores are sorted so we can break
        if score < thresh:
            break

        # color = label_color(label)
        if score < 0.99:
            color = (0, 0, 255)
        elif label == 1:
            color = (0, 255, 0)
        else:
            color = (0, 255, 255)
        b = np.round(box, 0).astype(int)
        draw_box(draw, b, color=color, thickness=1)

        # caption = f"{labels_to_names[label]} {score:.2f}"
        # draw_caption(draw, b, caption)

    # return cv2.cvtColor(draw, cv2.COLOR_RGB2BGR, dst=draw)
    return draw


if __name__ == '__main__':
    main()
