import numpy as np
import cv2
from bg_substractor_utils import do_substraction
import video_sources

do_substraction(video_sources.video_2, cv2.createBackgroundSubtractorKNN(detectShadows=False))