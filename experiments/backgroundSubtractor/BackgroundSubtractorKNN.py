import numpy as np
import cv2
import video_sources
from experiments.backgroundSubtractor.bg_substractor_utils import do_substraction

do_substraction(video_sources.video_2, cv2.createBackgroundSubtractorKNN(detectShadows=False))