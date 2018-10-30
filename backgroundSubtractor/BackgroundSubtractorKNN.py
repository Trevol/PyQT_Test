import numpy as np
import cv2 as cv
from bg_substractor_utils import do_substraction

do_substraction("d:\DiskE\Computer_Vision_Task\Video 2.mp4", cv.createBackgroundSubtractorKNN())