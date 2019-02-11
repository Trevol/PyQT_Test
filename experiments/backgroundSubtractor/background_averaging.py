import numpy as np
import cv2


class BackGroundSubtractor:
    # When constructing background subtractor, we
    # take in two arguments:
    # 1) alpha: The background learning factor, its value should
    # be between 0 and 1. The higher the value, the more quickly
    # your program learns the changes in the background. Therefore,
    # for a static background use a lower value, like 0.001. But if
    # your background has moving trees and stuff, use a higher value,
    # maybe start with 0.01.
    # 2) firstFrame: This is the first frame from the video/webcam.
    def __init__(self, alpha, firstFrame):
        self.alpha = alpha
        self.backGroundModel = firstFrame

    def getForeground(self, frame):
        # apply the background averaging formula:
        # NEW_BACKGROUND = CURRENT_FRAME * ALPHA + OLD_BACKGROUND * (1 - APLHA)
        self.backGroundModel = frame * self.alpha + self.backGroundModel * (1 - self.alpha)

        # after the previous operation, the dtype of
        # self.backGroundModel will be changed to a float type
        # therefore we do not pass it to cv2.absdiff directly,
        # instead we acquire a copy of it in the uint8 dtype
        # and pass that to absdiff.

        return cv2.absdiff(self.backGroundModel.astype(np.uint8), frame)


# Just a simple function to perform
# some filtering before any further processing.
def denoise(frame):
    frame = cv2.medianBlur(frame, 5)
    frame = cv2.GaussianBlur(frame, (5, 5), 0)
    return frame


def main():
    # cam = cv2.VideoCapture(0)
    cam = cv2.VideoCapture("d:\DiskE\Computer_Vision_Task\Video_2.mp4")
    ret, frame = cam.read()
    if not ret:
        return
    backSubtractor = BackGroundSubtractor(0.1, denoise(frame))

    while (True):
        # Read a frame from the camera
        ret, frame = cam.read()
        if not ret:
            break

        # Show the filtered image
        denoised_frame = denoise(frame)
        cv2.imshow('input', frame)
        cv2.imshow('input denoised', denoised_frame)

        # get the foreground
        foreGround = backSubtractor.getForeground(denoised_frame)

        # Apply thresholding on the background and display the resulting mask
        ret, mask = cv2.threshold(foreGround, 15, 255, cv2.THRESH_BINARY)

        # Note: The mask is displayed as a RGB image, you can
        # display a grayscale image by converting 'foreGround' to
        # a grayscale before applying the threshold.
        cv2.imshow('mask', mask)

        if cv2.waitKey(10) & 0xFF == 27:
            break

    cam.release()
    cv2.destroyAllWindows()


main()
