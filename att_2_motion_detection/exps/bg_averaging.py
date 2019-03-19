import cv2
from detection_attempts.att_2_motion_detection.background_subtractor_avg import BackgroundSubtractorAVG
from video_capture import VideoCapture
import utils
import video_sources

def denoise(frame):
    # frame = cv2.medianBlur(frame, 5, dst=frame)
    frame = cv2.GaussianBlur(frame, (3, 3), 0, dst=frame)

    return frame


def main():
    video = VideoCapture(video_sources.video_2)

    frame = video.read()
    backSubtractor = BackgroundSubtractorAVG(0.2, denoise(frame))

    for frame in video.frames():
        with utils.timeit_context():
            frame = denoise(frame)
            foreGround = backSubtractor.getForeground(frame)
            # Apply thresholding on the background and display the resulting mask
            ret, mask = cv2.threshold(foreGround, 15, 255, cv2.THRESH_BINARY)

        cv2.imshow('input', frame)
        cv2.imshow('foreground', foreGround)
        # Note: The mask is displayed as a RGB image, you can
        # display a grayscale image by converting 'foreGround' to
        # a grayscale before applying the threshold.
        cv2.imshow('mask', mask)

        if cv2.waitKey(10) & 0xFF == 27:
            break

    video.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
