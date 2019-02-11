from detection_attempts.att_1.detect_on_video import get_capture_and_calibration_image_video2
import cv2
import time


def main():
    video, _ = get_capture_and_calibration_image_video2()

    fourcc = cv2.VideoWriter_fourcc(*'XVID')  # fourcc = cv2.VideoWriter_fourcc('M', 'J', 'P', 'G')
    out = cv2.VideoWriter('d:/DiskE/Computer_Vision_Task/Video_2_out_xvid.avi', fourcc, video.fps(), video.resolution())

    t0 = time.time()
    for frame in video.frames():
        if video.frame_pos() > 1000:
            break
        out.write(frame)
        # time.sleep(1)
    print('Done!', time.time() - t0)

    video.release()
    out.release()


main()
