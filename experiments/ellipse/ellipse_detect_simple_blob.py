import cv2, numpy as np

def drawEllipse(im):
    '''
    Parameters
        img	Image.
        center	Center of the ellipse.
        axes	Half of the size of the ellipse main axes.
        angle	Ellipse rotation angle in degrees.
        startAngle	Starting angle of the elliptic arc in degrees.
        endAngle	Ending angle of the elliptic arc in degrees.
        color	Ellipse color.
        thickness	Thickness of the ellipse arc outline, if positive. Otherwise, this indicates that a filled ellipse sector is to be drawn.
        lineType	Type of the ellipse boundary. See the line description.
        shift	Number of fractional bits in the coordinates of the center and values of axes.
    '''
    cv2.ellipse(im, (100, 100), (30, 30), 0, 0, 360, (0, 255, 0), -1, 8, 0)
    # ellipse( img, Point(dx+150, dy+100), Size(100,70), 0, 0, 360, white, -1, 8, 0 );


#im = cv2.imread('images/contacts/3contacts.jpg')
im = np.zeros((201, 201, 3), dtype=np.uint8)
drawEllipse(im)




params = cv2.SimpleBlobDetector_Params()
params.filterByArea = True;
params.minArea = 250;

detector = cv2.SimpleBlobDetector.create(params)

keypoints = detector.detect(im)
print(len(keypoints))
# im_with_keypoints = cv2.drawKeypoints(im, keypoints, np.array([]), (0, 0, 255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
im_with_keypoints = im.copy()
for keypoint in keypoints:
    cv2.circle(im_with_keypoints, (int(keypoint.pt[0]), int(keypoint.pt[1])), 4, (255, 0, 255), -1)


cv2.imshow("Keypoints", im_with_keypoints)

cv2.imshow('Im', im)
cv2.waitKey()
