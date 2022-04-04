import cv2
import matplotlib.pyplot as plt

# Split Video into Frames
def get_frames(input_path):
    cap = cv2.VideoCapture(input_path)
    i = 0
    # a variable to set how many frames you want to skip
    frame_skip = 5
    # a variable to keep track of the frame to be saved
    frame_count = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        if i > frame_skip - 1:
            frame_count += 1
            cv2.imwrite('./images/' + str(frame_count * frame_skip) + '.jpg', frame)
            i = 0
            continue
        i += 1

    cap.release()
    cv2.destroyAllWindows()


input_path = './videos/video.mp4'
# get_frames(input_path)

# Reading the images and converting into B/W, if they aren't greyscale already
# image = cv2.imread('pathtoimage')
# gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

image1 = cv2.imread('./images/5.jpg')
image2 = cv2.imread('./images/10.jpg')

orb = cv2.ORB_create(nfeatures=2000)
kp1, des1 = orb.detectAndCompute(image1, None)
kp2, des2 = orb.detectAndCompute(image2, None)
# Use cv2.drawKeypoints to visualize keypoints.


# Nearest neighbour matching
FLANN_INDEX_KDTREE = 1
FLANN_INDEX_LSH = 6
index_params = dict(algorithm=FLANN_INDEX_LSH,
                    table_number=6,  # 12
                    key_size=12,  # 20
                    multi_probe_level=1)
search_params = dict(checks=50)  # or pass empty dictionary
flann = cv2.FlannBasedMatcher(index_params, search_params)
matches = flann.knnMatch(des1, des2, k=2)

# Need to draw only good matches, so create a mask
matchesMask = [[0, 0] for i in range(len(matches))]

# ratio test as per Lowe's paper
for i, (m, n) in enumerate(matches):
    if m.distance < 0.7 * n.distance:
        matchesMask[i] = [1, 0]
draw_params = dict(matchColor=(0, 255, 0),
                   singlePointColor=(255, 0, 0),
                   matchesMask=matchesMask,
                   flags=cv2.DrawMatchesFlags_DEFAULT)
img3 = cv2.drawMatchesKnn(image1, kp1, image2, kp2, matches, None, **draw_params)
plt.imshow(img3, )
plt.show()