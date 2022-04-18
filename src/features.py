import cv2 as cv
from pathlib import Path
from src import enums
'''
    This function should be called when detecting features.
    img_pth corresponds to the path of the image
    used_matcher is a Enum defied in the slam class that represents which detector should be used
    save is boolean and indicates if the features should be saved as an image on the disk
    out_pth and name correspond to the saved image output pth and name
'''


def detector(img_pth, used_matcher=enums.Matchers.OrbHamming, save=False, out_pth=Path(''), name='orb_out.jpg'):
    if used_matcher == enums.Matchers.OrbHamming or used_matcher == enums.Matchers.OrbFlann:
        return orb_detector(img_pth, save, out_pth, name)
    # TODO implement SuperPoint
    # elif used_matcher == slam.Matchers.SuperPoint:


'''
    This function should be called when matching features.
    img1 corresponds to the first image. It is a dict with attribute names
        "des"  the descriptor of the features
        "kp"   the keypoints as pixel indices corresponding to the descriptors
        "name" the file name of the image
    used_matcher is a Enum defied in the slam class that represents which detector should be used
    save is boolean and indicates if the matches should be saved as an image on the disk
    img_pth corresponds to the path of the image
'''


def matcher(img1, img2, used_matcher=enums.Matchers.OrbHamming, save=False, img_pth=Path('')):
    if used_matcher == enums.Matchers.OrbHamming:
        return orb_matcher(img1, img2, save, img_pth), []
    elif used_matcher == enums.Matchers.OrbFlann:
        return orb_matcher_FLANN(img1, img2)


def orb_detector(img_pth, save=False, out_pth=Path(''), name='orb_out.jpg'):
    img = cv.imread(str(img_pth), 0)
    # Initiate ORB detector
    orb = cv.ORB_create(nfeatures=500)
    # find the keypoints and descriptors with ORB
    kp, des = orb.detectAndCompute(img, None)

    if save:
        # draw only keypoints location,not size and orientation
        img2 = cv.drawKeypoints(img, kp, None, color=(0, 255, 0), flags=0)
        cv.imwrite(str(out_pth / 'images/detector' / name), img2)

    return kp, des


def orb_matcher(keypoint, query, save=False, img_pth=Path('')):
    des1 = keypoint["des"]
    des2 = query["des"]

    # create BFMatcher object
    bf = cv.BFMatcher(cv.NORM_HAMMING, crossCheck=True)
    # Match descriptors.
    matches = bf.match(des1, des2)
    # Sort them in the order of their distance.
    matches = sorted(matches, key=lambda x: x.distance)
    if save:
        draw_matches(keypoint, query, matches, img_pth)
    return matches


def orb_matcher_FLANN(keypoint, query):
    des1 = keypoint["des"]
    des2 = query["des"]
    # Nearest neighbour matching
    FLANN_INDEX_KDTREE = 1
    FLANN_INDEX_LSH = 6
    index_params = dict(algorithm=FLANN_INDEX_LSH,
                        table_number=6,  # 12
                        key_size=12,  # 20
                        multi_probe_level=1)
    search_params = dict(checks=50)  # or pass empty dictionary
    flann = cv.FlannBasedMatcher(index_params, search_params)
    matches = flann.knnMatch(des1, des2, k=2)

    # Need to draw only good matches, so create a mask
    matchesMask = [[0, 0] for i in range(len(matches))]

    # ratio test as per Lowe's paper
    for i, (m, n) in enumerate(matches):
        if m.distance < 0.7 * n.distance:  # If they are both equidistant, the ratio will be 1. Ambiguous, so discard.
            matchesMask[i] = [1, 0]
    return matches, matchesMask


def draw_matches(current_keyframe, _detector, _matches, img_pth, out_pth, indx=0):
    img1 = cv.imread(str(img_pth / current_keyframe["name"]), 0)
    img2 = cv.imread(str(img_pth / _detector["name"]), 0)
    # Draw first n matches.
    img3 = cv.drawMatches(img1, current_keyframe["kp"], img2, _detector["kp"], _matches[:10], None,
                          flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    name = "usedMatch_" + str(indx) + ".jpg"
    result = cv.imwrite(str(out_pth / 'images/matcher' / name), img3)


def draw_matches_knn(current_keyframe, _detector, _matches, matchesMask, img_pth, out_pth, indx=0):
    img1 = cv.imread(str(img_pth / current_keyframe["name"]), 0)
    img2 = cv.imread(str(img_pth / _detector["name"]), 0)
    draw_params = dict(matchColor=(0, 255, 0),
                       singlePointColor=(255, 0, 0),
                       matchesMask=matchesMask,
                       flags=cv.DrawMatchesFlags_DEFAULT)
    img3 = cv.drawMatchesKnn(img1, current_keyframe["kp"], img2, _detector["kp"], _matches, None, **draw_params)
    name = "usedMatch_" + str(indx) + ".jpg"
    cv.imwrite(str(out_pth / 'images/matcher' / name), img3)
