import numpy as np
import pycolmap
from PIL import Image, ImageOps
import matplotlib.pyplot as plt
from hloc import extract_features, match_features, reconstruction, visualization, pairs_from_exhaustive
from hloc.visualization import plot_images, read_image
from hloc.utils import viz_3d
import cv2
import glob
import os
from os import listdir

# initializing some variables
curr_frame=0
image_id=0
folder_dir = "/home/vule/Documents/codes/sequence_49/images/"

# feature extractor and matcher
orb = cv2.ORB_create(nfeatures=1000)
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

# init camera
w, h = 1280, 1024
cam = pycolmap.infer_camera_from_image(os.path.join(folder_dir,'00000.jpg'))

# init reconstruction objects
graph = pycolmap.CorrespondenceGraph()
rec = pycolmap.Reconstruction()
rec.add_camera(cam)
max_reproj_error=10.0
max_angle_error= 5.0
min_tri_angle=5

options = pycolmap.IncrementalTriangulatorOptions()
options.create_max_angle_error = max_angle_error
options.continue_max_angle_error = max_angle_error
options.merge_max_reproj_error = max_reproj_error
options.complete_max_reproj_error=max_reproj_error

length_of_video=4000
sampling_rate=50
#loading some images from the dataset
for image_path in sorted( filter(os.path.isfile, glob.glob(folder_dir + '*'))):
    if(curr_frame<=length_of_video and curr_frame>length_of_video/2): # just select some of the frames of video
        if(curr_frame%sampling_rate==0): # sample from frames
            # load image
            img=Image.open(image_path)
            img=np.array(img)
            # extract features
            keypoints, des = orb.detectAndCompute(img, None)
            points2D=[keypoint.pt for keypoint in keypoints]
            # create image object
            i=pycolmap.Image(id=image_id,name=str(image_id),camera_id=cam.camera_id)
            i.registered=True
            i.points2D = pycolmap.ListPoint2D([pycolmap.Point2D(p) for p in points2D])
            # init - first keyframe
            if(image_id==0):
                i.tvec=[0,0,0]
                tvec=i.tvec
                qvec=i.qvec
                # add image to graph
                graph.add_image(i.image_id, len(i.points2D))  
            # for other keyframes match them with previous keyframe
            else:
                # matching images
                matches = bf.match(des,old_des)
                matches = sorted(matches, key = lambda x:x.distance)
                matches=matches[:100]
                # determine relative pose
                update = pycolmap.two_view_geometry_estimation(
                    [keypoints[match.queryIdx].pt for match in matches],
                    [old_keypoints[match.trainIdx].pt for match in matches],
                    cam,
                    cam
                )
                i.tvec=old_tvec+update["tvec"]
                i.qvec=old_qvec+update["qvec"]
                matches=[(match.trainIdx,match.queryIdx) for match in matches] 
                matches=np.array(matches,dtype=np.uint32)
                # add image and correspondence to graph
                graph.add_image(i.image_id, len(i.points2D))  
                graph.add_correspondences(old_id, image_id, matches)  
            # add image to reconstruction
            rec.add_image(i)
            # save old values
            old_keypoints=keypoints
            old_des=des
            old_tvec=tvec
            old_qvec=qvec
            old_id=image_id
            image_id=image_id+1
    elif(curr_frame>length_of_video):
        break
    curr_frame=curr_frame+1

# init triangulator and traingulate images in graph and reconstruction
triangulator = pycolmap.IncrementalTriangulator(graph, rec)
for i in range(0,image_id):
    triangulator.triangulate_image(options, i)

# plot 3D reconstruction
fig = viz_3d.init_figure()
viz_3d.plot_reconstruction(fig, rec, min_track_length=0, color='rgb(255,0,0)')
fig.show()