import open3d as o3d
import numpy as np

'''
Returns the lineset for a camera

@param img = image object
@param wh_ratio = width to height ratio of the camera from intrinsics
@param scale = scale size of the camera in the render
@param color = color of the camera, defaults to black
'''
def create_cam(img, wh_ratio, scale=1, color=[0,0,0]):
    line = o3d.geometry.LineSet()
    center = img.projection_center()
    scale/=10

    pts = np.array([[0,0,0], [wh_ratio, 1/wh_ratio, 2],[-wh_ratio, 1/wh_ratio, 2],[-wh_ratio, -1/wh_ratio, 2],[wh_ratio, -1/wh_ratio, 2]])
    pts = pts @ img.rotmat()

    pts = pts*scale + center
    
    
    ln = [[0,1],[0,2],[0,3],[0,4],[1,2],[2,3],[3,4],[4,1]]

    line.points = o3d.utility.Vector3dVector(pts)
    line.colors = o3d.utility.Vector3dVector([color]*len(ln))
    line.lines = o3d.utility.Vector2iVector(ln)

    return line

def generate_pts(rec, path=None):
    pcd = o3d.geometry.PointCloud()
    pts = []
    colors = []

    # Extract colors
    if path:
        rec.extract_colors_for_all_images(str(path))

    # Build the 3d structure
    for p in rec.points3D:
        pts += [rec.points3D[p].xyz]
        colors += [np.array(rec.points3D[p].color)/255]

    pcd.points = o3d.utility.Vector3dVector(pts)
    pcd.colors = o3d.utility.Vector3dVector(colors)

    return pcd

def generate_path(rec):
    i = -1

    # Build the camera path
    max_img = max(rec.images)
    min_img = min(rec.images)
    
    cam_path = o3d.geometry.LineSet()
    pts = []
    ln = []
    colors = []
        
    for img in rec.images:
        i+=1
        pts += [(img, rec.images[img].projection_center())]
        colors += [[(img-min_img)/max_img,1-(img-min_img)/max_img,0]]
        ln += [[i, i+1]]

    ln[-1] = [i,i]
    pts = [x[1] for x in pts]

    cam_path.points = o3d.utility.Vector3dVector(pts)
    cam_path.colors = o3d.utility.Vector3dVector(colors)
    cam_path.lines = o3d.utility.Vector2iVector(ln)

    return cam_path

def generate_cams(rec, scale):
    cams = []
    ratio = rec.cameras[0].width/rec.cameras[0].height

    max_img = max(rec.images)
    min_img = min(rec.images)

    for img in rec.images:
        cams += [create_cam(rec.images[img],ratio, scale, [(img-min_img)/max_img,1-(img-min_img)/max_img,0])]

    return cams

def generate_tracks(rec, pt_id):
    tracks = o3d.geometry.LineSet()

    if pt_id not in rec.points3D:
        return tracks

    pts = []
    ln = []
    colors = []

    pts += [rec.points3D[pt_id].xyz]
    for i, t in enumerate(rec.points3D[pt_id].track.elements):
        pts += [rec.images[t.image_id].projection_center()]
        ln += [[0,i]]
        colors += [[.2,.2,.2]]

    ln[-1] = [i,i]

    tracks.points = o3d.utility.Vector3dVector(pts)
    tracks.lines = o3d.utility.Vector2iVector(ln)
    tracks.colors = o3d.utility.Vector3dVector(colors)

    return tracks

'''
Visualize the SLAM system with open3D
red is the start, green is the end of the camera track

@param rec = pass in the reconstruction object
'''
def show(rec, img_path=None, show_cam_path=True, show_track=-1):
    
    pcd = generate_pts(rec, img_path)

    path = []
    tracks = []
    cams = []

    if show_cam_path:
        path = generate_path(rec)

    tracks = generate_tracks(rec, show_track)

    cams = generate_cams(rec, .5)

    o3d.visualization.draw_geometries([pcd, path, *cams, tracks])

