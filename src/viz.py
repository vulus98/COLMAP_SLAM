import open3d as o3d

'''
Returns the lineset for a camera
@param img = image object
'''
def create_cam(img):
    line = o3d.geometry.LineSet()
    pts = []
    ln = []



'''
Visualize the SLAM system with open3D
red is the start, green is the end of the camera track

@param rec = pass in the reconstruction object
'''
def show(rec):
    pcd = o3d.geometry.PointCloud()
    pts = []
    colors = []

    cams = o3d.geometry.LineSet()
    ln = []

    # Build the 3d structure
    for p in rec.points3D:
        pts += [rec.points3D[p].xyz]
        colors += [rec.points3D[p].color]

    pcd.points = o3d.utility.Vector3dVector(pts)
    pcd.colors = o3d.utility.Vector3dVector(colors)

    pts = []
    colors = []

    i = -1
    # Build the camera path
    img_count = rec.num_reg_images()
        
    for img in rec.images:
        i+=1
        pts += [(img, rec.images[img].projection_center())]
        colors += [[1-i/img_count,i/img_count,0]]
        ln += [[i, i+1]]

    ln[-1] = [i,i]
    pts.sort(key=lambda x: x[0])
    pts = [x[1] for x in pts]

    cams.points = o3d.utility.Vector3dVector(pts)
    cams.colors = o3d.utility.Vector3dVector(colors)
    cams.lines = o3d.utility.Vector2iVector(ln)

    o3d.visualization.draw_geometries([pcd, cams])

