import pyceres

# Adapted from: https://github.com/cvg/pyceres/blob/main/examples/test_BA.ipynb

# Full BA optimizing 3D points and poses
# Should be used at map initialization
# rec is the reconstruction object
# skip_pose indicates a list of image.id where the pose should not be optimized (i.e. origin)
#
import pycolmap


def global_BA(rec, skip_pose=None, skip_points=None):
    img_list = [im for im in rec.images.values() if im.registered]
    if skip_points is None:
        img_points = img_list
    else:
        img_points = [element for element in img_list if element.image_id not in skip_points]
    problem = define_problem_points(rec, img_points)
    solve(problem)

    img_pose = img_list
    # if skip_pose is None:
    #    img_pose = img_list
    # else:
    #    img_pose = [element for element in img_list if element.image_id not in skip_pose]
    problem = define_problem_pose(rec, img_pose, skip_pose)
    solve(problem)


# Local BA: all points associated to images in the img_list are optimized, the poses stay fixed
# rec is the reconstruction object
# img_list stores the id s of images that should get optimized
def local_BA(rec, img_list):
    problem = define_problem_points(rec, [element for element in rec.images.values() if element.image_id in img_list])
    solve(problem)

# Motion only BA: all points stay fixed and only image poses get optimized
# rec is the reconstruction object
# img_list stores the ids of the images where motion only BA should be applied to
def motion_only_BA(rec, img_list):
    problem = define_problem_pose(rec, [element for element in rec.images.values() if element.image_id in img_list])
    solve(problem)


# Optmizes all the 3D points
def define_problem_points(rec, img_list):
    prob = pyceres.Problem()
    loss = pyceres.TrivialLoss()
    for im in img_list:
        cam = rec.cameras[im.camera_id]
        im.qvec = pycolmap.normalize_qvec(im.qvec)
        for p in im.get_valid_points2D():
            cost = pyceres.factors.BundleAdjustmentCost(cam.model_id, p.xy, im.qvec, im.tvec)
            prob.add_residual_block(cost, loss, [rec.points3D[p.point3D_id].xyz, cam.params])
    for cam in rec.cameras.values():
        prob.set_parameter_block_constant(cam.params)
    return prob


def define_problem_pose(rec, img_list, skip_pose=None):
    prob = pyceres.Problem()
    loss = pyceres.TrivialLoss()
    points3D = []
    for im in img_list:
        cam = rec.cameras[im.camera_id]
        im.qvec = pycolmap.normalize_qvec(im.qvec)
        constant_pose = True if im.image_id in skip_pose else False
        for p in im.get_valid_points2D():
            if constant_pose:
                # cost = pyceres.factors.CreateBundleAdjustmentConstantPoseCost(cam.model_id, p.xy, im.qvec, im.tvec)
                cost = pyceres.factors.BundleAdjustmentConstantPoseCost(cam.model_id, p.xy, im.qvec, im.tvec)
                prob.add_residual_block(cost, loss, [rec.points3D[p.point3D_id].xyz, cam.params])
            else:
                cost = pyceres.factors.BundleAdjustmentCost(cam.model_id, p.xy)
                prob.add_residual_block(cost, loss, [im.qvec, im.tvec, rec.points3D[p.point3D_id].xyz, cam.params])
            points3D.append(rec.points3D[p.point3D_id])
        prob.set_parameterization(im.qvec, pyceres.QuaternionParameterization())
    for cam in rec.cameras.values():
        prob.set_parameter_block_constant(cam.params)
    #for p in rec.points3D.values():
    for p in points3D:
        prob.set_parameter_block_constant(p.xyz)
    return prob


def solve(prob, debug = True):
    if debug:
        print(prob.num_parameter_bocks(), prob.num_parameters(), prob.num_residual_blocks(), prob.num_residuals())
    options = pyceres.SolverOptions()
    options.linear_solver_type = pyceres.LinearSolverType.DENSE_QR
    options.minimizer_progress_to_stdout = debug
    options.num_threads = -1
    summary = pyceres.SolverSummary()
    pyceres.solve(options, prob, summary)
    if debug:
        print(summary.BriefReport())
