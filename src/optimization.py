import pyceres


# Adapted from: https://github.com/cvg/pyceres/blob/main/examples/test_BA.ipynb

# Full BA optimizing 3D points and poses
# Should be used at map initialization
# rec is the reconstruction object
# skip_pose indicates a list of image.id where the pose should not be optimized (i.e. origin)
#
def global_BA(rec, skip_pose=None, skip_points=None):
    if skip_points is None:
        img_list = rec.images.values()
    else:
        img_list = [element for element in rec.images.values() if element.image_id not in skip_points]
    problem = define_problem_points(rec, img_list)
    solve(problem)

    if skip_pose is None:
        img_list = rec.images.values()
    else:
        img_list = [element for element in rec.images.values() if element.image_id not in skip_pose]
    problem = define_problem_pose(rec, img_list)
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
    problem = define_problem_points(rec, [element for element in rec.images.values() if element.image_id in img_list])
    solve(problem)


# Optmizes all the 3D points
def define_problem_points(rec, img_list):
    prob = pyceres.Problem()
    loss = pyceres.TrivialLoss()
    for im in img_list:
        cam = rec.cameras[im.camera_id]
        for p in im.get_valid_points2D():
            cost = pyceres.factors.BundleAdjustmentCost(cam.model_id, p.xy, im.qvec, im.tvec)
            prob.add_residual_block(cost, loss, [rec.points3D[p.point3D_id].xyz, cam.params])
    for cam in rec.cameras.values():
        prob.set_parameter_block_constant(cam.params)
    return prob


def define_problem_pose(rec, img_list):
    prob = pyceres.Problem()
    loss = pyceres.TrivialLoss()
    for im in img_list:
        cam = rec.cameras[im.camera_id]
        for p in im.get_valid_points2D():
            cost = pyceres.factors.BundleAdjustmentCost(cam.model_id, p.xy)
            prob.add_residual_block(cost, loss, [im.qvec, im.tvec, rec.points3D[p.point3D_id].xyz, cam.params])
        prob.set_parameterization(im.qvec, pyceres.QuaternionParameterization())
    for cam in rec.cameras.values():
        prob.set_parameter_block_constant(cam.params)
    for p in rec.points3D.values():
        prob.set_parameter_block_constant(p.xyz)
    return prob


def solve(prob):
    print(prob.num_parameter_bocks(), prob.num_parameters(), prob.num_residual_blocks(), prob.num_residuals())
    options = pyceres.SolverOptions()
    options.linear_solver_type = pyceres.LinearSolverType.DENSE_QR
    options.minimizer_progress_to_stdout = True
    options.num_threads = -1
    summary = pyceres.SolverSummary()
    pyceres.solve(options, prob, summary)
    print(summary.BriefReport())
