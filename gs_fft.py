from scene.dataset_readers import readColmapCameras
from scene.colmap_loader import read_extrinsics_text, read_intrinsics_text, qvec2rotmat, \
    read_extrinsics_binary, read_intrinsics_binary, read_points3D_binary, read_points3D_text
from scene import Scene
import os
import torch
import numpy as np
import subprocess
from scene.gaussian_model import GaussianModel
from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams, get_combined_args, OptimizationParams
from gaussian_renderer import render, network_gui
import matplotlib.pyplot as plt
from train import training
from utils.general_utils import safe_state
from scene.colmap_loader import rotmat2qvec, qvec2rotmat
import sys
from operator import itemgetter
import torchvision
from torchvision import transforms
import copy


# def readColmapSceneInfo(path, images='images', eval=False, llffhold=8):
#     try:
#         cameras_extrinsic_file = os.path.join(path, "sparse/0", "images.bin")
#         cameras_intrinsic_file = os.path.join(path, "sparse/0", "cameras.bin")
#         cam_extrinsics = read_extrinsics_binary(cameras_extrinsic_file)
#         cam_intrinsics = read_intrinsics_binary(cameras_intrinsic_file)
#     except:
#         cameras_extrinsic_file = os.path.join(path, "sparse/0", "images.txt")
#         cameras_intrinsic_file = os.path.join(path, "sparse/0", "cameras.txt")
#         cam_extrinsics = read_extrinsics_text(cameras_extrinsic_file)
#         cam_intrinsics = read_intrinsics_text(cameras_intrinsic_file)

#     reading_dir = "images" if images == None else images
#     cam_infos_unsorted = readColmapCameras(cam_extrinsics=cam_extrinsics, cam_intrinsics=cam_intrinsics, images_folder=os.path.join(path, reading_dir))
#     cam_infos = sorted(cam_infos_unsorted.copy(), key = lambda x : x.image_name)
#     return cam_infos

def createRotationandTraslationMatrix(cam_infos):
    translation_list = np.empty((len(cam_infos), 3))    # N x 3
    rotmat_list = np.empty((len(cam_infos), 3 , 3))       # N x 3 x 3 (rotmat, not q)
    quaternion_list = np.empty((len(cam_infos), 4))    # N x 4
    # camera_center_list =  np.empty((len(cam_infos), 3 , 3)) 
    for i in range(len(cam_infos)):
        translation_list[i] = cam_infos[i].T
        rotmat_list[i] = cam_infos[i].R
        quaternion_list[i] = rotmat2qvec(cam_infos[i].R)
    
    return translation_list, rotmat_list, quaternion_list
    
def createDistanceMatrix(translation_list):
    views = torch.tensor(translation_list)
    # Compute pairwise distances between
    distances = torch.cdist(views, views, p=2) # An N x N matrix representing the Euclidean distances between every point pairs
    return distances

def sortDistances(yet_visit, view_index:int, distances:torch.tensor):  # Sort the distances of all views in respect of a certain view
    # Get the distances of all views from the view of interest
    distances_from_view = distances[view_index]
    # remove visited elements
    distances_from_view = np.multiply(distances_from_view, yet_visit)
    # Sort distances and get the indices
    sorted_distances, sorted_indices = torch.sort(distances_from_view)
    slice_index = next((index for index, value in enumerate(sorted_distances) if value != 0), -1)
    return sorted_distances[slice_index:], sorted_indices[slice_index:]

def removePoint(taken_views: torch.tensor, sorted_indices: torch.tensor): # To remove views already selected
    mask = ~torch.isin(sorted_indices, taken_views)
    remaining_indices = sorted_indices[mask]
    return remaining_indices

def removeDistances(taken_views: torch.tensor, sorted_distances: torch.tensor): # To remove distances of views already selected
    # Create a mask that is True for indices not in `taken_views`
    mask = torch.ones(sorted_distances.size(0), dtype=bool)
    mask[taken_views] = False
    # Apply the mask to remove the specified indices
    remaining_distances = sorted_distances[mask]
    return remaining_distances

@torch.no_grad()
def get_views(dataset_path, dataset_img_path, iterations, SH=0):
    # Set up command line argument parser
    parser = ArgumentParser(description="Get camera views from dataset")
    pipe = PipelineParams(parser)
    model = ModelParams(parser, sentinel=True)
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--skip_train", action="store_true")
    parser.add_argument("--skip_test", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    args = get_combined_args(parser)

    # Equal to the "Command Line Arguments for train.py" in 3DGS
    args.source_path= dataset_img_path
    args.model_path= dataset_path
    args.images = 'images'
    args.eval = False
    args.resolution = 1.0
    args.data_device = "cuda"
    model_0 = model.extract(args)
    model_0.model_path = dataset_path
    model_0.source_path = dataset_img_path

    # Problem: how to prevent the model to read from the args....
    # load the Gaussian model (dataset)
    current_pointcloud_path = os.path.join(dataset_path, "point_cloud/iteration_"+str(iterations)+"/point_cloud.ply")
    gaussians = GaussianModel(SH)
    gaussians.load_ply(current_pointcloud_path)
    scene = Scene(model_0, gaussians, load_iteration=args.iteration, shuffle=False)
    views = scene.getTrainCameras()
    return views

def cam_render(gaussians, view, bg= torch.tensor([0, 0, 0], dtype=torch.float32, device="cuda")):
    parser = ArgumentParser(description="gaussian splatting")
    pipe = PipelineParams(parser)
    rendering = render(view, gaussians, pipe, bg)["render"]
    return rendering

def train(source_path, model_path, iterations):
    parser = ArgumentParser(description="Training script parameters")
    lp = ModelParams(parser)

    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    parser.add_argument('--ip', type=str, default="127.0.0.1")
    parser.add_argument('--port', type=int, default=6009)
    parser.add_argument('--debug_from', type=int, default=-1)
    parser.add_argument('--detect_anomaly', action='store_true', default=False)
    parser.add_argument("--test_iterations", nargs="+", type=int, default=[7_000, 30_000])
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[7_000, 30_000])
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[])
    parser.add_argument("--start_checkpoint", type=str, default = None)
    args = parser.parse_args(sys.argv[1:])
    args.source_path= source_path
    args.model_path= model_path
    args.iterations= iterations
    args.save_iterations.append(args.iterations)
    args.sh_degree = 0
    
    print("Optimizing " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    # Start GUI server, configure and run training
    # network_gui.init(args.ip, args.port)
    torch.autograd.set_detect_anomaly(args.detect_anomaly)
    training(lp.extract(args), op.extract(args), pp.extract(args), args.test_iterations, args.save_iterations, args.checkpoint_iterations, args.start_checkpoint, args.debug_from)

    # All done
    print("\nTraining complete.")

def load_gaussians(datapath, SH = 0):
    current_pointcloud_path = os.path.join(datapath, "point_cloud/iteration_"+str(iterations)+"/point_cloud.ply")
    gaussians = GaussianModel(sh_degree = SH)
    gaussians.load_ply(current_pointcloud_path)
    return gaussians

@torch.no_grad()
def fft2d(img):
    gray = transforms.Grayscale(img)
    img_fft = torch.fft.fft2(img)
    return img_fft.cpu().numpy()

def init_scene(dataset_path, dataset_img_path, initial_view_idx, iterations = 1000, SH = 0):
    # read all views from the dataset
    # load camera information of all views
    all_views = get_views(dataset_path=dataset_path, dataset_img_path= dataset_img_path, iterations=10000, SH=SH)
    yet_visit = np.ones(len(all_views))
    visited = list(range(initial_view_idx+1))
    # record the vistied views
    yet_visit[0: initial_view_idx+1] = 0
    translation_list, rotmat_list, _ = createRotationandTraslationMatrix(all_views)
    distance_matrix = createDistanceMatrix(translation_list)
    return all_views, yet_visit, visited, translation_list, rotmat_list, distance_matrix
    all_views, yet_visit, visited, translation_list, rotmat_list, distance_matrix

def cal_drift(current_view, gt_view):
    T_drift = current_view.T - gt_view.T
    q_drift = rotmat2qvec(current_view.R) - rotmat2qvec(gt_view.R)
    return T_drift, q_drift

def estimate_similarity_transformation(source: np.ndarray, target: np.ndarray) -> np.ndarray:
    """
    Estimate similarity transformation (rotation, scale, translation) from source to target (such as the Sim3 group).
    """
    k, n = source.shape

    mx = source.mean(axis=1)
    my = target.mean(axis=1)
    source_centered = source - np.tile(mx, (n, 1)).T
    target_centered = target - np.tile(my, (n, 1)).T

    sx = np.mean(np.sum(source_centered**2, axis=0))
    sy = np.mean(np.sum(target_centered**2, axis=0))

    Sxy = (target_centered @ source_centered.T) / n

    U, D, Vt = np.linalg.svd(Sxy, full_matrices=True, compute_uv=True)
    V = Vt.T
    rank = np.linalg.matrix_rank(Sxy)
    if rank < k:
        raise ValueError("Failed to estimate similarity transformation")

    S = np.eye(k)
    if np.linalg.det(Sxy) < 0:
        S[k - 1, k - 1] = -1

    R = U @ S @ V.T

    s = np.trace(np.diag(D) @ S) / sx
    t = my - s * (R @ mx)

    return R, s, t


def get_current_view(current_views, current_img_name):
    return next((view for view in current_views if view.image_name == current_img_name), None)

def select_nbv_c(current_view, current_view_idx, distance_matrix, all_views, current_gaussians, nearest_n = 10):
    
    current_gt_view = all_views[current_view_idx]
    T_drift, q_drift = cal_drift(current_view, current_gt_view)
    # Sort the distances of all views in respect of the current view
    sorted_distances, sorted_indices = sortDistances(yet_visit, current_view_idx, distance_matrix)
    close_views_indices = sorted_indices[: nearest_n].cpu().numpy() # Get the 5 nearest views

    med_freq_dict = {}

    for i in close_views_indices:
        close_view = all_views[i]
        T_drift = close_view.T - current_gt_view.T
        q_drift = rotmat2qvec(close_view.R) - rotmat2qvec(current_gt_view.R)

        # Transform the close view from the ground truth frame to the current Gaussian 
        # Rectify the drift
        render_view = copy.deepcopy(close_view)
        render_view.T = render_view.T + T_drift
        render_view.R = qvec2rotmat(rotmat2qvec(render_view.R) + q_drift)

        # render and save image
        render_img = cam_render(current_gaussians, render_view)
        target_dir = "./output/render_img"
        img_path = os.path.join(target_dir, f"img_{i+1}.png")
        torchvision.utils.save_image(render_img, img_path)


        frequency = fft2d(render_img)
        med_freq_dict[i] = np.median(np.log(abs(frequency)))

        # TUNE v to be other numbers to fit colmap
        filtered_dict = {k: v for k, v in med_freq_dict.items() if v > 0}

        # print("img: ", i+1, "median frequency: ", med_freq_dict[i])

        # fig = plt.figure()
        # ax1, ax2 = fig.subplots(2, 1)
        # ax1.imshow(np.log(abs(frequency)[0]))
    
    if filtered_dict == {}:
        nbv_idx = max(med_freq_dict, key=med_freq_dict.get)
    else:
        # Select the view with the minimum median frequency (Most blurry)
        nbv_idx = min(filtered_dict, key=filtered_dict.get)

    if nbv_idx == None:
        nbv_idx = max(med_freq_dict, key=med_freq_dict.get)

    return nbv_idx

def select_nbv_u(R, s, t, rot_q, current_view_idx, distance_matrix, all_views, current_gaussians, nearest_n = 10):
    
    current_gt_view = all_views[current_view_idx]

    # Sort the distances of all views in respect of the current view
    sorted_distances, sorted_indices = sortDistances(yet_visit, current_view_idx, distance_matrix)
    close_views_indices = sorted_indices[: nearest_n].cpu().numpy() # Get the 5 nearest views

    med_freq_dict = {}

    for i in close_views_indices:
        close_view = all_views[i]
        # T_drift = close_view.T - current_gt_view.T
        # q_drift = rotmat2qvec(close_view.R) - rotmat2qvec(current_gt_view.R)

        # Transform the close view from the ground truth frame to the current Gaussian 
        # Rectify the drift
        render_view = copy.deepcopy(close_view)
        render_view.T = s * R @ render_view.T + t
        render_view.R = qvec2rotmat(rotmat2qvec(render_view.R) + rot_q)

        # render and save image
        render_img = cam_render(current_gaussians, render_view)
        target_dir = "./output/render_img"
        img_path = os.path.join(target_dir, f"img_{i+1}.png")
        torchvision.utils.save_image(render_img, img_path)


        frequency = fft2d(render_img)
        med_freq_dict[i] = np.median(np.log(abs(frequency)))

        # TUNE v to be other numbers to fit colmap
        filtered_dict = {k: v for k, v in med_freq_dict.items() if v > 0}
    
    if filtered_dict == {}:
        nbv_idx = max(med_freq_dict, key=med_freq_dict.get)
    else:
        # Select the view with the minimum median frequency (Most blurry)
        nbv_idx = min(filtered_dict, key=filtered_dict.get)

    if nbv_idx == None:
        nbv_idx = max(med_freq_dict, key=med_freq_dict.get)

    return nbv_idx



if __name__ == "__main__":
    # Parameters to change
    dataset_path = './output/playroom_sh0_10000'
    dataset_img_path = './tandt_db/db/playroom'
    current_view_idx = 9 # Select the first n images as already visited
    current_img_name = "DSC05581"

    training_data_path = './tandt_db/db/playroom_10'
    output_path = "./output/debug"
    iterations = 1000
    iterations_nbv = 90

    # Initialize the ground_truth scene
    all_views, yet_visit, visited, translation_list, rotmat_list, distance_matrix = init_scene(dataset_path, dataset_img_path, current_view_idx, iterations = iterations)
    gt_pt_mat = np.empty((current_view_idx+1, 3))
    gt_name_list = []

    for i in range(current_view_idx+1):
        gt_pt_mat[i] = all_views[i].T
        gt_name_list.append(all_views[i].image_name)

    # The place for While loop
    round = 0
    while round < iterations_nbv:
        # Clean up
        if os.path.exists(training_data_path+'/distorted'):
            subprocess.run(["rm", "-r", f"{training_data_path}/distorted"])
            subprocess.run(["rm", "-r", f"{training_data_path}/images"])
            subprocess.run(["rm", "-r", f"{training_data_path}/sparse"])
            subprocess.run(["rm", "-r", f"{training_data_path}/stereo"])
            subprocess.run(["rm", f"{training_data_path}/run-colmap-geometric.sh"])
            subprocess.run(["rm", f"{training_data_path}/run-colmap-photometric.sh"])


        if os.path.exists(output_path+'/cameras.json'):
            subprocess.run(["rm", f"{output_path}/cameras.json"])
            subprocess.run(["rm", f"{output_path}/cfg_args"])
            subprocess.run(["rm", f"{output_path}/input.ply"])
            subprocess.run(["rm", "-r", f"{output_path}/point_cloud/iteration_{iterations-100}/"])


    # train the Gaussian model and load the Gaussians
    #if not os.path.exists(output_path+'/point_cloud'):
        # Setup the colmap for the training data
        colmap_cmd = "python " + "convert.py " + "-s " + training_data_path
        colmap = subprocess.run(["python", "convert.py", "-s", training_data_path])
        print(colmap.stdout)
        
        # Does not work, need to find a way to run the training script
        #update_gs = subprocess.run(["python", "train.py", "-s", training_data_path, "-m", output_path, "-i", str(iterations)])
        #print(update_gs.stdout)

        train(source_path=training_data_path, model_path=output_path, iterations=iterations)
        current_gaussians = load_gaussians(output_path, SH = 0)
        
        current_views = get_views(dataset_path=output_path, dataset_img_path= training_data_path, iterations=iterations, SH=0)
        current_view = get_current_view(current_views, current_img_name)

        gs_pt_mat = np.empty([10, 3])
        for name in gt_name_list:
            gs_pt_mat[gt_name_list.index(name)] = next((view.T for view in current_views if view.image_name == name), None)
        
        # R: rotation, s: scale, t: translation
        R, s, t = estimate_similarity_transformation(gt_pt_mat.T, gs_pt_mat.T)

        # gs_pt_after_conversion = (s * R @ gt_pt_mat.T).T + t
        

        # Calculate the rotation
        q_drift_sum = np.zeros(4, dtype=float)
        for i in visited:
            img_name = all_views[i].image_name
            gt_pose_q = rotmat2qvec(all_views[i].R)
            gs_view = next((view for view in current_views if view.image_name == img_name), None)
            if gs_view is None:
                print(f"View {img_name} is not found in the current views")
                continue
            gs_pose_q = rotmat2qvec(gs_view.R)
            q_drift_sum += gs_pose_q - gt_pose_q
        rot_q = q_drift_sum / len(visited)

        if current_view is None:
            visiting_file = open(f"{output_path}/visited.txt", mode="a")
            visiting_file.write(f"Current view {current_img_name} is discarded, changing to previous view {prev_img_name}" + "\n")
            visiting_file.close()
            current_view = prev_view
            current_img_name = prev_img_name
            current_view_idx = prev_view_idx

        # Select the next best view
        # nbv_idx = select_nbv_c(current_view, current_view_idx, distance_matrix, all_views, current_gaussians)

        nbv_idx = select_nbv_u(R, s, t, rot_q, current_view_idx, distance_matrix, all_views, current_gaussians)

        # Back up the previous view
        prev_view_idx = current_view_idx
        prev_view = current_view
        prev_img_name = all_views[prev_view_idx].image_name

        copy_img = subprocess.run(["cp", f"{dataset_img_path}/images/{all_views[nbv_idx].image_name}.jpg", f"{training_data_path}/input/{all_views[nbv_idx].image_name}.jpg"])
        print(copy_img.returncode)

        visited.append(nbv_idx)
        # Umeyama does not work with too many reference points...
        # gt_pt_mat = np.append(gt_pt_mat, (all_views[nbv_idx].T).reshape(1, 3), axis=0)
        # gt_name_list.append(all_views[nbv_idx].image_name)

        current_view_idx = nbv_idx
        yet_visit[current_view_idx] = 0
        current_img_name = all_views[current_view_idx].image_name

        
        visiting_file = open(f"{output_path}/visited.txt", mode="a")
        visiting_file.write(current_img_name + "\n")
        visiting_file.close()

        if round % 10 == 0:
            save_ply = subprocess.run(["cp", f"{output_path}/point_cloud/iteration_{iterations}/point_cloud.ply", f"{output_path}/point_cloud/save_0/point_cloud_{round}.ply"])
            print("Save ply: round ", round, save_ply.returncode)
        round += 1
        iterations += 100
        # Drag the ply file to https://antimatter15.com/splat/ to visualize