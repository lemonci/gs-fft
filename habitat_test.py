import math
import os
import random
import sys

import git
import imageio
import magnum as mn
import numpy as np

from matplotlib import pyplot as plt

# function to display the topdown map
from PIL import Image

import habitat_sim
from habitat_sim.utils import common as utils
from habitat_sim.utils import viz_utils as vut

import subprocess
from train import training
import torch
from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams, get_combined_args, OptimizationParams
from utils.general_utils import safe_state

test_scene = "/home/mistlab/Documents/ml/habitat/test/mp3d_example/17DRP5sb8fy/17DRP5sb8fy.glb"
data_path  = "/home/mistlab/Documents/ml/gs-fft/tandt_db/habitat"


rgb_sensor = True # @param {type:"boolean"}
depth_sensor = True # @param {type:"boolean"}
semantic_sensor = True # @param {type:"boolean"}
step_radius = 0.1
initial_view_num = 10
output_path = "./output/habit_debug"
iterations = 1000


sim_settings = {
    "width": 256, # Spatial resolution of the observations
    "height": 256,
    "scene": test_scene, # Scene path
    "default_agent": 0,
    "sensor_height": 1.5, # Height of sensors in meters
    "color_sensor": rgb_sensor,
    "depth_sensor": depth_sensor, # Depth sensor
    "semantic_sensor": semantic_sensor, # Semantic sensor
    "seed": 1,
    "enable_physics": False, # kinematics only
}

# simulator

def make_cfg(settings):
    sim_cfg = habitat_sim.SimulatorConfiguration()
    sim_cfg.gpu_device_id = 0
    sim_cfg.scene_id = settings["scene"]
    sim_cfg.enable_physics = settings["enable_physics"]

    # Note: all sensors must have the same resolution
    sensors = {
        "color_sensor": {
            "sensor_type": habitat_sim.SensorType.COLOR,
            "resolution": [settings["height"], settings["width"]],
            "position": [0.0, settings["sensor_height"], 0.0],
        },
        "depth_sensor": {
            "sensor_type": habitat_sim.SensorType.DEPTH,
            "resolution": [settings["height"], settings["width"]],
            "position": [0.0, settings["sensor_height"], 0.0],
        },    
        "semantic_sensor": {
            "sensor_type": habitat_sim.SensorType.SEMANTIC,
            "resolution": [settings["height"], settings["width"]],
            "position": [0.0, settings["sensor_height"], 0.0],
        },
    }

    sensor_specs = []
    for sensor_uuid, sensor_params in sensors.items():
        if settings[sensor_uuid]:
            sensor_spec = habitat_sim.CameraSensorSpec()
            sensor_spec.uuid = sensor_uuid
            sensor_spec.sensor_type = sensor_params["sensor_type"]
            sensor_spec.resolution = sensor_params["resolution"]
            sensor_spec.position = sensor_params["position"]
            sensor_spec.camera_type = habitat_sim.SensorSubType.PINHOLE

            sensor_specs.append(sensor_spec)

    # Here you can specify the amount of displacement in a forward action and then turn angle
    agent_cfg = habitat_sim.agent.AgentConfiguration()
    agent_cfg.sensor_specifications = sensor_specs
    agent_cfg.action_space = {
        "move_forward": habitat_sim.agent.ActionSpec(
                "move_forward", habitat_sim.agent.ActuationSpec(amount = 0.1)
        ),
        "move_backward": habitat_sim.agent.ActionSpec(
                "move_backward", habitat_sim.agent.ActuationSpec(amount = 0.1)
        ),
        "move_left": habitat_sim.agent.ActionSpec(
                "move_left", habitat_sim.agent.ActuationSpec(amount = 0.1)
        ),
        "move_right": habitat_sim.agent.ActionSpec(
                "move_right", habitat_sim.agent.ActuationSpec(amount = 0.1)
        ),
        "turn_left": habitat_sim.agent.ActionSpec(
            "turn_left", habitat_sim.agent.ActuationSpec(amount=10.0)
        ),
        "turn_right": habitat_sim.agent.ActionSpec(
            "turn_right", habitat_sim.agent.ActuationSpec(amount=10.0)
        ),
    }
    return habitat_sim.Configuration(sim_cfg, [agent_cfg])

def print_scene_recur(scene, limit_output=10):
    print(
        f"House has {len(scene.levels)} levels, {len(scene.regions)} regions and {len(scene.objects)} objects"
    )

    count = 0
    for level in scene.levels:
        print(
            f"Level id: {level.id}, center:{level.aabb.center},"
            f" dims:{level.aabb.sizes}"
        )
        for region in level.regions:
            print(
                f"Region id:{region.id}, category:{region.category.name()},"
                f" center:{region.aabb.center}, dims:{region.aabb.sizes}"
            )
            for obj in region.objects:
                print(
                    f"Object id:{obj.id}, category:{obj.category.name()},"
                    f" center:{obj.aabb.center}, dims:{obj.aabb.sizes}"
                )
                count += 1
                if count >= limit_output:
                    return None
                
def get_rgb_image(sim):
    observations = sim.get_sensor_observations()
    rgb = observations["color_sensor"]
    #semantic = observations["semantic_sensor"]
    #depth = observations["depth_sensor"]
    return rgb

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


# def initial_views(agent, sim, initial_view_num):



cfg = make_cfg(sim_settings)
# Needed to handle out of order cell run in Colab
try: # Got to make initialization idiot proof
    sim.close()
except NameError:
    pass
sim = habitat_sim.Simulator(cfg)

# Print semantic annotation information (id, category, bounding box details)
# about levels, regions and objects in a hierarchical fashion
scene = sim.semantic_scene
# print_scene_recur(scene)

# the randomness is needed when choosing the actions
random.seed(sim_settings["seed"])
sim.seed(sim_settings["seed"])

# Set agent state
agent = sim.initialize_agent(sim_settings["default_agent"])
agent_state = habitat_sim.AgentState()
agent_state.position = np.array([-0.6, 0.0, 0.0]) # world space
agent.set_state(agent_state)

# Get agent state
agent_state = agent.get_state()
# print("agent_state: position", agent_state.position, "rotation", agent_state.rotation)

# Initial observations
image_list = []
camera_pose = []


# clean up
if os.path.exists(data_path+'/sparse'):
    subprocess.run(["rm", "-r", f"{data_path}/sparse"])
    subprocess.run(["rm", "-r", f"{data_path}/images"])
    subprocess.run(["rm", "-r", f"{data_path}/database.db"])
    subprocess.run(["mkdir", f"{data_path}/sparse"])
    subprocess.run(["mkdir", f"{data_path}/sparse/model"])
    subprocess.run(["mkdir", f"{data_path}/sparse/0"])
    subprocess.run(["mkdir", f"{data_path}/images"])
#    subprocess.run(["rm", f"{training_data_path}/run-colmap-geometric.sh"])
#    subprocess.run(["rm", f"{training_data_path}/run-colmap-photometric.sh"])


# if os.path.exists(output_path+'/cameras.json'):
#     subprocess.run(["rm", f"{output_path}/cameras.json"])
#     subprocess.run(["rm", f"{output_path}/cfg_args"])
#     subprocess.run(["rm", f"{output_path}/input.ply"])
#     subprocess.run(["rm", "-r", f"{output_path}/point_cloud/iteration_{iterations-100}/"])

with open(f"{data_path}/sparse/model/camera.txt", 'a') as camera_txt:
    camera_txt.write("1 SIMPLE_PINHOLE 256 256 128 128 128\r\n")
    camera_txt.close()

with open(f"{data_path}/sparse/model/points3D.txt", 'a') as points3D_txt:
    points3D_txt.close()


for i in range(initial_view_num):
    # get the rgb image
    rgba = get_rgb_image(sim)
    rgb = Image.fromarray(rgba).convert('RGB')
    rgb.save(data_path+'/images/'+str(i+1).zfill(5)+'.jpg')
    image_list.append(rgb)
    agent_state = agent.get_state()
    rotation = agent_state.rotation
    position = agent_state.position
    camera_pose.append([rotation, position])
    # Write the index of image
    with open(f"{data_path}/sparse/model/images.txt", 'a') as images_txt:
        images_txt.write(str(i+1)+" ")
        # Write the rotation quaternion
        images_txt.write(str(round(rotation.w, 6))+" ")
        images_txt.write(str(round(rotation.x, 6))+" ")
        images_txt.write(str(round(rotation.y, 6))+" ")
        images_txt.write(str(round(rotation.z, 6))+" ")
        # Write the translation
        images_txt.write(str(round(position[0], 6))+" ")
        images_txt.write(str(round(position[1], 6))+" ")
        images_txt.write(str(round(position[2], 6))+" ")
        # Write the camera id aka 1
        images_txt.write("1 ")
        # Write the image name
        images_txt.write(str(i+1).zfill(5)+'.jpg\r\n\r\n')
        images_txt.close()

    pathfinder_seed = 4 # @param {type:"integer"}
    sim.pathfinder.seed(pathfinder_seed)
    nav_point = sim.pathfinder.get_random_navigable_point_near(circle_center=position, radius = 0.1, max_tries=100)
    agent_state.position = nav_point
    agent.set_state(agent_state)
    
    # take an action
    pathfinder_seed = 4
    sim.pathfinder.seed(pathfinder_seed)
    
    
   
    # action = "move_forward"
    # sim.step(action)

subprocess.run(["colmap", "feature_extractor", "--database_path", f"{data_path}/database.db", "--image_path", f"{data_path}/images"])
subprocess.run(["colmap", "exhaustive_matcher", "--database_path", f"{data_path}/database.db"]
)
subprocess.run(["colmap", "point_triangulator", "--database_path", f"{data_path}/database.db", "--image_path", f"{data_path}/images", "--input_path", f"{data_path}/sparse/model", "--output_path", f"{data_path}/sparse/0"]
)
train(source_path=data_path, model_path=output_path, iterations=iterations)


# Display the images

print("Displaying images")










print("Agent's initial state: ", sim.agents[0].get_state())