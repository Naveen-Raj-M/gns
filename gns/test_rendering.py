import pytest
import os
import numpy as np
import tempfile
import shutil
import pickle
from omegaconf import DictConfig

import sys

@pytest.fixture
def verify_file_creation(dummy_pkl_data):
    temp_dir, pkl_file_name = dummy_pkl_data
    pkl_file_path = os.path.join(temp_dir, pkl_file_name)
    assert os.path.exists(pkl_file_path), f"Expected file not found: {pkl_file_path}"


sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from gns.train import rendering # Replace with actual module import

@pytest.fixture
def cfg_gif():
    """Fixture that provides a config for GIF rendering"""
    return DictConfig({
        'rendering': {
            'mode': 'gif',
        },
        'gif': {
            'step_stride': 1,
            'change_yz': False,
        }
    })

@pytest.fixture
def cfg_vtk():
    """Fixture that provides a config for VTK rendering"""
    return DictConfig({
        'rendering': {
            'mode': 'vtk',
        },
    })


@pytest.fixture
def temp_dir():
    directory = tempfile.mkdtemp()
    yield directory
    shutil.rmtree(directory)


@pytest.fixture
def dummy_pkl_data(temp_dir):
    n_trajectories = 1
    n_timesteps = 20
    n_particles = 3
    dim = 2

    metadata = {
    "bounds": [[0.1, 0.9], [0.1, 0.9]],
    "sequence_length": 1000,
    "default_connectivity_radius": 0.015,
    "dim": 2,
    "dt": 0.0025,
    "vel_mean": [-3.964619574176163e-05, -0.00026272129664401046],
    "vel_std": [0.0013722809722366911, 0.0013119977252142715],
    "acc_mean": [2.602686518497945e-08, 1.0721623948191945e-07],
    "acc_std": [6.742962470925277e-05, 8.700719180424815e-05]
    }

    initial_positions = np.random.rand(7, n_particles, dim)
    predictions = np.random.rand(n_timesteps, n_particles, dim)
    ground_truth_positions = np.random.randn(n_timesteps, n_particles, dim)
    particle_type = np.full(n_particles, n_trajectories%3)
    material_property = np.full(n_particles, np.random.rand())
    loss = (predictions - ground_truth_positions)**2

    dummy_rollout = {
        "initial_positions": initial_positions,
        "predicted_rollout": predictions, 
        "ground_truth_rollout": ground_truth_positions,
        "particle_types": particle_type, 
        "material_property": material_property
    }
    dummy_rollout["metadata"] = metadata
    dummy_rollout["loss"] = loss.mean()
    pkl_file_name = "test_input_file.pkl"
    pkl_file_path = os.path.join(temp_dir, pkl_file_name)
    with open(pkl_file_path, "wb") as f:
        pickle.dump(dummy_rollout, f)

    return temp_dir, pkl_file_name


def test_rendering_gif(dummy_pkl_data, cfg_gif):
    input_dir, input_file = dummy_pkl_data
    input_dir += '/'

    # Call the rendering function for GIF
    rendering(input_dir, "test_input_file", cfg_gif)

    # Check if the expected GIF file was created
    gif_file = os.path.join(input_dir, f"{input_file.replace('.pkl', '')}.gif")
    assert os.path.exists(gif_file), f"GIF file was not created at {gif_file}"


def test_rendering_vtk(dummy_pkl_data, cfg_vtk):
    input_dir, input_file = dummy_pkl_data
    input_dir += '/'

    rendering(input_dir, "test_input_file", cfg_vtk)

    # Check if the expected VTK output directories were created
    vtk_dir_reality = os.path.join(input_dir, f"{input_file.replace('.pkl', '')}_vtk-Reality")
    vtk_dir_gns = os.path.join(input_dir, f"{input_file.replace('.pkl', '')}_vtk_GNS")

    assert os.path.exists(vtk_dir_reality), f"VTK Reality directory was not created at {vtk_dir_reality}"
    assert os.path.exists(vtk_dir_gns), f"VTK GNS directory was not created at {vtk_dir_gns}"
