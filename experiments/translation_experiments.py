""" This script demonstrates the effect of changing det_row_offset and det_channel_offset parameters, essentially
moving the detector in the cone beam geometry.


"""

import numpy as np
import time
import pprint
import jax.numpy as jnp
import mbirjax


# Create a phantom of a small block of pixels
"""**Set the geometry parameters**"""

# Choose the geometry type
geometry_type = 'cone'

# Set parameters for the problem size
num_views = 1
num_det_rows = 5
num_det_channels = 20

# For cone beam geometry, we need to describe the distances source to detector and source to rotation axis.
source_detector_dist = 4 * num_det_channels
source_iso_dist = source_detector_dist

# For cone beam reconstruction, we need a little more than 180 degrees for full coverage.
detector_cone_angle = 2 * np.arctan2(num_det_channels / 2, source_detector_dist)

start_angle = -(np.pi + detector_cone_angle) * (1/2)
end_angle = (np.pi + detector_cone_angle) * (1/2)

# Initialize sinogram
sinogram_shape = (num_views, num_det_rows, num_det_channels)
angles = jnp.linspace(start_angle, end_angle, num_views, endpoint=False)

ct_model_for_generation = mbirjax.ConeBeamModel(sinogram_shape, angles, source_detector_dist=source_detector_dist, source_iso_dist=source_iso_dist)

# Generate 3D Shepp Logan phantom
print('Creating phantom')
phantom = ct_model_for_generation.gen_modified_3d_sl_phantom()

# Generate synthetic sinogram data
print('Creating sinogram')
sinogram = ct_model_for_generation.forward_project(phantom)
sinogram = np.array(sinogram)

# View sinogram
title = 'Original sinogram \nUse the sliders to change the view or adjust the intensity range.'
mbirjax.slice_viewer(sinogram, slice_axis=0, title=title, slice_label='View')