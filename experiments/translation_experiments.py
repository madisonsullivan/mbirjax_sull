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
num_det_rows = 40
num_det_channels = 128

# For cone beam geometry, we need to describe the distances source to detector and source to rotation axis.
source_detector_dist = 4 * num_det_channels
source_iso_dist = source_detector_dist

# For cone beam reconstruction, we need a little more than 180 degrees for full coverage.
detector_cone_angle = 2 * np.arctan2(num_det_channels / 2, source_detector_dist)

# Initialize sinogram
sinogram_shape = (num_views, num_det_rows, num_det_channels)
angles = jnp.linspace(start_angle, end_angle, num_views, endpoint=False)

ct_model_for_generation = mbirjax.ConeBeamModel(sinogram_shape, angles, source_detector_dist=source_detector_dist, source_iso_dist=source_iso_dist)
