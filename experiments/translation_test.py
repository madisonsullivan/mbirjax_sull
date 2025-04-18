import numpy as np
import time
import pprint
import jax.numpy as jnp


import mbirjax





# Set parameters for the problem size
num_views = 1
num_det_rows = 3
num_det_channels = 8

# For cone beam geometry, we need to describe the distances source to detector and source to rotation axis.
source_detector_dist = 4 * num_det_channels
source_recon_dist = source_detector_dist


# Initialize sinogram
sinogram_shape = (num_views, num_det_rows, num_det_channels)
translations = jnp.array([[0,0]])
print(translations[0][0])
ct_model_for_generation = mbirjax.TranslationModeModel(sinogram_shape, translations, source_detector_dist=source_detector_dist, source_recon_dist=source_recon_dist)

# Generate 3D Shepp Logan phantom
print('Creating original phantom')
phantom = ct_model_for_generation.gen_modified_3d_sl_phantom()

title = f'Phantom'
mbirjax.slice_viewer(phantom, title=title)

# Generate synthetic sinogram data
print('Creating original sinogram')
sinogram = ct_model_for_generation.forward_project(phantom)
sinogram = np.array(sinogram)

# View sinogram
title = 'Original sinogram'
mbirjax.slice_viewer(sinogram, slice_axis=0, title=title, slice_label='View')