import jax
import jax.numpy as jnp
from functools import partial
from collections import namedtuple
import warnings
import mbirjax
from mbirjax import tomography_utils

from mbirjax import TomographyModel, ParameterHandler, tomography_utils



class TranslationModeModel(TomographyModel):
    """
    A class designed for handling forward and backward projections using translations with a cone beam source. This extends
    :ref:`TomographyModelDocs`. This class offers specialized methods and parameters tailored for translation mode.

    This class inherits all methods and properties from the :ref:`TomographyModelDocs` and may override some
    to suit translation mode geometrical requirements. See the documentation of the parent class for standard methods
    like setting parameters and performing projections and reconstructions.

    Parameters not included in the constructor can be set using the set_params method of :ref:`TomographyModelDocs`.
    Refer to :ref:`TomographyModelDocs` documentation for a detailed list of possible parameters.

    Args:
        sinogram_shape (tuple):
            Shape of the sinogram as a tuple in the form `(views, rows, channels)`, where 'views' is the number of
            different translations, 'rows' correspond to the number of detector rows, and 'channels' index columns of
            the detector that are assumed to be aligned with the rotation axis.
        translations (jnp.ndarray):
            A 2D array of translation vectors in ALUs, specifying the translation relative to the origin.

    See Also
    --------
    TomographyModel : The base class from which this class inherits.
    """

    def __init__(self, sinogram_shape, translations, source_detector_dist, source_recon_dist):
        ''' Following the API reference, Args:
            param1 = source_detector_dist : Distance from source to detector in ALU
            param2 = source_recon_dist : Distance from source to first row in ROR in ALU
            view_dependent_vec1 = translations : 1D array of 2D translations in ALU (viewed from the source)
        '''
        view_dependent_vecs = [vec.flatten() for vec in [translations]]
        self.bp_psf_radius = 1
        self.entries_per_cylinder_batch = 128
        self.slice_range_length = 0
        try:
            view_params_array = jnp.stack(view_dependent_vecs, axis=1)
        except ValueError as e:
            raise ValueError("Incompatible view dependent vector lengths:  all view-dependent vectors must have the "
                             "same length.")

        super().__init__(sinogram_shape, view_params_array=view_params_array, source_detector_dist=source_detector_dist,
                         source_recon_dist=source_recon_dist)

    @classmethod
    def from_file(cls, filename):
        """
        Construct a TranslationModeModel from parameters saved using save_params()

        Args:
            filename (str): Name of the file containing parameters to load.

        Returns:
            TranslationModeModel with the specified parameters.
        """
        # Load the parameters and convert view-dependent parameters to the TranslationModeModel keywords.
        required_param_names = ['sinogram_shape', 'source_detector_dist', 'source_recon_dist']
        required_params, params = mbirjax.ParameterHandler.load_param_dict(filename, required_param_names, values_only=True)

        # Collect the required parameters into a separate dictionary and remove them from the loaded dict.
        translations = params['view_params_array']
        del params['view_params_array']
        required_params['translations'] = translations

        # Get an instance with the required parameters, then set any optional parameters
        new_model = cls(**required_params)
        new_model.set_params(**params)
        return new_model

    def get_magnification(self):
        """
        Compute the scale factor from a voxel in first row of ROR to its projection on the detector.

        Note: This version assumes that the samples distance from the source remains constant.

        Returns:
            (float): magnification = source_detector_dist / source_recon_dist
        """
        source_detector_dist, source_recon_dist = self.get_params(['source_detector_dist', 'source_recon_dist'])
        if jnp.isinf(source_detector_dist):
            magnification = 1
        else:
            magnification = source_detector_dist / source_recon_dist
        return magnification


    def verify_valid_params(self):
        """
        Check that all parameters are compatible for a reconstruction.
        """
        super().verify_valid_params()
        sinogram_shape, view_params_array = self.get_params(['sinogram_shape', 'view_params_array'])

        # Check number of translations match sinogram views
        if view_params_array.shape[0] != sinogram_shape[0]:
            error_message = "Number view dependent parameter vectors must equal the number of views. \n"
            error_message += "Got {} for length of view-dependent parameters and "
            error_message += "{} for number of views.".format(view_params_array.shape[0], sinogram_shape[0])
            raise ValueError(error_message)

        # Check for cone angle > 45 degrees
        source_detector_dist, delta_det_row, det_row_offset = \
            self.get_params(['source_detector_dist', 'delta_det_row', 'det_row_offset'])
        half_detector_height = delta_det_row * sinogram_shape[1] / 2 + jnp.abs(det_row_offset)
        if half_detector_height > source_detector_dist:
            warnings.warn('Cone angle is more than 45 degrees.  This will likely produce recon artifacts.')

        # TODO: Check other necessary parameters?

    def get_geometry_parameters(self):
        """
        Function to get a list of the primary geometry parameters for cone beam projection.

        Returns:
            namedtuple of required geometry parameters.
        """
        # TODO: Include additional names as needed for the projectors.
        # First get the parameters managed by ParameterHandler
        geometry_param_names = ['delta_det_row', 'delta_det_channel', 'det_row_offset', 'det_channel_offset',
                                'source_detector_dist', 'delta_voxel']
        geometry_param_values = self.get_params(geometry_param_names)

        # Then get additional parameters that are calculated separately, such as psf_radius and magnification.
        geometry_param_names += ['magnification', 'psf_radius', 'bp_psf_radius',
                                 'entries_per_cylinder_batch', 'slice_range_length']
        geometry_param_values.append(self.get_magnification())
        geometry_param_values.append(self.get_psf_radius())
        geometry_param_values.append(self.bp_psf_radius)
        geometry_param_values.append(self.entries_per_cylinder_batch)
        geometry_param_values.append(self.slice_range_length)

        # Then create a namedtuple to access parameters by name in a way that can be jit-compiled.
        GeometryParams = namedtuple('GeometryParams', geometry_param_names)
        geometry_params = GeometryParams(*tuple(geometry_param_values))

        return geometry_params

    def get_psf_radius(self):
        """
        Compute the integer radius of the PSF kernel for cone beam projection.
        """
        delta_det_row, delta_det_channel, source_detector_dist, recon_shape, delta_voxel = self.get_params(
            ['delta_det_row', 'delta_det_channel', 'source_detector_dist', 'recon_shape', 'delta_voxel'])
        magnification = self.get_magnification()

        # Compute minimum detector pitch
        delta_det = jnp.minimum(delta_det_row, delta_det_channel)

        # Compute maximum magnification
        if jnp.isinf(source_detector_dist):
            max_magnification = 1
            min_magnification = 1
        else:
            source_to_iso_dist = source_detector_dist / magnification
            # This isn't exactly the closest pixel since we're not accounting for rotation but for realistic cases it shouldn't matter.
            source_to_closest_pixel = source_to_iso_dist - jnp.maximum(recon_shape[0], recon_shape[1])*delta_voxel
            max_magnification = source_detector_dist / source_to_closest_pixel
            source_to_farthest_pixel = source_to_iso_dist + jnp.maximum(recon_shape[0], recon_shape[1])*delta_voxel
            min_magnification = source_detector_dist / source_to_farthest_pixel

        # Compute the maximum number of detector rows/channels on either side of the center detector hit by a voxel
        psf_radius = int(jnp.ceil(jnp.ceil((delta_voxel * max_magnification / delta_det)) / 2))
        # Then repeat for the back projection from detector elements to voxels.
        # The voxels closest to the detector will be covered the most by a given detector element.
        # With magnification=1, the number of voxels per element would be delta_det / delta_voxel
        max_voxels_per_detector = delta_det / (min_magnification * delta_voxel)
        self.bp_psf_radius = int(jnp.ceil(jnp.ceil(max_voxels_per_detector) / 2))

        self.slice_range_length = int(1 + 2 * self.bp_psf_radius + \
                                  jnp.ceil(self.entries_per_cylinder_batch * max_voxels_per_detector))

        return psf_radius


    def auto_set_recon_size(self, sinogram_shape, no_compile=True, no_warning=False):
        """ Compute the automatic recon shape cone beam reconstruction.
        """
        delta_det_row, delta_det_channel = self.get_params(['delta_det_row', 'delta_det_channel'])
        delta_voxel = self.get_params('delta_voxel')
        num_det_rows, num_det_channels = sinogram_shape[1:3]
        magnification = self.get_magnification()

        num_recon_rows = int(jnp.round(num_det_channels * ((delta_det_channel / delta_voxel) / magnification)))
        num_recon_cols = num_recon_rows
        num_recon_slices = int(jnp.round(num_det_rows * ((delta_det_row / delta_voxel) / magnification)))

        recon_shape = (num_recon_rows, num_recon_cols, num_recon_slices)
        self.set_params(no_compile=no_compile, no_warning=no_warning, recon_shape=recon_shape)

    @staticmethod
    @partial(jax.jit, static_argnames='projector_params')
    def forward_project_pixel_batch_to_one_view(voxel_values, pixel_indices, translation, projector_params):
        """
                Forward project a set of voxels determined by indices into the flattened array of size num_rows x num_cols.

                Args:
                    voxel_values (jax array):  2D array of shape (num_indices, num_slices) of voxel values, where
                        voxel_values[i, j] is the value of the voxel in slice j at the location determined by indices[i].
                    pixel_indices (jax array of int):  1D vector of indices into flattened array of size num_rows x num_cols.
                    translation (array):  translation for this view
                    projector_params (namedtuple):  tuple of (sinogram_shape, recon_shape, get_geometry_params())

                Returns:
                    jax array of shape (num_det_rows, num_det_channels)
                """
        recon_shape = projector_params.recon_shape
        num_recon_slices = recon_shape[2]
        if voxel_values.shape[0] != pixel_indices.shape[0] or len(voxel_values.shape) < 2 or \
                voxel_values.shape[1] != num_recon_slices:
            raise ValueError('voxel_values must have shape[0:2] = (num_indices, num_slices)')

        vertical_fan_projector = TranslationModeModel.forward_vertical_fan_pixel_batch_to_one_view
        horizontal_fan_projector = TranslationModeModel.forward_horizontal_fan_pixel_batch_to_one_view

        new_voxel_values = vertical_fan_projector(voxel_values, pixel_indices, translation, projector_params)
        sinogram_view = horizontal_fan_projector(new_voxel_values, pixel_indices, translation, projector_params)

        return sinogram_view

    @staticmethod
    def forward_vertical_fan_pixel_batch_to_one_view(voxel_values, pixel_indices, translation, projector_params):
        """
        Apply a fan beam forward projection in the vertical direction separately to each voxel determined by indices
        into the flattened array of size num_rows x num_cols.  This returns an array corresponding to the same pixel
        locations, but using values obtained from the projection of the original voxel cylinders onto a detector column,
        so the output array has size (len(pixel_indices), num_det_rows).

        Args:
            voxel_values (jax array):  2D array of shape (num_pixels, num_recon_slices) of voxel values, where
                voxel_values[i, j] is the value of the voxel in slice j at the location determined by indices[i].
            pixel_indices (jax array of int):  1D vector of shape (len(pixel_indices), ) holding the indices into
                the flattened array of size num_rows x num_cols.
            translation (array):  translation for this view
            projector_params (namedtuple):  tuple of (sinogram_shape, recon_shape, get_geometry_params())

        Returns:
            jax array of shape (num_pixels, num_det_rows)
        """
        pixel_map = jax.vmap(TranslationModeModel.forward_vertical_fan_one_pixel_to_one_view,
                             in_axes=(0, 0, None, None))
        new_pixels = pixel_map(voxel_values, pixel_indices, translation, projector_params)

        return new_pixels

    @staticmethod
    def forward_horizontal_fan_pixel_batch_to_one_view(voxel_values, pixel_indices, translation, projector_params):
        """
        Apply a horizontal fan beam transformation to a set of voxel cylinders. These cylinders are assumed to have
        slices aligned with detector rows, so that a horizontal fan beam maps a cylinder slice to a detector row.
        This function returns the resulting sinogram view.

        Args:
            voxel_values (jax array):  2D array of shape (num_pixels, num_recon_slices) of voxel values, where
                voxel_values[i, j] is the value of the voxel in slice j at the location determined by indices[i].
            pixel_indices (jax array of int):  1D vector of shape (len(pixel_indices), ) holding the indices into
                the flattened array of size num_rows x num_cols.
            translation (array):  translation for this view
            projector_params (namedtuple):  tuple of (sinogram_shape, recon_shape, get_geometry_params())

        Returns:
            jax array of shape (num_det_rows, num_det_channels)
        """

        # Get all the geometry parameters - we use gp since geometry parameters is a named tuple and we'll access
        # elements using, for example, gp.delta_det_channel, so a longer name would be clumsy.
        gp = projector_params.geometry_params
        num_views, num_det_rows, num_det_channels = projector_params.sinogram_shape

        # Get the data needed for horizontal projection
        n_p, n_p_center, W_p_c, cos_alpha_p_xy = TranslationModeModel.compute_horizontal_data(pixel_indices, translation, projector_params)
        L_max = jnp.minimum(1, W_p_c)

        # Allocate the sinogram array
        sinogram_view = jnp.zeros((num_det_rows, num_det_channels))

        # Do the horizontal projection
        for n_offset in jnp.arange(start=-gp.psf_radius, stop=gp.psf_radius + 1):
            n = n_p_center + n_offset
            abs_delta_p_c_n = jnp.abs(n_p - n)
            L_p_c_n = jnp.clip((W_p_c + 1) / 2 - abs_delta_p_c_n, 0, L_max)
            A_chan_n = gp.delta_voxel * L_p_c_n / cos_alpha_p_xy
            A_chan_n *= (n >= 0) * (n < num_det_channels)
            sinogram_view = sinogram_view.at[:, n].add(A_chan_n.reshape((1, -1)) * voxel_values.T)

        return sinogram_view

    @staticmethod
    def forward_vertical_fan_one_pixel_to_one_view(voxel_cylinder, pixel_index, translation, projector_params):
        """
        Apply a fan beam forward projection in the vertical direction to the pixel determined by indices
        into the flattened array of size num_rows x num_cols.  This returns a vector obtained from the projection of
        the original voxel cylinder onto a detector column, so the output vector has length num_det_rows.

        Args:
            voxel_cylinder (jax array):  1D array of shape (num_recon_slices, ) of voxel values, where
                voxel_cylinder[j] is the value of the voxel in slice j at the location determined by pixel_index.
            pixel_index (int):  Index into the flattened array of size num_rows x num_cols.
            translation (array):  translation for this view.
            projector_params (namedtuple):  tuple of (sinogram_shape, recon_shape, get_geometry_params())

        Returns:
            jax array of shape (num_det_rows,)

        Note:
            This is a helper function used in vmap in :meth:`ConeBeamModel.forward_vertical_fan_pixel_batch_to_one_view`
        This method has the same signature and output as that method, except single int pixel_index is used
        in place of the 1D pixel_indices, and likewise only a single voxel cylinder is returned.
        """
        # Get all the geometry parameters - we use gp since geometry parameters is a named tuple and we'll access
        # elements using, for example, gp.delta_det_channel, so a longer name would be clumsy.
        gp = projector_params.geometry_params
        num_views, num_det_rows, num_det_channels = projector_params.sinogram_shape
        recon_shape = projector_params.recon_shape
        num_slices = voxel_cylinder.shape[0]

        # From pixel index, compute y and pixel_mag
        y, pixel_mag = TranslationModeModel.compute_y_mag_for_pixel(pixel_index, translation, recon_shape, projector_params)

        # The code above depends only on the pixel - a single point.  z is a potentially large vector
        # Here we compute cos_phi_p:  1 / cos_phi_p determines the projection length through a voxel
        # For computational efficiency, we use that to scale the voxel_cylinder values.
        # TODO:  possibly convert to a jitted function with donate_argnames to avoid copies for z, v, phi_p, cos_phi_p
        k = jnp.arange(len(voxel_cylinder))
        z = gp.delta_voxel * (k - (num_slices - 1) / 2.0) + gp.recon_slice_offset  # recon_ijk_to_xyz
        v = pixel_mag * z  # geometry_xyz_to_uv_mag
        # Compute vertical cone angle of voxels
        phi_p = jnp.arctan2(v, gp.source_detector_dist)  # compute_vertical_data_single_pixel
        cos_phi_p = jnp.cos(phi_p)  # We assume the vertical angle |phi_p| < 45 degrees so cos_alpha_p_z = cos_phi_p
        scaled_voxel_values = voxel_cylinder / cos_phi_p
        # End TODO

        # Get the length of projection of detector on vertical voxel profile (in fraction of voxel size)
        # This is also the slope of the map from voxel index to detector index
        W_p_r = (pixel_mag * gp.delta_voxel) / gp.delta_det_row
        slope_k_to_m = W_p_r
        L_max = jnp.minimum(1, W_p_r)  # Maximum fraction of a detector that can be covered by one voxel.

        # Set up detector row indices array (0, 10, 20, ..., 10*num_slice_batches)
        det_rows_per_batch = gp.entries_per_cylinder_batch
        det_rows_per_batch = min(det_rows_per_batch, num_det_rows)
        num_det_row_batches = (num_det_rows + det_rows_per_batch - 1) // det_rows_per_batch
        det_row_indices = det_rows_per_batch * jnp.arange(num_det_row_batches)

        det_center_row = (num_det_rows - 1) / 2.0
        row_batch = jnp.arange(det_rows_per_batch)

        # Set up a function to map over subsets of detector rows
        def create_det_column_rows(start_index):
            # We need to match the back projector, so we have to determine the fraction of each voxel that projects
            # to each detector.
            # First project the detector centers to the voxel cylinder
            m_center = start_index + row_batch  # Center of detector elements
            v_m = (m_center - det_center_row) * gp.delta_det_row - gp.det_row_offset  # Detector center in ALUs
            z_m = v_m / pixel_mag  # z coordinate of the projection of the center of the first detector element in this batch
            # Convert to voxel fractional index and find the center of each voxel
            k_m = (z_m - gp.recon_slice_offset) / gp.delta_voxel + (num_slices - 1) / 2.0
            k_m_center = jnp.round(k_m).astype(int)  # Center of the voxel hit by the center of the detector
            # Then map the center of the voxels back to the detector.
            m_p = slope_k_to_m * (k_m_center - k_m[0]) + m_center[0]  # Projection to detector of voxel centers

            # Allocate space
            new_column_batch = jnp.zeros(det_rows_per_batch)
            # Do the vertical projection
            for k_offset in jnp.arange(start=-gp.bp_psf_radius, stop=gp.bp_psf_radius+1):
                k_ind = k_m_center + k_offset  # Indices of the current set of voxels touched by the detector elements
                # The projection of these centers is the projection of k_m_center (which is m_p) plus
                # the offset times the slope of the map from voxel index to detector index
                abs_delta_p_r_m = jnp.abs(m_p + slope_k_to_m * k_offset - m_center)  # Distance from projection of center of voxel to center of detector
                A_row_k = jnp.clip((W_p_r + 1) / 2 - abs_delta_p_r_m, 0, L_max)  # Fraction of the detector hit by this voxel
                A_row_k *= (k_ind >= 0) * (k_ind < num_slices)
                new_column_batch = jnp.add(new_column_batch, A_row_k * scaled_voxel_values[k_ind])

            return new_column_batch, None

        det_column, _ = jax.lax.map(create_det_column_rows, det_row_indices)
        det_column = det_column.flatten()
        det_column = jax.lax.slice_in_dim(det_column, 0, num_det_rows)
        return det_column

    @staticmethod
    @jax.jit
    def compute_y_mag_for_pixel(pixel_index, angle, recon_shape, projector_params):

        gp = projector_params.geometry_params
        row_index, col_index = jnp.unravel_index(pixel_index, recon_shape[:2])

        # Compute the un-rotated coordinates relative to iso
        # Note the change in order from (i, j) to (y, x)!!
        y_tilde = gp.delta_voxel * (row_index - (recon_shape[0] - 1) / 2.0)
        x_tilde = gp.delta_voxel * (col_index - (recon_shape[1] - 1) / 2.0)

        # Precompute cosine and sine of view angle, then do the rotation
        cosine = jnp.cos(angle)  # length = num_views
        sine = jnp.sin(angle)  # length = num_views

        y = sine * x_tilde + cosine * y_tilde

        # Convert from xyz to coordinates on detector
        pixel_mag = 1 / (1 / gp.magnification - y / gp.source_detector_dist)
        return y, pixel_mag

    @staticmethod
    def compute_horizontal_data(pixel_indices, translation, projector_params):
        """
        Compute the quantities n_p, n_p_center, W_p_c, cos_alpha_p_xy needed for vertical projection.

        Args:
            pixel_indices (1D jax array of int):  indices into flattened array of size num_rows x num_cols.
            translation (array): The translation array for this view.
            projector_params (namedtuple): tuple of (sinogram_shape, recon_shape, get_geometry_params()).

        Returns:
            n_p, n_p_center, W_p_c, cos_alpha_p_xy
        """
        # Get all the geometry parameters - we use gp since geometry parameters is a named tuple and we'll access
        # elements using, for example, gp.delta_det_channel, so a longer name would be clumsy.
        gp = projector_params.geometry_params

        num_views, num_det_rows, num_det_channels = projector_params.sinogram_shape
        recon_shape = projector_params.recon_shape
        num_recon_rows, num_recon_cols, num_recon_slices = recon_shape

        # Convert the index into (i,j,k) coordinates corresponding to the indices into the 3D voxel array
        row_index, col_index = jnp.unravel_index(pixel_indices, recon_shape[:2])
        slice_index = jnp.arange(1)

        x_p, y_p, _ = TranslationModeModel.recon_ijk_to_xyz(row_index, col_index, slice_index, gp.delta_voxel,
                                                      gp.delta_recon_row, translation)

        # Convert from xyz to coordinates on detector
        # pixel_mag should be kept in terms of magnification to allow for source_detector_dist = jnp.Inf
        pixel_mag = 1 / (1 / gp.magnification - y_p / gp.source_detector_dist)
        # Compute the physical position that this voxel projects onto the detector
        u_p = pixel_mag * x_p
        det_center_channel = (num_det_channels - 1) / 2.0  # num_of_cols

        # Calculate indices on the detector grid
        n_p = (u_p + gp.det_channel_offset) / gp.delta_det_channel + det_center_channel  # Sync with detector_uv_to_mn
        n_p_center = jnp.round(n_p).astype(int)

        # Compute horizontal and vertical cone angle of pixel
        # theta_p = jnp.arctan2(u_p, gp.source_detector_dist)

        # Compute projected voxel width along columns and rows (in fraction of detector size)
        W_p_c = pixel_mag * (gp.delta_voxel / gp.delta_det_channel)

        horizontal_data = (n_p, n_p_center, W_p_c)

        return horizontal_data








    @staticmethod
    @partial(jax.jit, static_argnames='projector_params')
    def back_project_one_view_to_pixel_batch(sinogram_view, pixel_indices, single_view_params, projector_params,
                                             coeff_power=1):


        return voxel_values_cylinder



    @staticmethod
    def recon_ijk_to_xyz(i, j, k, delta_voxel, delta_recon_row, translation):
        """
        Convert (i, j, k) indices into the recon volume to corresponding (x, y, z) coordinates.

        Args:
            translation (1D array):  [horizontal_offset, vertical_offset], array specifying the translation for the current
            view relative to the origin

        Note: This version assumes that the samples distance from the source remains constant.
        """
        # Compute the x, y, z coordinates for the given translation
        # Note the change in order from (i, j) to (y, x)!!
        y = delta_recon_row * i
        x = delta_voxel * j - translation[0]
        z = delta_voxel * k - translation[1]

        return x, y, z

    @staticmethod
    def geometry_xyz_to_uv_mag(x, y, z, source_detector_dist, magnification):
        """
        Convert (x, y, z) coordinates to to (u, v) detector coordinates plus the pixel-dependent magnification.
        """
        # Compute the magnification at this specific voxel
        # The following expression is valid even when source_detector_dist = jnp.Inf
        pixel_mag = 1 / (1 / magnification - y / source_detector_dist)

        # Compute the physical position that this voxel projects onto the detector
        u = pixel_mag * x
        v = pixel_mag * z

        return u, v, pixel_mag

    @staticmethod
    @jax.jit
    def detector_uv_to_mn(u, v, delta_det_channel, delta_det_row, det_channel_offset, det_row_offset, num_det_rows,
                          num_det_channels):
        """
        Convert (u, v) detector coordinates to fractional indices (m, n) into the detector.

        Note:
            This version does not account for nonzero detector rotation.
        """
        # Get the center of the detector grid for columns and rows
        det_center_row = (num_det_rows - 1) / 2.0  # num_of_rows
        det_center_channel = (num_det_channels - 1) / 2.0  # num_of_cols

        # Calculate indices on the detector grid
        m = (v + det_row_offset) / delta_det_row + det_center_row
        n = (u + det_channel_offset) / delta_det_channel + det_center_channel  # Sync with compute_horizontal_data

        return m, n