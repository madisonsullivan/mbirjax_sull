import jax
import jax.numpy as jnp
from functools import partial
from collections import namedtuple

from build.lib.experiments.demo_5_half_sino import delta_det_row

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
        translation_vectors (jnp.ndarray):
            A 2D array of translation vectors in ALUs, specifying the translation relative to the origin.

    See Also
    --------
    TomographyModel : The base class from which this class inherits.
    """

    def __init__(self, sinogram_shape, translation, source_detector_dist, source_recon_dist):
        # param1 = source_detector_dist
        # param2 = source_recon_dist
        # view_dependent_vec1 = translation

        super().__init__(sinogram_shape, param1=param1, param2=param2, view_params_array=view_params_array)

    @classmethod
    def from_file(cls, filename):
        """
        Construct a TranslationModeModel from parameters saved using save_params()

        Args:
            filename (str): Name of the file containing parameters to load.

        Returns:
            TranslationModeModel with the specified parameters.
        """
        # Load the parameters and convert view-dependent parameters to use the geometry-specific keywords.
        # TODO: Adjust these to match the signature of __init__
        required_param_names = ['sinogram_shape', 'source_detector_dist', 'source_recon_dist']
        required_params, params = ParameterHandler.load_param_dict(filename, required_param_names, values_only=True)

        # TODO: Adjust these to match the signature of __init__
        translation = params['view_params_array']
        required_params['translation'] = translation
        del params['view_params_array']

        new_model = cls(**required_params)
        new_model.set_params(**params)
        return new_model

    def get_magnification(self):
        """
        Compute the scale factor from a voxel in first row of ROR to its projection on the detector.

        translation[t,1] = 0 means that the samples distance from the source remains constant.

        Returns:
            (float): magnification = source_detector_dist / ( source_recon_dist - translation[t,1] )
        """

        # TODO: Adjust as needed for the geometry.
        magnification = 1.0
        return magnification

    def verify_valid_params(self):
        """
        Check that all parameters are compatible for a reconstruction.
        """
        super().verify_valid_params()
        sinogram_shape, view_params_array = self.get_params(['sinogram_shape', 'view_params_array'])

        # TODO: Modify as needed for the geometry.
        if view_params_array.shape[0] != sinogram_shape[0]:
            error_message = "Number view dependent parameter vectors must equal the number of views. \n"
            error_message += "Got {} for length of view-dependent parameters and "
            error_message += "{} for number of views.".format(view_params_array.shape[0], sinogram_shape[0])
            raise ValueError(error_message)

    def get_geometry_parameters(self):
        """
        Function to get a list of the primary geometry parameters for cone beam projection.

        Returns:
            namedtuple of required geometry parameters.
        """
        # TODO: Include additional names as needed for the projectors.
        # First get the parameters managed by ParameterHandler
        geometry_param_names = ['delta_det_row', 'delta_det_channel', 'det_row_offset', 'det_channel_offset', 'delta_voxel']
        geometry_param_values = self.get_params(geometry_param_names)

        # Then get additional parameters that are calculated separately, such as psf_radius and magnification.
        # geometry_param_names += ['psf_radius']
        # geometry_param_values.append(self.get_psf_radius())
        # geometry_param_names += ['magnification']
        # geometry_param_values.append(self.get_magnifiction())

        # Then create a namedtuple to access parameters by name in a way that can be jit-compiled.
        GeometryParams = namedtuple('GeometryParams', geometry_param_names)
        geometry_params = GeometryParams(*tuple(geometry_param_values))

        return geometry_params

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
    def forward_project_pixel_batch_to_one_view(voxel_values, pixel_indices, single_view_params, projector_params):


        return sinogram_view

    @staticmethod
    @partial(jax.jit, static_argnames='projector_params')
    def back_project_one_view_to_pixel_batch(sinogram_view, pixel_indices, single_view_params, projector_params,
                                             coeff_power=1):


        return voxel_values_cylinder





    @staticmethod
    def recon_ijk_to_xyz(i, j, k, delta_voxel, delta_recon_row, t):
        """
        Convert (i, j, k) indices into the recon volume to corresponding (x, y, z) coordinates.
        """

        return x, y, z

    @staticmethod
    def geometry_xyz_to_uv_mag(x, y, z, source_detector_dist, source_recon_dist):
        """
        Convert (x, y, z) coordinates to to (u, v) detector coordinates plus the pixel-dependent magnification.
        """
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
        return m, n