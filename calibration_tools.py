"""
===============================================================================
calibration_tools
- Sam Scholten 2019/04/29
-------------------------------------------------------------------------------

Module with tools for calibrating a stereo vision system. The core of the
module is kept in the CalibrationSystem class, which is initialised pointing to
image and data directories used for reading and writing all information, as
well as the camera it is calibrating.

There are three secondary classes. The first is the CalibrationPlate, used to
represent the physical (empircally known) calibration plate used. The Plate is
assumed to be a grid of dots, but can be abstracted to other methods - note
there is no support for this in for example the correlation_tools module.
There is then the CalibrationImage class, which represents/holds an actual
image of the calibration plate at some z value from the camera(s), and a Class
called the CalibrationImageSet which is a conveniency class for reading all of
the images to the script in sorted order.

The meat of the module is to get the pixel coordinates of the dots, then make
a polymodel, via a function in MATLAB (using the python MATLAB API) called
polyfitn (https://au.mathworks.com/matlabcentral/fileexchange/34765-polyfitn),
out of the pixel coordinates and the empirically known dot locations in real
space.

Relies heavily on correlation_tools.
Relies on stereo_camera_tools to provide information on the camera size it is
calibrating to.
MATLAB API needs to be installed independently of command line/pip etc., and
requires a license
    - see MATLAB documentation.

--------
Objects:
-------------------------------------------------------------------------------
CalibrationSystem
-------------------------------------------------------------------------------
    Methods:
    --------
        __init__(camera, img_dir, data_dir)
        add_cal_plate(cal_plate)
        add_cal_images(path_dict)
        add_dot_object(dot)
        get_sub_pixel_coords()
        make_polymodel(deg)
        pickle_polymodel(filename)
        polymodel(filename)

-------------------------------------------------------------------------------
CalibrationPlate
-------------------------------------------------------------------------------
    Methods:
    --------
        __init__(seperation, num_down, num_right, z_vals, parent)
        get_array() - uses the attributes listed in init

-------------------------------------------------------------------------------
CalibrationImageSet
-------------------------------------------------------------------------------
    Methods:
    --------
        __init__(path_dict)
        make_cal_img_list() - uses path_dict in init
                                creates CalibrationImage(s)

-------------------------------------------------------------------------------
CalibrationImage
-------------------------------------------------------------------------------
    Methods:
    --------
         __init__(z, left_path, right_path, camera)
         get_array() - creates TiffCorrObject from left and right paths

--------
Functions:
-------------------------------------------------------------------------------
    None
-------------------------------------------------------------------------------

===============================================================================
Example
+------------------------------------------------------------------------------
import matlab.engine
import correlation_tools as corrt
import calibration_tools as calbt
import stereo_vision_tools as camt
from pathlib import Path

img_dir = Path.cwd()/'images'
data_dir = Path.cwd()/'data'
mlab = matlab.engine.start_matlab()

cal_path_dict = {1900: ("cal_imgs/cal_image_left_1900.tiff",
                        "cal_imgs/cal_image_right_1900.tiff")}

camera = camt.Camera(1801, 2400, img_dir, data_dir, mlab)

cal_system = calbt.CalibrationSystem(camera, img_dir, data_dir)
cal_plate = calbt.CalibrationPlate(50, 17, 21, cal_path_dict.keys(),
                                   parent=cal_system)
cal_system.add_cal_plate(cal_plate)
cal_system.add_cal_images(cal_path_dict)

dot = corrt.GaussianCorrObject(parent=cal_system)

cal_system.add_dot_object(dot)

cal_system.get_sub_pixel_coords()
cal_system.make_polymodel(deg=3.0)
cal_system.pickle_polymodel('polymodel3.p')
camera.add_polymodel("polymodel3.p", method="pickled")
+------------------------------------------------------------------------------
+------------------------------------------------------------------------------
"""

__author__ = "Sam Scholten"


import numpy as np                      # basic numerics/array ops
import click                            # beautiful command line control
import pickle                           # save/read python object to/from disk
import matlab.engine                    # use matlab to make polyfit

import correlation_tools as corrt

###############################################################################


class CalibrationSystem(object):
    """
    Class to create a calibration function for a stereo vision system.
    Assumes dot-finding rectangular calibration plate, although it can be made
    general.
    """

    def __init__(self, camera, img_dir, data_dir):
        """
        define the pixels on the camera, we're going to pad all input
        images up to these dimensions (actually get it from parent camera)
        """
        self.parent = self.camera = camera
        self.img_dir = img_dir
        self.data_dir = data_dir

        self.cal_plate = self.dot = self.cal_img_list = self.cal_img_set = None
        self.sub_pixel_coords = self.real_coords = None
        self.polymodel = None

    ###########################################################

    def add_cal_plate(self, cal_plate):
        """
        Add a calibration plate to the system, this gives us the real space
        coordinates of the dots
        """
        if type(cal_plate) is not CalibrationPlate:
            raise TypeError('cal_plate needs to be of type CalibrationPlate')
        self.cal_plate = cal_plate
        self.real_coords = self.cal_plate.ar

    ###########################################################

    def add_cal_images(self, path_dict):
        """
        Step through the dictionary of calibration images (paths) and
        add them all (in ascending z order) to the system
        """
        self.cal_img_set = CalibrationImageSet(path_dict, self)
        self.cal_img_list = self.cal_img_set.cal_img_list

    ###########################################################

    def add_dot_object(self, dot):
        """ Add a distribution to use to find dots. I.e. could be Gaussian,
            Laplace etc.
        """
        if not isinstance(dot, corrt.CorrObject):
            raise TypeError("dot object needs to be a CorrObject")
        self.dot = dot

    ###########################################################

    def get_sub_pixel_coords(self):
        """
        Step through all the calibration images, get all the coordinates
        of the pixel locations, compile into an array.
        """

        if self.cal_img_list is None:
            raise RuntimeError("Couldn't find the cal_img_list... " +
                               "try using add_cal_images first")
        if self.dot is None:
            raise RuntimeError("No dot object found, try using add_dot_object")

        if self.cal_plate is None:
            raise RuntimeError("couldn't find calibration plate, add it first")

        coords_list = None
        # so we have the cal_img_set, step through images, correlate...
        num_d = self.cal_plate.num_d
        num_r = self.cal_plate.num_r
        with click.progressbar(self.cal_img_list,
                               label='dot finding progress') as bar:
            for cal_img in bar:
                # left image
                left_sys = corrt.CorrSystem(self.img_dir, self.data_dir,
                                            parent=self)
                left_sys.add_template(self.dot)
                left_sys.add_search_region(cal_img.left)
                left_sys.cross_correlate(method='spectral')

                # highlight=True here to graphically see dot detection
                left_coords = left_sys.find_dots(num_d, num_r, highlight=False)

                # right image
                right_sys = corrt.CorrSystem(self.img_dir, self.data_dir,
                                             parent=self)
                right_sys.add_template(self.dot)
                right_sys.add_search_region(cal_img.right)
                right_sys.cross_correlate(method='spectral')
                right_coords = right_sys.find_dots(num_d, num_r,
                                                   highlight=False)
                coords = np.concatenate((left_coords, right_coords), axis=1)

                if coords_list is None:
                    coords_list = coords
                else:
                    coords_list = np.concatenate((coords_list, coords), axis=0)

        self.sub_pixel_coords = np.array(coords_list)

    ###########################################################

    def make_polymodel(self, deg=3.0):
        """
        Make a polymodel/calibration function for the system.
        deg = degree we want to take the polyfit to (needs to be a float as
        MATLAB doesn't play nice with ints)
        """
        if self.sub_pixel_coords is None:
            raise RuntimeError("couldn't find sub pixel coordinates")
        if type(deg) is not float:
            raise TypeError("deg needs to be a float!")

        mpcoords = matlab.double(
                    np.array(np.around(self.sub_pixel_coords, decimals=0),
                             dtype=float
                             ).tolist())

        # We need to convert to list first so matlab recognises it,
        # then wrap it in a method that converts to a matlab array of doubles
        # [0] here converts to 1D vector (MATLAB is weird)
        y_real = matlab.double(self.real_coords[:, 0].copy().tolist())[0]
        x_real = matlab.double(self.real_coords[:, 1].copy().tolist())[0]
        z_real = matlab.double(self.real_coords[:, 2].copy().tolist())[0]

        x_model = self.camera.mlab.polyfitn(mpcoords, x_real, deg)
        y_model = self.camera.mlab.polyfitn(mpcoords, y_real, deg)
        z_model = self.camera.mlab.polyfitn(mpcoords, z_real, deg)

        # so this is a dict of dicts

        # these models are dicts as well (well they're structs in matlab)
        # we save these, then pass them to polyvaln (in the matlab engine)
        # when we want to actually evaluate something

        # e.g. camera.mlab.polyvaln(polymodel['x'], v) where v is:
        # v = matlab.double([il, jl, ir, jr])

        self.polymodel = {'x': x_model, 'y': y_model,
                          'z': z_model}

    ###########################################################

    def pickle_polymodel(self, filename='polymodel.p'):
        """
        Saves the interpolation(s) to filename, using pickle.
        This allows us to read in the calibration quickly without reading
        all of the images etc. which is quite time consuming
        """
        if self.polymodel is None:
            raise RuntimeError("coundn't find a !")
        path = str(self.data_dir) + '/' + filename
        pickle.dump(self.polymodel, open(path, "wb"))

    ###########################################################

    def polymodel(self, filename=None):
        """
        gets the interpolation from the system, if filename is specified
        it looks for a pickled interpolation file
        """
        # returns interpolation object (as tuple of objs or something)
        if not filename:
            if self.polymodel is None:
                raise RuntimeError(
                            "coundn't find a non-pickled polymodel!")
            return self.polymodel
        else:
            path = str(self.data_dir) + '/' + filename
            polymodel = pickle.load(open(path, "rb"))
            return polymodel

###############################################################################


class CalibrationPlate(object):
    """
    A construction, via our empirical knowledge of the plates, of the real
    space coordinates of the dots on the Calibration plate
    """
    def __init__(self, separation, num_down, num_right, z_vals, parent=None):
        """ Represents the real space object, the calibration plate.
            Also defines array of real space coords. Order of dots in this
            array is defined by sweeping through first in x, then y, then z.
            So the top row at z1, second row at z1 ... last row at z1 then
            onto z2 etc.
        """
        self.parent = parent
        self.sep = separation
        self.num_d = num_down
        self.num_r = num_right
        self.z_vals = list(z_vals)
        self.get_array()

    ###########################################################

    def get_array(self):
        """
        Get an array of the real positions of the dots in the cal plate.
        Order of array: defined by sweepign through x, then y, then z.
        So the top row at z1, second row at z1 ... last row at z1 then
        onto z2 etc.
        """

        real_pos = np.zeros((self.num_d*self.num_r*len(self.z_vals), 3))
        dot_counter = 0
        for z in self.z_vals:
            for y in range(0, self.num_d*self.sep, self.sep):
                for x in range(0, self.num_r*self.sep, self.sep):
                    real_pos[dot_counter] = np.array([float(y),
                                                      float(x),
                                                      float(z)])
                    dot_counter += 1
        self.ar = np.array(real_pos)

###############################################################################


class CalibrationImageSet(object):
    """
    Set of Calibration Images, allows us to specify the cal images
    as a dictionary and read them all in nicely and elegantly
    path_dict looks like: {z: (left_path, right_path), ...}
    """
    def __init__(self, path_dict, parent=None):
        self.parent = parent
        self.path_dict = path_dict

        self.make_cal_img_list()

    ###########################################################

    def make_cal_img_list(self):
        """
        sort the images by z value, return a list of CalImage objects
        """
        sorted_z = sorted(self.path_dict, key=self.path_dict.get)
        self.cal_img_list = []
        for z in sorted_z:
            left, right = self.path_dict[z]
            self.cal_img_list.append(
                    CalibrationImage(z, str(self.parent.img_dir) + '/' + left,
                                     str(self.parent.img_dir) + '/' + right,
                                     self.parent.camera))


###############################################################################


class CalibrationImage(object):
    """
    Calibration image containing a z value and paths to a left and
    right image
    """
    def __init__(self, z, left_path, right_path, camera):
        self.z = z
        self.left_path = left_path
        self.right_path = right_path
        self.camera = camera
        self.get_arrays()

    ###########################################################

    def get_arrays(self):
        self.left = corrt.TiffCorrObject(self.left_path, parent=self)
        self.right = corrt.TiffCorrObject(self.right_path, parent=self)

        camera_shape = self.camera.shape
        self.left.ar = corrt.pad_to_shape(self.left.ar, camera_shape)
        self.right.ar = corrt.pad_to_shape(self.right.ar, camera_shape)

###############################################################################
