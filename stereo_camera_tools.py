"""
===============================================================================
stereo_camera_tools
- Sam Scholten 2019/04/29
-------------------------------------------------------------------------------

Module with tools for Computer Stereo Vision. Requires the correlation_tools
module for cross correlation methods/objects, and the calibration_tools module
if a new calibration is to be created (although one that has been 'saved') can
be read into the Camera object without the module imported.
The MATLAB engine API for Python3.6 (strictly 3.6 must be used, due to MATLAB's
restriction) must be installed and licensed on the user computer.

The core of the module is kept in the Camera class, which is initialised with
it's pixel height and width, the address of the image and data directories for
reading and writing all information, and the MATLAB engine object (which must
be running).

There is one secondary class, SearchGeometry, which holds the geometry of the
'window' that passes over the images used to construct a real space plot. Thus
SeachGeometry defines the resolution of the scan.

The general idea of the module is to read in a calibration, read in an image
pair (left, right), then iterate over the windows (defined by a SearchGeometry)
in the left image, find them in the right, pass the pixel coordinates into
the calibration function (evaluated in MATLAB), which returns real space
coordinates of the window. There are then analysis methods to remove spurious
vectors, cut the data down to the calibration (i.e. valid) region, and plot.

--------
Objects:
-------------------------------------------------------------------------------
Camera
-------------------------------------------------------------------------------
    Methods:
    --------
        __init__(height, width, img_dir, data_dir, mlab)
        add_polymodel(input, method="pickled")        - calibration fn
        add_left_tiff(path)
        add_right_tiff(path)
        pad_to_cam(ar)                      - pad images to size of camera
        multi_pass(geom_lst, conv_to_real=True)
                                            - multi pass over left image
        pickle_result(filename='result.p')  - save real space coords array
        load_result(filename='result.p')    - load real space coords array
        plot_depth_map()                    - plot dpx as fn of il,jl location
        clip_boundaries(ybounds=None, xbounds=None, zbounds=None)
                                            - clip result to a domain
        remove_outliers(cont=0.1, neighs=10, set_to=None)
                                            - remove outliers via
                                              LocalOutlierFactor sklearn alg.
        subtract_quad(half_width, indep='x', highlight=False)
                                            - subtract of a quadratic in one
                                              independent variable
        plot_real_space(zmin=None, zmax=None, scatter=False)
                                            - plot real space coords, as a
                                              triangulated surface (trisurf)

-------------------------------------------------------------------------------
SearchGeometry
-------------------------------------------------------------------------------
    Methods:
    --------
         __init__(self, t_shape, w_shape, overlap)



----------
Functions:
-------------------------------------------------------------------------------
non_neg(x)      - returns x or 0 if x is negative
left_x_cal      - basic function to help with sorting by first element of array
-------------------------------------------------------------------------------

===============================================================================
Example
+------------------------------------------------------------------------------
import matlab.engine
import stereo_vision_tools as camt
from pathlib import Path

img_dir = Path.cwd()/'images'
data_dir = Path.cwd()/'data'
mlab = matlab.engine.start_matlab()

# get the camera ready
camera = camt.Camera(1801, 2400, img_dir, data_dir, mlab)
camera.add_polymodel("polymodel3.p", method="pickled")
path = str(img_dir) + '/non_cal_imgs/'
camera.add_left_tiff(path + 'test_left_2.tiff')
camera.add_right_tiff(path + 'test_right_2.tiff')

# template shape, window shape (+template...), overlaps
fst_pass = camt.SearchGeometry((90, 90), (90, 120), (0, 0))
sec_pass = camt.SearchGeometry((30, 30), (30, 60), (15, 15))

# pass over images to get real space data
camera.multi_pass([fst_pass, sec_pass], conv_to_real=True)

camera.pickle_result('test2_p3.p')

# analyse data
camera.load_result('test2_p3.p')
camera.clip_boundaries(ybounds=(-200, 1000),
                       xbounds=(-200, 1200),
                       zbounds=(1850, 2050))
camera.subtract_quad(350, indep='x', highlight=False)
camera.remove_outliers(cont=0.05, neighs=25, set_to=None)

# plot data
camera.plot_real_space(zmin=0, zmax=2200, scatter=True)
camera.plot_real_space(zmin=1800, zmax=2200, scatter=True)
+------------------------------------------------------------------------------
+------------------------------------------------------------------------------
"""

__author__ = "Sam Scholten"

import numpy as np                          # basic numerics/array ops
import click                                # pretty command line interface
import pickle                               # w/r python object to/from disk
import matplotlib.pyplot as plt             # basic plotting
from mpl_toolkits.mplot3d import Axes3D     # 3D plotting module
from matplotlib import cm                   # colourmaps
import matlab.engine                        # access matlab from python!
from sklearn.neighbors import LocalOutlierFactor    # find outliers!
import numpy.polynomial.polynomial as n_poly   # 1D polyfit

import correlation_tools as corrt


class Camera(object):
    """
    This object represents the 2-Camera Stereo Vision detector (ok perhaps not
    the best naming). It holds a calibration and the images (left, right) to be
    converted to real space (result). It also holds methods for analysing and
    printing the resultant data.
    Requires the MATLAB Python API engine to run.
    """
    def __init__(self, height, width, img_dir, data_dir, mlab):

        self.height = height
        self.width = width
        self.shape = (height, width)
        self.img_dir = img_dir
        self.data_dir = data_dir
        self.mlab = mlab
        self.calibration = self.left = self.right = self.geom = None
        self.pxl_lst = None

    ###########################################################

    def add_polymodel(self, input, method="pickled"):
        """
        add via a pickled polymodel, or directly as dictionary
        """

        # input is either an interpolation dict, or a path

        if method not in ("pickled", "dict"):
            raise ValueError("method needs to be either 'pickled' or 'dict'")

        if method == 'pickled':
            if type(input) is not str:
                raise TypeError("for pickled method, input must be str")
            path = str(self.data_dir) + '/' + input
            self.calibration = pickle.load(open(path, "rb"))

        elif method == 'dict':
            if type(input) is not dict:
                raise TypeError("for dict method, input must be dict")
            self.calibration = input

    ###########################################################

    def add_left_tiff(self, path):
        """
        add left tiff image via the path to the image
        """
        self.left = corrt.TiffCorrObject(path, parent=self)
        self.left.ar = self.pad_to_cam(self.left.ar)

    ###########################################################

    def add_right_tiff(self, path):
        """
        add right tiff image via the path to the image
        """
        self.right = corrt.TiffCorrObject(path, parent=self)
        self.right.ar = self.pad_to_cam(self.right.ar)

    ###########################################################

    def pad_to_cam(self, ar):
        """
        pad an array with zeros to the shape of the camera.
        (required, as the calibration is done with respect to pixel locations
        of a certain sized array, and if we input an image of another shape
        that calibration won't make sense anymore)
        """
        if np.ndim(ar) != 2:
            raise RuntimeError("ar needs to be 2D")
        h, w = self.shape
        if ar.shape[0] > h or ar.shape[1] > w:
            # cut down to the size of the camera
            return ar[:h, :w]
        change_y = abs(ar.shape[0] - h)
        change_x = abs(ar.shape[1] - w)

        bottom = change_y//2
        top = change_y - bottom

        right = change_x//2
        left = change_x - right
        pad_width = ((top, bottom), (left, right))
        return np.pad(ar, pad_width, 'constant', constant_values=0)

    ###########################################################

    def multi_pass(self, geom_lst, conv_to_real=True):
        """
        Multi pass over the left image, finding each window in the right
        image and immediately converting to real space coordinates. At each
        window, it steps to the next geometry in geom_lst (if there is one)
        recursively to find each subwindow near the correlated region found
        for the parent window. For maximum effect, pick subsequent geometries
        that are smaller than their parent, but are an integer factor of their
        size in each dimension.
        """
        if conv_to_real and self.calibration is None:
            raise RuntimeError("No calibration found for this camera")
        if type(geom_lst) is not list:
            raise TypeError("geom_lst needs to be a list of SearchGeometry" +
                            " Objects, corresponding to each secondary pass")
        self.pxl_lst = []
        try:
            self._recursive_pass(self.left, geom_lst, 0)
            if conv_to_real:
                self._conv_to_real_space()
        except IndexError:
            raise RuntimeError("the input geom_lst was empty?")

    ###########################################################

    def _recursive_pass(self,
                        window, geom_lst, depth, shift=None, parent=None):
        """
        pop off each geometry in geom_lst, each time taking the right coords
        as a new location (shift) about which to search at the next 'geom_lst
        depth'. Of course, iterates through left at each step (where left is
        decreased in size at each depth).
        """
        if type(geom_lst) is not list:
            raise TypeError("geom_lst needs to be a list of SearchGeometry" +
                            " Objects, corresponding to each secondary pass")
        if not isinstance(window, corrt.CorrObject):
            raise TypeError("inputted window is not a CorrObject")

        try:
            next_geom = geom_lst[depth]
        except IndexError:
            raise IndexError

        # first turn window into an iterable
        iter_win = corrt.IterableCorrObject(window, next_geom, parent=parent)
        while True:
            try:
                # coordinates of the child window wrt parent (iter) window
                w_jl, w_il, child_win = iter_win._next_wind()
            except StopIteration:
                # print('\tStopped')
                break
            if not child_win.ar.any():
                # if the window is all zeros, let's skip it
                continue
            try:
                pxls = self._wind_finder(w_jl, w_il, child_win, next_geom,
                                         shift=shift)
            except corrt.SizeError:
                # size of search region is too small to correlate with templ
                # corresponds to not finding the object in the right image
                # - equivalent to skipping really bad shift values
                continue

            new_shift = (pxls[2] - pxls[0], pxls[3] - pxls[1])
            try:
                self._recursive_pass(child_win, geom_lst, depth+1,
                                     shift=new_shift, parent=iter_win)
            except IndexError:
                self.pxl_lst.append(pxls)

    ###########################################################

    def _wind_finder(self, jl, il, window, geom, shift=None):
        """
        find object in right array about left image coords.
        """

        # create the correlation system
        corr_sys = corrt.CorrSystem(self.img_dir, self.data_dir)
        templ = window

        # now cut out the right array about the input coords
        search_region = corrt.CorrObject()
        wh, ww = geom.w_shape
        th, tw = geom.t_shape
        if shift is not None:
            dpy, dpx = shift
        else:
            dpy, dpx = 0, 0
        lo_y = non_neg(jl + dpy - wh//2 - th//2)
        hi_y = non_neg(jl + dpy + wh//2 + th//2 +
                       int(2 * (wh/2 - wh//2)) +
                       int(2 * (th/2 - th//2)))
        lo_x = non_neg(il + dpx - ww//2 - tw//2)
        hi_x = non_neg(il + dpx + ww//2 + tw//2 +
                       int(2 * (ww/2 - ww//2)) +
                       int(2 * (tw/2 - tw//2)))
        search_region.ar = self.right.ar[
                            lo_y: hi_y, lo_x: hi_x
                            ]

        corr_sys.add_template(templ)
        corr_sys.add_search_region(search_region)
        corr_sys.cross_correlate(method='spectral')
        jr, ir = corr_sys.max_corr_loc()

        # the locations above are pixel *within the right.ar slice
        # above* so we shift all the pixel locations to the center of
        # the window and shift to within the 'whole' array not just the
        # right slice
        jr += lo_y + window.ar.shape[0]//2
        ir += lo_x + window.ar.shape[1]//2

        # also shift left locations to middle of window
        jl += window.ar.shape[0]//2
        il += window.ar.shape[1]//2

        pxls = [jl, il, jr, ir]
        return pxls

    ###########################################################

    def _conv_to_real_space(self):
        """
        pass the data across to MATLAB to calculate the real space values
        from the calibration function
        """
        if self.pxl_lst is None:
            raise RuntimeError("couldn't find pixel list to convert.")
        mpxls = matlab.double(self.pxl_lst)
        x = self.mlab.polyvaln(self.calibration['x'], mpxls)
        y = self.mlab.polyvaln(self.calibration['y'], mpxls)
        z = self.mlab.polyvaln(self.calibration['z'], mpxls)
        self.result = np.concatenate((y, x, z), axis=1)

    ###########################################################

    def plot_depth_map(self):
        """ docstring """
        if self.pxl_lst is None:
            raise RuntimeError("Couldn't find pixel data, run multi_pass.")

        # first convert to numpy
        self.pxl_ar = np.array([np.array(xi) for xi in self.pxl_lst])

        depth_map_dict = {}

        # ok find unique elements in y array, have those as seperate rows
        # then sort the x values in each row
        unique_y = np.unique(self.pxl_ar[:, 0])
        for y in unique_y:
            depth_map_dict[y] = -1
        # hash the pxl values by their jl values (to be sorted by x later)
        for pxl in self.pxl_ar:
            # if we haven't added an x value there yet, set up the list
            # here we store both the Left image and Rigth image x value
            # so we can calculate the shift dpx later
            if depth_map_dict[pxl[0]] == -1:
                depth_map_dict[pxl[0]] = [[pxl[1], pxl[3]]]
            else:
                depth_map_dict[pxl[0]].append([pxl[1], pxl[3]])

        depth_map_lst = []
        # ok so now we have the x pixel values in their y order, convert the
        # dict to a numpy array sorted by y then x:

        # get length in x
        len_x = max(map(len, depth_map_dict.values()))

        for i, yval in enumerate(sorted(depth_map_dict)):
            sorted_x_vals = sorted(depth_map_dict[yval], key=left_x_val)
            dpx = [x[0]-x[1] for x in sorted_x_vals]
            depth_map_lst.append(dpx)

        avg_shift = np.mean([i for sublst in depth_map_lst for i in sublst])
        # add avg_shift for any missing data
        for row in depth_map_lst:
            if len(row) != len_x:
                row.extend([avg_shift for i in range(len_x-len(row))])

        # self.depth_map = np.array([np.array(row) for row in depth_map_lst])
        self.depth_map = np.array(depth_map_lst)
        fig = plt.figure()
        ax = fig.gca(projection='3d')
        y_vals = np.arange(self.depth_map.shape[0])
        x_vals = np.arange(self.depth_map.shape[1])
        X, Y = np.meshgrid(x_vals, y_vals)
        Z = self.depth_map
        depth = ax.plot_surface(X, Y, Z, cmap=cm.BuPu)
        fig.colorbar(depth, shrink=0.5, aspect=15)
        ax.set_ylabel('y')
        ax.set_xlabel('x')
        ax.set_zlabel('z')
        ax.set_title('dpx depth map')
        ax.elev = -90.
        ax.azim = -90.
        ax.axis('off')

    ###########################################################

    def pickle_result(self, filename='result.p'):
        """
        pickle the real space coordinates array
        """
        if self.result is None:
            raise RuntimeError("Couldn't find real space coords array?")
        path = str(self.data_dir) + '/' + filename
        pickle.dump(self.result, open(path, "wb"))

    ###########################################################

    def load_result(self, filename='result.p'):
        """
        load real space coordinates array
        """
        path = str(self.data_dir) + '/' + filename
        self.result = pickle.load(open(path, "rb"))

    ###########################################################

    def clip_boundaries(self, ybounds=None, xbounds=None, zbounds=None):
        """
        clip the real space coordinates array to the domain inputted, as length
        2 tuples [min, max], or None to not clip on that axis.
        """
        if self.result is None:
            raise RuntimeError("couldn't find real space coords array")

        if ybounds is not None:
            lo, hi = ybounds
            self.result = self.result[self.result[:, 0] <= hi]
            self.pxl_lst = self.result[self.result[:, 0] >= lo]
        if xbounds is not None:
            lo, hi = xbounds
            self.result = self.result[self.result[:, 1] <= hi]
            self.result = self.result[self.result[:, 1] >= lo]
        if zbounds is not None:
            lo, hi = zbounds
            self.result = self.result[self.result[:, 2] <= hi]
            self.result = self.result[self.result[:, 2] >= lo]

    ###########################################################

    def remove_outliers(self, cont=0.1, neighs=10, set_to=None):
        """
        removes outliers via LocalOutlierFactor algorithm from sklearn
        cont := contamination, the fraction expected to be outliers
        neighs := number of neighbours to compare each point to
        set_to := option to set all outliers to a z-value, or None to delete
        """
        if self.result is None:
            raise RuntimeError("couldn't find real space coords array")

        clf = LocalOutlierFactor(n_neighbors=neighs, contamination=cont)
        pred = clf.fit_predict(self.result)
        inliers = (pred == 1)
        outliers = (pred == -1)
        if set_to is not None:
            self.result[outliers, 2] = set_to
        else:
            self.result = self.result[inliers]

    ###########################################################

    def subtract_quad(self, half_width, indep='x', highlight=False):
        """
        subtract off a quartic fit in z(x) or z(y)
        highlight option to compare (graph) of fit vs data points
        """
        if self.result is None:
            raise RuntimeError("couldn't find real space coords array")
        if indep != 'x' and indep != 'y':
            raise RuntimeError("indep input parameter must be 'x' or 'y'")

        if indep == 'x':
            dim = 1
        else:
            dim = 0
        R = self.result[:, dim]         # all x/y values
        Z = self.result[:, 2]           # all Z values

        # Sort values
        s_indices = np.argsort(R, kind='mergesort')
        R_sorted = R[s_indices]
        Z_sorted = Z[s_indices]

        # now only extract unique values of R (and then Z)
        unq_R, un_indices = np.unique(R_sorted, return_index=True)
        unq_Z = Z_sorted[un_indices]

        fit = n_poly.Polynomial.fit(unq_R, unq_Z, 2)
        # find turning point of p_fit
        r_root = fit.deriv().roots()[0]
        max_val = fit(r_root)
        if highlight:
            fig, ax = plt.subplots()
            ax.set_ylabel('z value')
            ax.set_xlabel('x value')
            ax.scatter(unq_R, unq_Z, s=2, c='xkcd:purple')
            ax.plot(unq_R, fit(unq_R), c='xkcd:sky blue')

        # now subtract this p_fit from z in the real space coords array
        # (as a function of the indep. variable inputted)
        with click.progressbar(
                length=self.result.shape[0],
                label="subtract_quad progress, indep var:{}".format(
                    indep)) as bar:
            for i in range(self.result.shape[0]):
                # only subtract off if we're not in the mildly flat region
                # half_width (350) on either side
                if abs(r_root - self.result[i, dim]) > half_width:
                    fit_val = fit(self.result[i, dim])
                    self.result[i, 2] += non_neg(max_val - fit_val)
                bar.update(1)

    ###########################################################

    def plot_real_space(self, zmin=None, zmax=None, scatter=False):
        """
        plot the real space coordinates array as a trisurf and (optionally)
        a scatter plot.
        option for zmin and zmax bounds on axis
        """
        if self.result is None:
            raise RuntimeError("Couldn't find real space coords array?")
        fig = plt.figure()
        ax = fig.gca(projection='3d')
        ax.set_zlabel('z (mm)')
        ax.set_xlabel('x (mm)')
        ax.set_ylabel('y (mm)')
        ax.set_zlim(zmin, zmax)
        ax.elev = -135.
        ax.azim = 145.
        # arguments below are the x, y, z vectors
        ax.plot_trisurf(self.result[:, 1], self.result[:, 0],
                        self.result[:, 2], antialiased=True,
                        cmap=plt.cm.inferno)
        if scatter:
            fig2 = plt.figure()
            ax2 = fig2.gca(projection='3d')
            ax2.set_zlabel('z (mm)')
            ax2.set_xlabel('x (mm)')
            ax2.set_ylabel('y (mm)')
            ax2.set_zlim(zmin, zmax)
            ax2.elev = -135.
            ax2.azim = 145.
            ax2.scatter(self.result[:, 1], self.result[:, 0],
                        self.result[:, 2])

###############################################################################


class SearchGeometry(object):
    """
    Holds the geometry of the 'window' that passes over the images used to
    construct real space plot. t_shape, w_shape, overlap are length-2 tuples
    (y, x). t_shape is defined in the left image, w_shape in the right image.
    w_shape is defined in addition to t_shape, i.e. 'extra' indices on top
    of the t_shape.
    """
    def __init__(self, t_shape, w_shape, overlap):
        if type(t_shape) is not tuple or type(w_shape) is not tuple or \
                type(overlap) is not tuple:
            raise TypeError("Inputs need to be tuples")
        if len(t_shape) != 2 or len(w_shape) != 2 or len(overlap) != 2:
            raise RuntimeError("Inputs need to be length 2 tuples")

        self.t_height, self.t_width = t_shape
        self.t_shape = t_shape
        self.w_shape = w_shape
        self.w_height, self.w_width = w_shape
        # check that the window makes sense

        self.overlap_y, self.overlap_x = overlap
        # check that the overlap makes sense
        if self.overlap_x >= self.t_width or self.overlap_x < 0 or \
                self.overlap_y >= self.t_height or self.overlap_y < 0:
            raise RuntimeError("the input overlaps need to be in the range: " +
                               "[0, width/height)")


###############################################################################
# helper functions

def non_neg(x):
    """ return 0 if x is negative, else return x """
    if x < 0:
        return 0
    else:
        return x


def left_x_val(pxl):
    """ for sorting in plot_depth_map """
    return pxl[0]
