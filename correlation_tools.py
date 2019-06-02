"""
===============================================================================
correlation_tools
- Sam Scholten 2019/04/29
-------------------------------------------------------------------------------

Module with tools for Cross Correlation. The core of the module is kept in the
CorrSystem class, which is initialised pointing to image and data directories
used for reading and writing all information.

The secondary class is the CorrObject. CorrObject is an abstract class that
is implemented in different ways depending on the data source. For example,
a .png and .txt file will need to be read differently, and the subclass
definitions of the CorrObject class allow for that to be integrated nicely.

Further subclasses of the CorrObject allow for more elegant solutions to
complex problems. For example, an IterableCorrObject is defined that contains
mathods for generating different window sizes, and then incrementing that
window over the full array attribute.

On top of the two classes, multiple helper functions are defined seperately
to the object heirachy.

CorrSystem contains the methods where all the actions happens. These include
general methods, such as methods to add a template and search region (both
CorrObjects), where template is 'looked for' in search region. There are also
methods designed for specific problems, such as a dot-finding algorithm.

--------
Objects:
-------------------------------------------------------------------------------
CorrSystem
-------------------------------------------------------------------------------
    Methods:
    --------
        add_template(CorrObject)
        add_search_region(CorrObject)
        spatial_corr(f, g)
        norm_spatial_corr(f, g)
        spectral_corr(t, A)
        cross_correlate(method='spectral' or 'normspatial' or 'spatial')
        max_corr_loc()
        plot_max_corr(corr_filename, search_region_filename)
        find_dots(num_down, num_right, highlight?)
        calculate_detector_sep1ration(speed, freq)
        show_corr()
        save_corr(filename, dpi)

Subclasses:
-----------
    None

-------------------------------------------------------------------------------
CorrObject
-------------------------------------------------------------------------------
    Methods:
    --------
        get_array()
        -> will fail unless implemented in subclass, this Object is only useful
           if you want to manually add via CorrObject.ar = foo

Subclasses:
-----------
    TxtCorrObject:
    -------------
            Methods:
            --------
                __init__(path, parent)
                get_array() - called from init, reads at path in init
                plot_and_save_signal(img_dir, filename)

    PngCorrObject:
    --------------
            Methods:
            --------
                __init__(path, parent)
                get_array() - caled from init, reads at path in init

    TiffCorrObject:
    ---------------
            Methods:
            --------
                __init__(path, parent)
                get_array() - caled from init, reads at path in init

    GaussianCorrObject:
    -------------------
        2D Gaussian array

            Methods:
            --------
                __init__(width, height, parent)
                get_array() - generates a gaussian array from the init params

    IterableCorrObject:
    -------------------
            Methods:
            --------
                __init__(oldCorrObject)
                    - converts oldCorrObject to an IterableCorrObject by
                      copying it's array, and setting the indices to 0
                _next_wind()
                    - passes geom shape over array and returns that window
                _increment()
                    - handles the increment/indices for _next_wind()
                add_geom(geom)
                    - adds a search geometry


----------
Functions:
-------------------------------------------------------------------------------
pad_array(t, A)                     - pads t to A with zeros, assumes 2D
pad_to_shape(ar, new_shape)         - pads array ar to new_shape with zeros
def trim_arrays(coords, t, A)       - trims array t to A, at 'coords' wrt A
-------------------------------------------------------------------------------

===============================================================================
Example
+------------------------------------------------------------------------------
import correlation_tools as corrt
from pathlib import Path

img_dir = Path.cwd()/'images'
data_dir = Path.cwd()/'data'

rocketman_path = str(img_dir/'wallypuzzle_rocketman.png')
puzzle_path = str(img_dir/'wallypuzzle.png')

wally_system = corrt.CorrSystem(img_dir, data_dir)

rocketman = corrt.PngCorrObject(rocketman_path)
puzzle = corrt.PngCorrObject(puzzle_path)

wally_system.add_template(rocketman)
wally_system.add_search_region(puzzle)

wally_system.cross_correlate(method='spectral')
wally_system.show_corr()
wally_system.save_corr(filename='corr', dpi=600)
wally_system.plot_max_corr(corr_filename='max_corr.png',
                           search_region_filename='search_region_max_corr.png')
+------------------------------------------------------------------------------
+------------------------------------------------------------------------------
"""

__author__ = "Sam Scholten"

import numpy as np                      # fundamental numerics/array ops
import matplotlib.pyplot as plt         # plotting/saving etc.
import matplotlib.image as mpimg        # reading RGBA arrays
import cv2                              # tools for drawing on (img) arrays
import click                            # beautiful command line interaction
import pyfftw                           # faster fourier transform in the west
from PIL import Image                   # for reading tiff image
from sklearn.cluster import KMeans      # clustering algorithm

import stereo_camera_tools as camt      # my module

###############################################################################


class CorrSystem(object):
    """
    This object represents a correlation system of a template and a search
    region. The template can be correlated against the search region,
    and then printed/saved.
    """

    def __init__(self, img_dir, data_dir, parent=None):
        self.parent = parent
        self.template = self.search_region = self.corr = self.method = None
        self.corr_fig = self.offset = None
        self.img_dir = img_dir
        self.data_dir = data_dir

    ###########################################################

    def add_template(self, template):
        """ adds a template to the correlation system
        """

        if not isinstance(template, CorrObject):
            raise TypeError('inputted corr_object needs to be a CorrObject')

        if template.ar is None:
            raise RuntimeError('inputted corr_object has no array!')

        self.template = template.ar

    ###########################################################

    def add_search_region(self, search_region):
        """
        adds a search region at the correlation system
        """

        if not isinstance(search_region, CorrObject):
            raise TypeError('inputted corr_object needs to be a CorrObject')

        if search_region.ar is None:
            raise RuntimeError('inputted corr_object has no array!')

        self.search_region = search_region.ar

    ###########################################################

    def spatial_corr(self, f, g):
        """
        computes the spatial correlation (un-normalised) between
        two input arrays (in n dimensions)
        """

        f = f - np.mean(f)
        f = g - np.mean(g)

        if np.ndim(f) != np.ndim(g):
            raise ValueError("input arrays f and g must be the same dimension")

        L = np.size(f)
        return (1/L)*np.sum(f*g)

    ###########################################################

    def norm_spatial_corr(self, f, g):
        """
        computes the normalisted spatial correlation between
        two input arrays (in n dimensions)
        """

        if np.ndim(f) != np.ndim(g):
            raise ValueError("input arrays f and g must be the same dimension")

        f = f - np.mean(f)
        g = g - np.mean(g)

        # sigma = sqrt( 1/n * sum ( ( f(i) - mean(f) )^2 ) )
        # ie standard dev, note all 'n' terms cancel in normed sp. corr
        f_sigma = np.sqrt(np.sum(np.square(f)))
        g_sigma = np.sqrt(np.sum(np.square(g)))
        # check the sigmas, if either is zero (ie a homogenous region found)
        # return an error (will be handled in caller function, which specifies
        # the dimension of the input & thus knows the coordinates this occurs)
        if not f_sigma or not g_sigma:
            raise HomogRegionError('one of the input arrays is homogenous - ' +
                                   'the normalised cross correlation breaks ' +
                                   'down, the standard deviation is zero.')

        return (1/(f_sigma*g_sigma))*np.sum(f*g)

    ###########################################################

    def spectral_corr(self, t, A):
        """
        correlate t and A via the convolution theorem (i.e. by Fourier
        Transform)
        """

        # need to zero pad t to A
        # mean shift the arrays first
        t = t - np.mean(t)
        A = A - np.mean(t)
        t_padded = pad_array(t, A)

        # make a list of the axes' in A
        axes = [i for i in range(np.ndim(A))]

        a = pyfftw.interfaces.numpy_fft.fftn(t_padded, axes=axes, norm='ortho')
        b = pyfftw.interfaces.numpy_fft.fftn(A, axes=axes, norm='ortho')

        a = np.conj(a)
        c = pyfftw.interfaces.numpy_fft.ifftn(a*b, axes=axes,
                                              norm='ortho')
        self.corr = np.real(c)

    ###########################################################

    def cross_correlate(self, method='spectral'):
        """
        produces a correlation array from the correlation objects, with the
        option to use a 'spectral', 'spatial' and 'normspatial' method.
        Default is spectral.
        """

        funcdict = {'spectral': self.spectral_corr,
                    'normspatial': self.norm_spatial_corr,
                    'spatial': self.spatial_corr}

        if method not in funcdict:
            raise KeyError("Optional input parameter 'method' is not an " +
                           "option. Options are: 'spectral', 'spatial' " +
                           "and 'normspatial'")

        if self.template is None or self.search_region is None:
            raise RuntimeError("This Correlation System does not yet have a " +
                               "template and search region to be correlated")

        self.method = method

        homog_regions = 0
        A = self.search_region
        t = self.template

        # check dimension of objects
        dim = np.ndim(A)
        if dim != np.ndim(t):
            raise ValueError(
                        "input arrays A and t must be of the same dimension")

        if dim != 1 and dim != 2:
            raise ValueError("this module only supports 1D and 2D data")

        if (np.array(A.shape) < np.array(t.shape)).any():
            # is this true?
            raise SizeError(
                        "search region (A) must be bigger than template (t)")

        # if method='spectral' then just run it
        # spectral correlation is in 'ndim' anyway, so don't need to worry
        if method == 'spectral':
            self.spectral_corr(self.template, self.search_region)

        elif dim == 1:
            if np.size(A) < np.size(t):
                # I don't think this is strictly true, but for the structure
                # of my methods etc. it is required
                raise ValueError("input array A must be bigger than t")
            t_pts = t.size
            A_pts = A.size
            Corr = np.zeros((t_pts + A_pts - 2))

            with click.progressbar(
                length=(A_pts + t_pts - 2), show_eta=False,
                    label='{} correlation progress'.format(method)) as bar:

                for pt in range(1, t_pts + A_pts - 1):
                    t_over_A, A_under_t = trim_arrays((pt,), t, A)

                    if t_over_A.size == 0 or A_under_t.size == 0:
                        # this shouldn't happen, but I don't want to raise an
                        # error, because it might be expected in some cases
                        click.echo('size=0 trim encountered')

                    try:
                        Corr[pt - 1] = funcdict[method](t_over_A, A_under_t)
                    except HomogRegionError:
                        Corr[pt - 1] = 0
                        homog_regions += 1
                    bar.update(1)

            if homog_regions:

                click.echo(
                    '\n{} homogenous regions found, '.format(homog_regions) +
                    'set to 0 in Corr.')
                click.echo('{} total comparison points\n'.format(
                                t_pts + A_pts - 2))
            self.corr = Corr

        elif dim == 2:
            t_rows, t_cols = t.shape[0], t.shape[1]
            A_rows, A_cols = A.shape[0], A.shape[1]
            # put correlation matrix/heatmap in Corr
            # -2: need to start at 1 overlap and finish overlapping
            Corr = np.zeros((t_rows + A_rows - 2, t_cols + A_cols - 2))

            with click.progressbar(
                length=(A_rows + t_rows - 2), show_eta=False,
                    label='{} correlation progress'.format(method)) as bar:

                # need 'overlap' to start at 1, -> need to subtract 1 from
                # indices in Corr, as it indexes from 0 (obviously)
                for row in range(1, A_rows + t_rows - 1):
                    for col in range(1, A_cols + t_cols - 1):
                        t_over_A, A_under_t = trim_arrays((row, col), t, A)

                        if t_over_A.size == 0 or A_under_t.size == 0:
                            click.echo('size=0 trim encountered')

                        try:
                            Corr[row - 1, col - 1] = funcdict[method](
                                                t_over_A, A_under_t)
                        # ValueError will be encountered for a region with
                        # std dev = 0 ie a homogenous region, this breaks the
                        # normalised cross correlation function, so let's set
                        # the value to zero
                        except ValueError:
                            Corr[row - 1, col - 1] = 0
                            homog_regions += 1
                    bar.update(1)

            if homog_regions:
                # keep track of how many homogenous regions are found

                click.echo(
                    '\n{} homogenous regions found, '.format(homog_regions) +
                    'set to 0 in Corr.')
                click.echo('{} total comparison points\n'.format(float(
                                (A_rows + t_rows - 1)*(A_cols + t_cols - 1))))
            self.corr = Corr

    ###########################################################

    def max_corr_loc(self):
        """
        Returns the maximum correlation *location* in the correlation
        matrix (works for 1D, 2D)
        """
        if self.corr is None:
            raise RuntimeError('no correlation matrix found')
        if self.method is None:
            raise RuntimeError("can't find the correlation method")

        max_corr = np.amax(self.corr)
        max_corr_loc = np.argwhere(self.corr == max_corr)[0]
        return tuple(max_corr_loc)

    ###########################################################

    def plot_max_corr(self, corr_filename='max_corr.png',
                      search_region_filename='search_region_max_corr.png'):
        """
        finds the point of max correlation in corr. Prints out the location
        on the original image, and saves to disk. Also highlights the max
        corr on the (self.) corr matrix
        NB: this function does not draw over the original arrays, it copies
        them and plots/saves independently of other printing/saving methods
        """

        if self.corr is None:
            raise RuntimeError('no correlation matrix found')
        if self.method is None:
            raise RuntimeError("can't find the correlation method")

        corr = self.corr.copy()
        search_region = self.search_region.copy()

        max_corr = np.amax(corr)
        click.echo('max corr: {:.5}'.format(float(max_corr)))
        max_corr_loc = np.where(corr == max_corr)
        click.echo('at: {}'.format(int(max_corr_loc[0])))

        # plot differently depending on if 1D or 2D

        if np.ndim(corr) == 2:
            # where to draw rectangle (depends on correlation method)
            if self.method == 'spectral':
                rect_top_left = (max_corr_loc[1], max_corr_loc[0])
                rect_bottom_right = (max_corr_loc[1] + self.template.shape[1],
                                     max_corr_loc[0] + self.template.shape[0])
            else:
                rect_top_left = (max_corr_loc[1] - self.template.shape[1],
                                 max_corr_loc[0] - self.template.shape[0])
                rect_bottom_right = (max_corr_loc[1], max_corr_loc[0])

            # draw and save on correlation image
            cv2.rectangle(corr, rect_top_left, rect_bottom_right,
                          color=float(max_corr), thickness=5)
            corr_fig, corr_ax = plt.subplots()
            corr_img = corr_ax.imshow(corr)
            corr_fig.colorbar(corr_img)
            # corr_ax.set_title('Correlation matrix, with highest corr\nregion'
            # + ' shown in the rectangle')
            corr_ax.axis('off')

            plt.imsave(str(self.img_dir) + '/' + corr_filename,
                       corr, dpi=600)

            # draw and save on search region image
            cv2.rectangle(search_region, rect_top_left, rect_bottom_right,
                          color=0, thickness=5)
            search_fig, search_ax = plt.subplots()
            search_ax.imshow(search_region, cmap='gray')
            # search_ax.set_title("Search region with highest correlation\n" +
            # "shown in the rectangle")
            search_ax.axis('off')

            plt.imsave(str(self.img_dir) + '/' + search_region_filename,
                       search_region, dpi=600, cmap='gray')
        else:
            # 1D data, don't use a rectangle, use 2 lines
            if self.method == 'spectral':
                left = max_corr_loc[0]
                right = max_corr_loc[0] + self.template.size
                self.offset = left
            else:
                left = max_corr_loc[0] - self.template.size
                right = max_corr_loc[0]
                self.offset = (-1)*left

            # draw and save on correlation plot
            corr_fig, corr_ax = plt.subplots()

            x_vals = np.linspace(1, corr.size, num=corr.size)
            corr_ax.scatter(x_vals, corr, s=2, c='xkcd:purple')
            corr_ax.axvline(x=left, color='g')
            corr_ax.axvline(x=right, color='g')
            # corr_ax.set_title('Correlation function, with region of ' +
            # 'highest\n correlation between the green lines')
            corr_ax.set_xlabel("Displacement between arrays")
            corr_ax.set_ylabel("Correlation coefficient 'r'")
            corr_fig.savefig(str(self.img_dir) + '/' + corr_filename, dpi=600,
                             bbox_inches='tight')

            # draw and save on search region
            search_fig, search_ax = plt.subplots()
            x_vals = np.linspace(1, self.search_region.size,
                                 num=self.search_region.size)
            search_ax.scatter(x_vals, search_region, s=2, c='xkcd:purple')
            search_ax.axvline(x=left, color='g')
            search_ax.axvline(x=right, color='g')
            # search_ax.set_title(
            # "Max correlation region on search_region (green)")
            search_ax.set_xlabel("'time'")
            search_ax.set_ylabel('signal')
            search_fig.savefig(
                        str(self.img_dir) + '/' + search_region_filename,
                        dpi=600, bbox_inches='tight', cmap='gray')

    ###########################################################

    def find_dots(self, num_d, num_r, highlight=False):
        """
        Finds dots in the correlation matrix (returns their pixel locations
        as a numpy array: [ [y1, x1], [y2, x2] ... ])
        Uses K-means algorithm, so must know the number of dots to find
        in advance.
        The 'highlight' option highlights (and shows) the dot locations
        found (for checking)
        """

        if self.corr is None:
            raise RuntimeError('no correlation matrix found')
        if self.method is None:
            raise RuntimeError("can't find the correlation method")

        if np.ndim(self.corr) == 1:
            raise NotImplementedError("Only 2D dots implemented at runtime")

        # copy the corr array as we might want to draw on it without affecting
        # the og copy

        # cut down the 'search' region to the camera size, if it's linked
        # to one
        if self.parent:
            corr = self.corr[:self.parent.camera.height,
                             :self.parent.camera.width].copy()
        else:
            corr = self.corr.copy()
        max_corr = np.amax(corr)

        # note that where searches in the same order as we 'make' the dots on
        # the cal plate (see CalibrationPlate.get_array() in calibration_tools)

        # lower threshold until number of points found is ~ 25 times expected
        # (i.e. 25 pts per dot)
        threshold = 0.99
        num_found = 0

        while num_found <= 25*(num_d*num_r):
            high_corr_y, high_corr_x = np.where(corr >= threshold*max_corr)
            num_found = high_corr_y.shape[0]
            threshold -= 0.01

        # cluster the high correlation points into their centroids

        high_corrs = np.column_stack((high_corr_y,  high_corr_x))

        if highlight:
            click.echo('\ninitially found {} high corrs'.format(
                                                high_corrs.shape[0]))

        # Cluster the points, using sklearn
        # this method taken from: https://mubaris.com/posts/kmeans-clustering/

        kmeans = KMeans(n_clusters=num_d*num_r, n_init=20, max_iter=5000)
        kmeans = kmeans.fit(high_corrs)
        # Centroid values
        centroids = kmeans.cluster_centers_

        # need to move the centroids by half the gaussian window size
        centroids[:, 0] += self.template.shape[0]//2
        centroids[:, 1] += self.template.shape[1]//2

        # sort by y value then x value
        dots = centroids.tolist()
        dots.sort()
        for i in range(0, num_d*num_r, num_r):
            dots[i:i+num_r] = sorted(dots[i:i+num_r], key=lambda x: x[1])
        sub_pixel_coords = np.array(dots)

        if highlight:
            # highlight found locations on an image (search region)

            dot_pxls = np.array(
                [tuple(elem) for elem in sub_pixel_coords.round().astype(
                 int)])
            click.echo("found {} dots".format(dot_pxls.shape[0]))
            f = 1000

            with click.progressbar(
                    dot_pxls, label='drawing dot locations') as bar:

                for coord in bar:
                    j, i = coord
                    self.search_region[j, i] = f

            upd_fig, upd_ax = plt.subplots()
            upd_img = upd_ax.imshow(self.search_region)
            upd_fig.colorbar(upd_img)
            upd_ax.set_title("corr matrix with 'found' dots at {}".format(f))
            upd_ax.axis('off')
            plt.show()

        return sub_pixel_coords

    ###########################################################

    def calculate_detector_separation(self, speed=333, freq=44000):
        """
        In correlation (1D) the offset of the two signals (assuming there
        is one) was stored. Here, taking the signal frequency (Hz, 1/s) and
        the waveform propogation speed (in m/s), we calculate the position
        seperation of the two 'detectors' in m
        """

        click.echo('detector separation: {:.2}m'.format(
                                float((self.offset/freq)*speed)))

    ###########################################################

    def show_corr(self):
        """
        prints the correlation matrix to the screen using pyplot. A scatter
        plot is used for 1D data, an image for 2D data.
        """

        if self.corr is None:
            raise RuntimeError("Tried to show correlation matrix, but the " +
                               "correlation hasn't been run yet")
        self.corr_fig, corr_ax = plt.subplots()

        if np.ndim(self.corr) == 1:
            x_vals = np.linspace(1, self.corr.size, num=self.corr.size)
            corr_ax.scatter(x_vals, self.corr, s=4, c='xkcd:purple')
            corr_ax.set_title('Correlation function')
            corr_ax.set_xlabel('Displacement between arrays')
            corr_ax.set_ylabel("Correlation coefficient 'r'")
        else:
            corr_img = corr_ax.imshow(self.corr)
            self.corr_fig.colorbar(corr_img)
            corr_ax.set_title('Correlation Matrix')
            corr_ax.axis('off')

    ###########################################################

    def save_corr(self, filename='corr', dpi=600):
        """
        save the correlation matrix to disk, doesn't show
        """

        if type(filename) is not str:
            raise TypeError('filename must be a string')

        if self.corr is None:
            raise RuntimeError("Tried to save correlation matrix, but the " +
                               "correlation hasn't been run yet")
        if np.ndim(self.corr) == 1:
            np.savetxt(str(self.data_dir) + '/' + filename + '.txt', self.corr)
            if self.corr_fig is not None:
                self.corr_fig.savefig(str(self.img_dir) + '/' + filename +
                                      '.png', dpi=dpi, bbox_inches='tight')
            else:
                raise RuntimeError("use 'show_corr' first")
        else:
            plt.imsave(str(self.img_dir) + '/' + filename + '.png',
                       self.corr, dpi=dpi)

###############################################################################


class CorrObject(object):
    """
    general class of correlation objects -> not necessarily ever called,
    this is just the 'definition' we can compare against (for example
    checking inputs are all correlation objects in some methods etc.)
    """
    def __init__(self):
        self.ar = None

    ###########################################################

    def get_array():
        raise NotImplementedError("get_array should be overriden by " +
                                  "subclass definition of CorrObject")

###############################################################################


class TxtCorrObject(CorrObject):
    """
    Correlation Object that reads data from a textfile into a numpy array
    """
    def __init__(self, path, parent=None):
        self.parent = parent
        self.path = path
        self.ar = None

        self.get_array()

    ###########################################################

    def get_array(self):
        """
        easy, txt file -> numpy array
        """
        self.ar = np.loadtxt(self.path)

    ###########################################################

    def plot_and_save_signal(self, img_dir, filename):
        """
        numpy array -> scatter plot, save at filename
        """
        if self.ar is None:
            raise RuntimeError('no array found in this object')
        if type(filename) is not str:
            raise TypeError('filename needs to be a str')

        sig_fig, sig_ax = plt.subplots()

        x_vals = np.linspace(1, self.ar.size, num=self.ar.size)
        sig_ax.scatter(x_vals, self.ar, s=2, c='xkcd:purple')
        # sig_ax.set_title(filename)
        sig_ax.set_xlabel("sample index (at 44kHz)")
        sig_ax.set_ylabel('signal amplitude (arb. units)')
        sig_fig.savefig(str(img_dir) + '/' + filename + '.png', dpi=600,
                        bbox_inches='tight')


###############################################################################


class PngCorrObject(CorrObject):
    """
    Correlation Object that reads a numpy array from a png image file
    (converts to greyscale)
    """
    def __init__(self, path, parent=None):
        self.path = path
        self.parent = parent

        self.get_array()

    ###########################################################

    def get_array(self):
        # get RGBA array -> convert to grayscale
        image_ar = mpimg.imread(self.path)

        if np.ndim(image_ar) == 3:
            if image_ar.shape[2] == 4:
                # i.e. an RGBA image, avg. only over RGB vals
                image_ar = np.mean(image_ar[:, :, :3], axis=-1)
            elif image_ar.shape[2] == 3:
                image_ar = np.mean(image_ar, axis=-1)
            else:
                raise RuntimeError('not sure what to do with an image ' +
                                   'inputted with this format')
        # otherwise assume grayscale input image
        self.ar = image_ar

###############################################################################


class TiffCorrObject(CorrObject):
    """
    Correlation Object that reads a numpy array from a tiff image file
    (converts to greyscale)
    """
    def __init__(self, path, parent=None):
        self.path = path
        self.parent = parent

        self.get_array()

    ###########################################################

    def get_array(self):
        image_ar = np.array(Image.open(self.path))

        if np.ndim(image_ar) == 3:
            if image_ar.shape[2] == 4:
                # i.e. an RGBA image, avg. only over RGB vals
                image_ar = np.mean(image_ar[:, :, :3], axis=-1)
            elif image_ar.shape[2] == 3:
                image_ar = np.mean(image_ar, axis=-1)
            else:
                raise RuntimeError('not sure what to do with an image ' +
                                   'inputted with this format')
        # otherwise assume grayscale input image
        self.ar = image_ar


###############################################################################


class GaussianCorrObject(CorrObject):
    """
    create a Gaussian function (2D) object
    """
    def __init__(self, width=0.5, height=250, parent=None):

        self.parent = parent
        self.get_array(width, height)

    ###########################################################

    def get_array(self, width, height):
        x, y = np.meshgrid(np.linspace(-1, 1, 20), np.linspace(-1, 1, 20))
        d = np.sqrt(x*x+y*y)
        sigma, mu = width, 0.0
        g = (height/(
                sigma*np.sqrt(2*np.pi)))*np.exp(-((d-mu)**2/(2.0 * sigma**2)))
        self.ar = g

###############################################################################


class IterableCorrObject(CorrObject):
    """
    Hmmm
    """
    def __init__(self, oldCorrObj, geom, parent=None):
        """
        copy the array from the old correlation object to this one, also
        specifying any parent IterableCorrObjects, and the geometry
        """
        if not isinstance(oldCorrObj, CorrObject):
            raise TypeError("oldCorrObj is not a Correlation Object")
        if not isinstance(geom, camt.SearchGeometry):
            raise TypeError("geom needs to be of type SearchGeometry")
        if parent is not None and not isinstance(parent, IterableCorrObject):
            raise TypeError("parent of IterableCorrObject must be an" +
                            " IterableCorrObject")

        # basically copy the old array
        self.ar = oldCorrObj.ar

        self.geom = geom
        self.height, self.width = self.ar.shape
        self.yi = self.xi = 0                   # iterator indices
        self.parent = parent

    ###########################################################

    def _next_wind(self):
        """
        Track where we are in the 2D array, return a window object (CorrObject)
        that just holds the region we want to look at (this handles the
        'template', not 'search_region' part)
        """

        # all cutting/padding will occur in the correlation fn, this slice
        # will !not! fail with an IndexError (ever)
        new_ar = self.ar[self.yi:self.yi+self.geom.t_height,
                         self.xi:self.xi+self.geom.t_width]

        aryi, arxi = self.yi, self.xi
        # shift indices by any parent window's locations:
        parent = self.parent
        while parent is not None:
            aryi += parent.yi
            arxi += parent.xi
            parent = parent.parent

        # if we found a zero slice, go again
        if 0 in new_ar.shape:
            try:
                self._increment()
            except StopIteration:
                raise StopIteration
            return self._next_wind()

        window = CorrObject()
        window.ar = new_ar
        # increment handles the xi, yi increments
        try:
            self._increment()
        except StopIteration:
            raise StopIteration
        return (aryi, arxi, window)

    ###########################################################

    def _increment(self):
        """
        Increments the _next_wind() indices, raises StopIteration when it gets
        to the end (in y)
        - Increments in x first, then y
        """
        if self.geom is None:
            raise RuntimeError("you haven't specified a search geometry")
        # step across in x
        self.xi += (self.geom.t_width - self.geom.overlap_x)
        # if we're at the edge, step down in y
        if self.xi >= (self.width-self.geom.t_width//2):
            self.xi = 0
            self.yi += (self.geom.t_height - self.geom.overlap_y)
            # we're finished iterating when we're off the bottom
            if self.yi >= (self.height - self.geom.t_height//2):
                self.yi = 0
                self.xi = 0
                raise StopIteration

    ###########################################################

    def get_array(self):
        raise RuntimeError("this shouldn't be called for this subclass")


class SizeError(Exception):
    pass


class HomogRegionError(Exception):
    pass

###############################################################################
# helper functions


def pad_array(t, A):
    """
    zero pad t to the shape of A (currently pads right and down only),
    as this is the form that works most intuitively with spectral_corr
    """
    if np.ndim(A) == 2:
        net_change = np.array(A.shape)[:2] - np.array(t.shape)[:2]
        # left = net_change[1]//2
        # right = net_change[1] - left
        # top = net_change[0]//2
        # bottom = net_change[0] - top
        left = 0
        top = 0
        bottom = net_change[0]
        right = net_change[1]
        pad_width = ((top, bottom), (left, right))

    if np.ndim(A) == 1:
        net_change = A.shape[0] - t.shape[0]
        # left = net_change//2
        # right = net_change - left
        left = 0
        right = net_change
        pad_width = (left, right)

    return np.pad(t, pad_width, 'constant', constant_values=0)

###########################################################


def pad_to_shape(ar, new_shape):
    """
    Zero pads numpy array 'ar' to 'new_shape'. pads down and right (assumes 2d)
    Pads with zeros
    """
    if np.ndim(ar) != 2:
        raise RuntimeError("ar needs to be 2D")
    if type(new_shape) is not tuple or len(new_shape) != 2:
        raise TypeError("requires new_shape to be a tuple of 2 elements")
    change_y = abs(ar.shape[0] - new_shape[0])
    change_x = abs(ar.shape[1] - new_shape[1])
    pad_width = ((0, change_y), (0, change_x))
    return np.pad(ar, pad_width, 'constant', constant_values=0)


###########################################################

def trim_arrays(coords, t, A):
    """
    Takes template (t) and search region (A) both of the same dimension,
    and trims them down only to the overlapping sections, where the bottom
    right of t is at 'coords' with respect to (0, 0) in A.
    """
    dim = np.ndim(A)
    if dim != np.ndim(t):
        raise ValueError("input arrays A and t must be of the same dimension")

    if dim != 1 and dim != 2:
        raise ValueError("this module only supports 1D and 2D data")

    if np.size(A) < np.size(t):
        # is this true?
        raise ValueError("search region (A) must be bigger than template (t)")

    # if dimension = 1, then we only check horizontally
    # if dimension = 2, then need to check vertical overlap
    # we don't care about the last dimension - that's the data.
    # i.e. don't check RGBA values for 3D data
    if dim == 1:
        t_pts = t.shape[0]
        A_pts = A.shape[0]

        max_pt = t_pts + A_pts - 1
        if type(coords) is not tuple or len(coords) != 1:
            raise TypeError(
                "coords input must be a tuple of 1 element (for dim 1 data)")

        pt = coords[0]

        if pt > max_pt:
            raise ValueError(
                "pt out of range: pt={}, max={}".format(pt, max_pt))

        t_over_A = t
        A_under_t = A

        # check horizontal overlap (that's all we've got...)
        if pt < t_pts:
            # left edge of A
            t_over_A = t_over_A[t_pts - pt:]
            A_under_t = A_under_t[:pt]
        elif pt > A_pts:
            # right edge of A
            t_over_A = t_over_A[: A_pts - pt]
            A_under_t = A_under_t[pt - t_pts:]
        else:
            # t is horizontally inside A
            A_under_t = A_under_t[pt - t_pts:pt]

    if dim == 2:

        if type(coords) is not tuple or len(coords) != 2:
            raise TypeError(
                "coords input must be a tuple of 2 elements (for dim 2 data)")

        row, col = coords

        t_rows, t_cols = t.shape[0], t.shape[1]
        A_rows, A_cols = A.shape[0], A.shape[1]

        max_row = t_rows + A_rows - 1
        max_col = t_cols + A_cols - 1

        if row > max_row:
            raise ValueError(
                "row out of range: row={}, max={}".format(row, max_row))
        if col > max_col:
            raise ValueError(
                "col out of range: col={}, max={}".format(col, max_col))

        t_over_A = t
        A_under_t = A

        # check vertical alignment/overlap
        if row < t_rows:
            # top edge of A
            t_over_A = t_over_A[t_rows - row:, :]
            A_under_t = A_under_t[: row, :]
        elif row > A_rows:
            # bottom edge of A
            t_over_A = t_over_A[: row - A_rows, :]
            A_under_t = A_under_t[row - t_rows:, :]
        else:
            # t is vertically inside A
            A_under_t = A_under_t[row - t_rows: row, :]

        # check horizontal alignment/overlap
        if col < t_cols:
            # left edge of A
            t_over_A = t_over_A[:, t_cols - col:]
            A_under_t = A_under_t[:row, :col]

        elif col > A_cols:
            # right edge of A
            t_over_A = t_over_A[:, : col - A_cols]
            A_under_t = A_under_t[:, col - t_cols:]
        else:
            # t is horizontally inside A
            A_under_t = A_under_t[:, col - t_cols: col]

    return t_over_A, A_under_t
