"""
Author: Sam Scholten
Date: 2019/05/30

Main script accompanying the "Stereo Vision and Pattern Detection" article
published by this author in <paper>.
"""

__author__ = "Sam Scholten"
__version__ = "4.1.2"

###############################################################################

# external modules:
from pathlib import Path            # access to path tools (i.e. cwd)
import time                         # see how long things take (relative only)
import click                        # beautifulk command line control
import matlab.engine                # access matlab functions from python :o
import matplotlib.pyplot as plt     # basic plotting
import numpy as np                  # fast numerics/array module
import quandl                       # read in the data (quandl API)
import pandas as pd                 # data science stuff (dataframe)

# my modules:
import correlation_tools as corrt
import calibration_tools as calbt
import stereo_camera_tools as camt
import pattern_detector_tools_1D as pdt
from quandl_codes import quandl_codes


# ok this got messy fast, but allows us to decide what we want to do...
# to make this prettier, with click integration, this could be quickly chosen
# from the command line!!! https://click.palletsprojects.com/en/7.x/
ALL = False
PART1 = ALL or False
PART2 = ALL or False
PART3 = ALL or False

SENSOR = PART1 or False
ROCKETMAN = PART1 or False
CALIBRATE = PART2 or False
TEST_CAL = PART2 or False
DEPTH_MAP = PART2 or False
MULTIPASS = PART2 or True
ANALYSE_REAL_SPACE = PART2 or True
READ_QSETS = PART3 or False
BUILD_TABLE = PART3 or False
BUILD_DF = PART3 or False
PRINT_COMPS = PART3 or False
AVO_ANALYSE = PART3 or False


def main():
    if CALIBRATE or TEST_CAL or MULTIPASS:
        click.echo("\nStarting MATLAB Engine")
        mlab = matlab.engine.start_matlab()
        click.echo("MATLAB Engine running\n")
    else:
        mlab = None

    img_dir = Path.cwd()/'images'

    data_dir = Path.cwd()/'data'

    # width, height, image directory, data directory, matlab engine reference
    camera = camt.Camera(1801, 2400, img_dir, data_dir, mlab)

    cal_path_dict = {
            1900: ("cal_imgs/cal_image_left_1900.tiff",
                   "cal_imgs/cal_image_right_1900.tiff"),
            1920: ("cal_imgs/cal_image_left_1920.tiff",
                   "cal_imgs/cal_image_right_1920.tiff"),
            1940: ("cal_imgs/cal_image_left_1940.tiff",
                   "cal_imgs/cal_image_right_1940.tiff"),
            1960: ("cal_imgs/cal_image_left_1960.tiff",
                   "cal_imgs/cal_image_right_1960.tiff"),
            1980: ("cal_imgs/cal_image_left_1980.tiff",
                   "cal_imgs/cal_image_right_1980.tiff"),
            2000: ("cal_imgs/cal_image_left_2000.tiff",
                   "cal_imgs/cal_image_right_2000.tiff"),
                    }

    if SENSOR:

        click.echo('\nbegin operation SENSOR\n')

        start = time.perf_counter()
        sensor1_path = str(data_dir/'sensor1_data.txt')
        sensor2_path = str(data_dir/'sensor2_data.txt')

        sensor_system = corrt.CorrSystem(img_dir, data_dir)

        sensor1 = corrt.TxtCorrObject(sensor1_path)
        sensor1.plot_and_save_signal(img_dir, 'Sensor 1')
        sensor2 = corrt.TxtCorrObject(sensor2_path)
        sensor2.plot_and_save_signal(img_dir, 'Sensor 2')

        sensor_system.add_template(sensor1)
        sensor_system.add_search_region(sensor2)

        sensor_system.cross_correlate(method='spectral')
        sensor_system.show_corr()
        sensor_system.save_corr()
        sensor_system.plot_max_corr()
        sensor_system.calculate_detector_separation(freq=44000, speed=333)

        rel_time = time.perf_counter() - start

        click.echo('time taken for SENSOR with method={} is {:.5}s'.format(
                                    sensor_system.method, rel_time))
        plt.show()

    if ROCKETMAN:
        click.echo('\nbegin operation ROCKETMAN\n')

        start = time.perf_counter()
        rocketman_path = str(img_dir/'wallypuzzle_rocketman.png')
        puzzle_path = str(img_dir/'wallypuzzle.png')

        wally_system = corrt.CorrSystem(img_dir, data_dir)
        rocketman = corrt.PngCorrObject(rocketman_path)
        puzzle = corrt.PngCorrObject(puzzle_path)

        wally_system.add_template(rocketman)
        wally_system.add_search_region(puzzle)

        wally_system.cross_correlate(method='spectral')
        wally_system.show_corr()
        wally_system.save_corr()
        wally_system.plot_max_corr()

        rel_time = time.perf_counter() - start
        click.echo('time taken for ROCKETMAN with method={} is {:.5}s'.format(
                                wally_system.method, rel_time))
        plt.show()

    if DEPTH_MAP:

        click.echo('\nbegin operation DEPTH_MAP')
        # non-calibration images path, a string
        nci_path = str(img_dir) + '/non_cal_imgs/'
        camera.add_left_tiff(nci_path + 'left_box.tiff')
        camera.add_right_tiff(nci_path + 'right_box.tiff')
        fst_pass = camt.SearchGeometry((30, 30), (0, 60), (25, 25))
        camera.multi_pass([fst_pass, ], conv_to_real=False)
        camera.plot_depth_map()
        plt.show()

    if CALIBRATE:

        click.echo('\nbegin operation CALIBRATE\n')
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

        plt.show()

    if TEST_CAL:
        # check and see if the calibration curve makes physical sense
        click.echo('\nbegin operation TEST_CAL')

        camera.add_polymodel("polymodel3.p", method="pickled")
        polymodel = camera.calibration

        # order of args: (jl, il, jr, ir)
        i = 500.0
        j = matlab.double([i, i, i, i])

        ans = camera.mlab.polyvaln(polymodel['x'], j)
        ans2 = camera.mlab.polyvaln(polymodel['y'], j)
        ans3 = camera.mlab.polyvaln(polymodel['z'], j)
        print(np.array(ans), np.array(ans2), np.array(ans3))

    if MULTIPASS:

        click.echo('\nbegin operation MULTIPASS')

        camera.add_polymodel("polymodel3.p", method="pickled")

        # non-calibration images path, a string
        nci_path = str(img_dir) + '/non_cal_imgs/'

        camera.add_left_tiff(nci_path + 'test_left_2.tiff')

        camera.add_right_tiff(nci_path + 'test_right_2.tiff')

        # template shape, window shape (+template...), overlaps
        fst_pass = camt.SearchGeometry((90, 90), (180, 180), (60, 60))
        sec_pass = camt.SearchGeometry((30, 30), (60, 60), (15, 15))

        camera.multi_pass([fst_pass, sec_pass], conv_to_real=True)

        camera.pickle_result('test2_p3.p')

    if ANALYSE_REAL_SPACE:

        click.echo('\nbegin operation ANALYSE')

        camera.load_result('test2_p3.p')
        # camera.plot_real_space(zmin=0, zmax=2200)
        camera.plot_real_space(zmin=1800, zmax=2200)
        # clip to rough z region
        camera.clip_boundaries(zbounds=(1800, 2100))
        # remove 5% of pts
        camera.remove_outliers(cont=0.05, neighs=25, set_to=None)

        # subtract off a quadratic in z(x) (if applicable)
        camera.subtract_quad(350, indep='x', highlight=True)
        camera.plot_real_space(zmin=1800, zmax=2200)

        # clip to x and y (and a lil extra z)
        # y range=800, x range=1000
        camera.clip_boundaries(ybounds=(-200, 1000),
                               xbounds=(-200, 1200),
                               zbounds=(1850, 2050))
        # harsher clipping:
        # camera.clip_boundaries(ybounds=(0, 800),
        #                        xbounds=(0, 1000),
        #                        zbounds=(1850, 2050))

        # remove 1%
        camera.remove_outliers(cont=0.05, neighs=25, set_to=None)
        camera.plot_real_space(zmin=1800, zmax=2200)

        # plot up the final solution, with scatter plots
        camera.plot_real_space(zmin=0, zmax=2200, scatter=True)
        camera.plot_real_space(zmin=1800, zmax=2200, scatter=True)

    if READ_QSETS or BUILD_TABLE or PRINT_COMPS or BUILD_DF or AVO_ANALYSE:
        dset = pdt.DataSet(quandl_codes())

    if READ_QSETS:
        quandl.ApiConfig.api_key = "w5oiYdfN-bvm2n1n6StN"
        dset.get_qsets()
        dset.get_time_series()
        dset.pickle_time_series()

    if BUILD_TABLE:
        dset.load_time_series()
        title = "Finding Patterns in Quandl data sets for Australia " + \
            "via Correlation and Cross Correlation Methods."
        dset.build_table()
        dset.print_table(sortby="Correlation r value",
                         reversesort=True, title=title)

    if BUILD_DF:
        dset.load_time_series()
        dset.build_dataframe()
        dset.print_dataframe()

    if PRINT_COMPS:
        # compare TimeSeries with each other on the same axis
        if not BUILD_TABLE or not BUILD_DF:
            dset.load_time_series()

        # TODO Sam pick 3 good examples with important attributes

        # comp1 = ["CURRENT ACCOUNT; N.I.E.",
        #          "BALANCE ON GOODS; SERV. & INC."]
        # dset.plot_ts("ts_comp", comp1)
        # comp2 = ["Export Quantity (tonnes)",
        #          "Primary Female Enrollment"]
        # dset.plot_ts("ts_comp2", comp2)
        # comp3 = ["Adult labour force ('000)",
        #          "Export Value (1000$)"]
        # dset.plot_ts("ts_comp3", comp3)
        # comp4 = ["Youth population ('000)",
        #          "6. Wholesale and Retail Trade and Restaurants and Hotels"]
        # dset.plot_ts("ts_comp4", comp4)
        # comp5 = ["Men: Total Coverage: All Unemployed",
        #          "7. Transport; Storage and Communication"]
        # dset.plot_ts("ts_comp5", comp5)
        # comp6 = ["K. Real Estate; Renting and Business Activities",
        #          "4. Electricity; Gas and Water"]
        # dset.plot_ts("RE_vs_Elec", comp6)
        # comp7 = ["Adult unemployment rate (%)",
        #         "Wool; degreased not carbonized; not carded or combed " +
        #         " (Metric tons)"]
        # dset.plot_ts("unemp_vs_wool", comp7)

    if AVO_ANALYSE:
        if not BUILD_DF:
            dset.load_time_series()
            dset.build_dataframe()
        # avocado example

        # so I think this is pretty neat, with a pandas dataframe we can
        # extract only the rows we want - so in this case we can simply search
        # for just avocado related data, and cross-examine that with
        # real_estate data.

        avoframe1 = dset.dataframe[
                            dset.dataframe["B's Quandl Set"].str.contains(
                                "avocado", case=False)]
        avoframe2 = dset.dataframe[
                            dset.dataframe["A's Quandl Set"].str.contains(
                                "avocado", case=False)]
        with open('avos.txt', "w") as text_file:
            text_file.write(avoframe1.to_string() + '\n' +
                            avoframe2.to_string())

        meme1 = avoframe1[avoframe1["A's Quandl Set"].str.contains(
                                "real_estate", case=False)]

        meme2 = avoframe2[avoframe2["B's Quandl Set"].str.contains(
                                "real_estate", case=False)]
        meme = pd.concat([meme1, meme2], ignore_index=True)
        # print(meme)
        with open('meme.txt', "w") as text_file:
            text_file.write(meme.to_string())

    plt.show()

    click.echo('\nfinished')
    return 0


if __name__ == '__main__':
    main()
