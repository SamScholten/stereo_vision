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
