import matlab.engine
import stereo_vision_tools as camt
from pathlib import Path

img_dir = Path.cwd()/'images'
data_dir = Path.cwd()/'data'
mlab = matlab.engine.start_matlab()

# get the camera ready
camera = camt.Camera(1801, 2400, img_dir, data_dir, mlab)
camera.add_polymodel("polymodel3.p", method="pickled")
camera.add_left_tiff(<LEFT_IMG_PATH>)
camera.add_right_tiff(<RIGHT_IMG_PATH>)

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
