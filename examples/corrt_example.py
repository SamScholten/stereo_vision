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
