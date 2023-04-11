import imageprocessor as imgprc
import cv2 as cv
import matplotlib.pyplot as plt
from tkinter import filedialog as fd
import sys

file_path = fd.askopenfilename(
    title='Open image file',
    initialdir='/',
    filetypes=[("TIF-file", ".tif .tiff")]
)

if not file_path:
    sys.exit()

relative_file_path = "Example_grids\\20230405_111436_Cleaned_grid_uncalibrated.tif"

img_orig = cv.imread(relative_file_path, cv.IMREAD_GRAYSCALE)
img_orig = imgprc.image(img_orig)
img_scaled = img_orig.scale_image(0.25)
img_copy = img_orig


preseq = imgprc.sequence([
    imgprc.bilateral_filtering(),
    imgprc.binarization(threshold=80)
    ])

preseq.load(img_copy)
preseq.process()

img_bin = preseq.get_result()

gridfndr = imgprc.rough_grid_finding()
gridfndr.load(img_bin)
gridfndr.process()

grid = gridfndr.get_result()
grid = grid.scale_grid(image=img_orig)
grid.select_intersections()


splitter = imgprc.subdividing(field_size=200)
splitter.load(grid)

splitter.process()

subgrid = splitter.get_result()

refine = imgprc.vector_intersection(threshold=0.5)

# Here:
# Origin & Grid coordinates are wrong
# Subgrid image data is empty due to wrong span in the original image -> bug in subdividing.find_split function

for row in range(len(subgrid)):
    for col in range(len(subgrid[row])):
        if subgrid[row, col] is None:
            continue

        preseq.load(subgrid[row, col].image)
        preseq.process()
        subgrid[row, col].image = preseq.get_result()
        
        refine.load(subgrid[row, col])
        refine.process()
        subgrid[row, col] = refine.get_result()

recombine = imgprc.recombining(orig_grid=grid)
recombine.load(subgrid)
recombine.process()

final = recombine.get_result()
final.show_intersections()

