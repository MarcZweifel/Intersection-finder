import cv2 as cv
import matplotlib.pyplot as plt
from tkinter import filedialog as fd
import numpy as np
import sys
import csv
import imageprocessor as imgprc
import exporter

# ******************************************************************
# Program variables:
# ******************************************************************

# Image properties
# ******************************************************************
# Color mode can be "color" or "gray"
mode = "color"

# Resolution in pixel/mm
resolution = 480.9

# Binarization parameters:
# If mode is "gray only the red channel (r) is used.
# ******************************************************************
# Color channel (RGB) thresholds:
r_threshold = 230 #40
g_threshold = 230 #60
b_threshold = 230 #60

# Color channel (RGB) binarization inverted:
# True: pixel value > threshold => 0
# False: pixel value < threshold => 0
r_is_inverted = False # True
g_is_inverted = False # True
b_is_inverted = False # True

# Rough intersection finding parameter:
# ******************************************************************
line_thickness = 80 # um
rough_summation_field_width = 1 # mm    Width over which the pixels are summed up horizontally and vertically. Approximate recommendation: line distance
rough_peak_threshold = 0.8              # Between 0 and 1

# Image subdivision:
# ******************************************************************
local_image_size = 1.66 # mm            Fixed value or if None split in the middle of two rough points.

# Fine intersection finding parameters:
# ******************************************************************
fine_peak_threshold = 0.6               # Between 0 and 1
fine_summation_field_width = 0.4 # mm   Width over which the pixels are summed up at the edges of the local image. Approximate recommendation: 1/8 to 1/4 of local image size

# ******************************************************************


# File dialoge for choosing the image
file_path = fd.askopenfilename(
    title="Open image file",
    initialdir="~/Desktop",
    filetypes=[("TIF-file", ".tif .tiff")]
)

# Terminate program if no file chosen
if not file_path:
    sys.exit()

# Import image in the chosen color mode
if mode == "gray":
    img_orig = cv.imread(file_path, cv.IMREAD_GRAYSCALE)
elif mode == "color":
    # color image is converted from bgr to rgb
    img_orig = cv.cvtColor(cv.imread(file_path), cv.COLOR_BGR2RGB) 
else:
    # Error if mode string is wrong
    raise ValueError("Choose 'gray' or 'color'.") 

# Save in custom image class
img_orig = imgprc.image(img_orig, resolution=resolution)

# Copy image for binarization
img_bin = img_orig

# Preprocessing sequence (At the moment only binarization)
preseq = imgprc.sequence([
    imgprc.binarization(
        threshold=[r_threshold, g_threshold, b_threshold],
        inverted=[r_is_inverted, g_is_inverted, b_is_inverted],
        mode=mode)
    ])
preseq.load(img_bin)
preseq.process()
img_bin = preseq.get_result()

# Check preprocessing result
print("Set the binarization parameters to achieve lines with sharp edges and little noise between lines in the binary image.")
# img_bin.show_image(title="Are the binarization parameters satisfactory? Close window to accept")

# Import previous points
file_path = fd.askopenfilename(
    title="Open previously saved points",
    initialdir="~/Desktop",
    filetypes=[("CSV-file", ".csv")]
)
# User chose file:
if file_path:
    rough_grid_finding_flag = False
    grid = imgprc.grid(image=img_bin)
    grid.import_points(file_path)

# User hit cancel
else:
    rough_grid_finding_flag = True

    # Roughly find approximate intersection points
    gridfndr = imgprc.rough_grid_finding(
        field_size=rough_summation_field_width,
        line_thickness=line_thickness,
        threshold=rough_peak_threshold)
    gridfndr.load(img_bin)
    gridfndr.process()
    grid = gridfndr.get_result()

# User selects intersection points for refinement
grid.select_intersections(title="Select the points you want to refine. Close window to accept.")
grid = grid.scale_grid(image=img_orig)

# Splitting the image into local images containing the intersections
splitter = imgprc.subdividing(field_size=local_image_size)
splitter.load(grid)
splitter.process()
subgrid = splitter.get_result()

# Refine the intersection points by looping over all local images
refine = imgprc.vector_intersection(
    threshold=fine_peak_threshold,
    field_size=fine_summation_field_width,
    line_thickness=line_thickness)

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

# Recombine local images to a global grid
recombine = imgprc.recombining(orig_grid=grid)
recombine.load(subgrid)
recombine.process()
final = recombine.get_result()

# User selects / deselects points for export
final.select_intersections(title="Check refined points before export. Are they on the intersections? Deselect them if not.", color_mode="color", standard_selection_mode="Deselect")
if rough_grid_finding_flag:
    final.pick_zero(color_mode="color", title="Pick the zero point of the grid.")
final.show_intersections(title="Check points again before export. Close window to export.")
final.export()




# TODO Zero point picker for imported points. Include zero point information in grid class & export

# TODO Punkte im 2. Bild verschieben & setzen können: grösserer Aufwand (Event handling) - Auf Eis gelegt

# TODO Schnittstelle zum Skript von Matthias. (Siehe Teams Ordner) - Fast fertig: Noch Koordinatensystem anpassen

# TODO Rotationswinkel der Punkte pos. oder neg. in Matthias (Gabriel Kommentar) automatisch finden.
# TODO Formatierung der mechanischen Kalibrationsfiles erarbeiten.