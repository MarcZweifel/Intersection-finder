import imageprocessor as imgprc
import cv2 as cv
import matplotlib.pyplot as plt

img_orig = cv.imread("Example_grids\\2023_03_14_#01_calibration_after0.tif", cv.IMREAD_GRAYSCALE)
img_orig = imgprc.image(img_orig)
img_scaled = img_orig.scale_image(0.25)



preseq = imgprc.sequence([
    imgprc.bilateral_filtering(),
    imgprc.binarization(threshold=90)
    ])

preseq.load(img_scaled)
preseq.process()

img_bin = preseq.get_result()

gridfndr = imgprc.rough_grid_finding()
gridfndr.load(img_bin)
gridfndr.process()

grid = gridfndr.get_result()
grid = grid.scale_grid(image=img_orig)

#grid.show_intersections()

splitter = imgprc.subdividing(field_size=800)
splitter.load(grid)

splitter.process()

subgrid = splitter.get_result()[5,6]

subgrid.show_intersections()

subimage = subgrid.image

preseq.load(subimage)
preseq.process()
subgrid.image = preseq.get_result()
refine = imgprc.vector_intersection()
refine.load(subgrid)

refine.process()
grid_fine = refine.get_result()
grid_fine.show_intersections()