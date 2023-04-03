import imageprocessor as imgprc
import cv2 as cv
import matplotlib.pyplot as plt

img_orig = cv.imread("Example_grids\\2023_03_14_#01_calibration_after0.tif", cv.IMREAD_GRAYSCALE)
print(type(img_orig))
img_orig = imgprc.image(img_orig)
img_scaled = img_orig.scale_image(0.25)



seq = imgprc.sequence([
    imgprc.bilateral_filtering(),
    imgprc.binarization(threshold=90),
    imgprc.rough_grid_finding()
    ])

seq.load_image(img_scaled)
seq.process_sequence()
grid = seq.get_result()
grid = grid.scale_grid(image=img_orig)

#grid.show_intersections()

splitter = imgprc.subdividing(field_size=800)
splitter.load_image(grid)

splitter.process_image()

subimages = splitter.get_result()

subimages[5, 6].show_intersections()



