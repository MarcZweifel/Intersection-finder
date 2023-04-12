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

mode = "color"

if mode == "gray":
    img_orig = cv.imread(file_path, cv.IMREAD_GRAYSCALE)

elif mode == "color":
    img_orig = cv.imread(file_path)

else:
    raise ValueError("Choose 'gray' or 'color'.")

img_orig = imgprc.image(img_orig)
img_scaled = img_orig.scale_image(0.25)
img_copy = img_orig

preseq = imgprc.sequence([
    #imgprc.bilateral_filtering(),
    imgprc.binarization(threshold=[80, 180, 180], inverted=[True, False, False], mode=mode)
    ])

preseq.load(img_copy)
preseq.process()

img_bin = preseq.get_result()
#img_bin.show_image()

gridfndr = imgprc.rough_grid_finding(line_thickness=70)
gridfndr.load(img_bin)
gridfndr.process()

grid = gridfndr.get_result()
grid = grid.scale_grid(image=img_orig)
grid.select_intersections()


splitter = imgprc.subdividing()#field_size=400)
splitter.load(grid)

splitter.process()

subgrid = splitter.get_result()

refine = imgprc.vector_intersection(threshold=0.2, field_size=400, line_thickness=70)

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

# TODO Punkte abwählen in beiden Bildern
# TODO Punkte im 2. Bild verschieben & setzen können
# TODO Konversion zu mm mithilfe der Auflösung
# TODO Schnittstelle zum Skript von Matthias. (Siehe Teams Ordner)
# TODO Rotationswinkel der Punkte pos. oder neg. in Matthias (Gabriel Kommentar) automatisch finden.