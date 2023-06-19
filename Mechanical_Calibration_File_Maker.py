import csv
import numpy as np
from tkinter import filedialog
import matplotlib.pyplot as plt
import math
from scipy.interpolate import interp2d
from scipy.interpolate import interp1d
from mpl_toolkits.mplot3d import Axes3D
from scipy import interpolate
from scipy.interpolate import Rbf
import sys

from imageprocessor import grid

def read_calibration_parameters():
    file_path = filedialog.askopenfilename(
        initialdir="~/Desktop",
        title="Select the file containing the calibration configuration of the setup.",
        filetypes=[('Text files', '*.txt')])

    if not file_path:
        return {}
    result = {}
    with open(file_path, "r") as variable_file:
        
        for idx, line in enumerate(variable_file):
            line = line.strip("\n")
            if not line:
                continue
            line = line.split("=", maxsplit=1)
            name = line[0].strip()

            value_string = line[1].split(";", maxsplit=1)
            value = value_string[0].strip()
            dtype = value_string[1].strip()

            if dtype.lower() == "float":
                value = float(value)
            elif dtype.lower() == "int":
                value = int(value)
            elif dtype.lower() == "bool":
                value = bool(int(value))
            else:
                print(f"Datatype in row {idx+1} is invalid!")
                continue

            result[name] = value
    return result

def create_ideal_grid(meas_grid, dx, dy):
    mask = meas_grid.get_mask()
    num_x = meas_grid.grid_size.x
    num_y = meas_grid.grid_size.y
    zero_index = meas_grid.zero_index
    x_pos = np.linspace(0, (num_x-1)*dx, num_x)
    x_pos -= x_pos[zero_index[1]]
    y_pos = np.linspace((num_y-1)*dy, 0, num_y)
    y_pos -= y_pos[zero_index[0]]
    return grid(grid_lines=[x_pos, y_pos], mask=mask, zero_index=zero_index)

def aerotech_exporter():#deviations, dlx, dly):
    file_path = filedialog.asksaveasfilename(
        title="Export aerotech calibration file",
        initialdir="~/Desktop",
        filetypes=[("Aerotech mechanical calibration file", ".cal")]
        )
    if not file_path:
        return
    if file_path[-4:]!=".cal":
            file_path = file_path + ".cal"
    
    dlx = 2.5
    dly = 2.5
    nx = 15
    
    with open(file_path, "w") as cal_file:
        cal_file.write("'        RowAxis ColumnAxis OutputAxis1 OutputAxis2 SampDistRow SampDistCol NumCols\n")
        cal_file.write(f":START2D    2         1          1           2          {dly}          {dlx}      {nx}\n")
        cal_file.write(":START2D POSUNIT=PRIMARY CORUNIT=PRIMARY OFFSETROW=0.0 OFFSETCOL=0.0\n")
        cal_file.write("Here go the correction values.\n")
        cal_file.write(":END\n")
        cal_file.write("'\n'\n' Notes:\n' X-axis is axis 1, Y-axis is axis 2.\n' All distances are in mm.\n")
        cal_file.write("' Correction values outside of the calibration table are clipped to the outermost correction value.")

#----------------------------------------------------------------------------------------------------------------------
#----------------------------------------------------------------------------------------------------------------------

system_configuration = read_calibration_parameters()

if not system_configuration:
    sys.exit()

# open a file dialog to select a CSV file
file_path = filedialog.askopenfilename(initialdir="~/Desktop" ,title="Select the array with x and y corrections from the calibration before", filetypes=[('CSV Files', '*.csv')])
if file_path:
    prev_cal_dev = grid()
    prev_cal_dev.import_points(file_path, mm_to_pixel=False, switch_y_direction=False)
    first_calibration_flag = False
else:
    first_calibration_flag = True

# Load the measured positions from a new calibration
file_path = filedialog.askopenfilename(initialdir="~/Desktop", title="Select Select the file with the data of the new measurement after the previous calibration", filetypes=(("CSV files", "*.csv"),))

# Terminate program if no file chosen
if not file_path:
    sys.exit()

meas_grid = grid()
meas_grid.import_points(file_path, mm_to_pixel=False, switch_y_direction=True)
meas_grid.move_origin_to_zero()

# TODO Integrate functionality in A3200 program to export a file with the offset distance, line distance, axis numbers and positive directions.
# TODO Change inter- / extrapolation to suit mechanical calibration procedure.

# Define the distance between x rows and y columns

dx = float(input("Input the ideal line spacing of the measured grid in X-direction in mm.\n"))
dy = float(input("Input the ideal line spacing of the measured grid in Y-direction in mm.\n"))

ideal_grid = create_ideal_grid(meas_grid, dx, dy)

fig, ax = plt.subplots()
fig.set_size_inches(9, 6)
ideal_grid.show_intersections(axis=ax, color="c")
meas_grid.select_intersections(title="Deselect intersections to exclude from rotation correction.", axis=ax, standard_selection_mode="Deselect")

# Rotate grid
meas_grid.rotate_grid_to_x()

fig, ax = plt.subplots()
fig.set_size_inches(9, 6)
ideal_grid.show_intersections(plot_axes=True, axis=ax, color="c")
meas_grid.select_intersections(title="Deselect the intersections you want to exclude from interpolation", axis=ax, standard_selection_mode="Deselect")

ideal_grid.create_mask(mask=meas_grid.get_mask())

mx = meas_grid.get_active()[:,0]
my = meas_grid.get_active()[:,1]
ix = ideal_grid.get_active()[:,0]
iy = ideal_grid.get_active()[:,1]
dx = ix - mx
dy = iy - my

fig, ax = plt.subplots()
meas_grid.show_intersections(axis=ax, color="r", title="Measured (red) and ideal (blue) points", label="Measured")
ideal_grid.show_intersections(axis=ax, label="Ideal")
for i, j, k, l in zip(mx, dx, my, dy):
    ax.plot([i, i+j], [k, k+l], 'r-', alpha=0.5) # red solid lines
# set plot labels and legend
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.legend()
plt.show()
ideal_grid.create_mask(mask=meas_grid.get_mask())

smoothing = float(input("Input the smoothing factor for the interpolation:\n"))

rbf3_y = Rbf(ix, iy, dy, function="multiquadric", smooth=smoothing)
rbf3_x = Rbf(ix, iy, dx, function="multiquadric", smooth=smoothing)

counts_per_unit = int(input("Input counts per unit of the scanner:\n"))

xs = np.linspace(-32768/counts_per_unit, 32767/counts_per_unit, 65) # some extrapolation to negative numbers // used to be +/- 22.474
ys = np.linspace(32767/counts_per_unit, -32768/counts_per_unit, 65) # some extrapolation to negative numbers
cal_grid = grid(grid_lines=[xs, ys], mask=False, zero_index=[(65-1)//2, (65-1)//2])
cal_deviations = cal_grid.evaluate_for_active(func1=rbf3_x, func2=rbf3_y)

points = cal_grid.get_active()[:,(0,1)]
xnew, ynew = points[:,0], points[:,1]
znew_y = cal_deviations.get_active()[:,1]
znew_x = cal_deviations.get_active()[:,0]

# create 3D plot
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(ix, iy, dy, color="r", marker='o')
ax.scatter(xnew, ynew, znew_y, color="b", marker='o')
# set plot labels and title
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.set_title('3D Plot of Distance in Y')
# show plot
plt.show()

# create 3D plot
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(ix, iy, dx, color="r", marker='o')
ax.scatter(xnew, ynew, znew_x, color="b", marker='o')
# set plot labels and title
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.set_title('3D Plot of Distance in X')
# show plot
plt.show()

if not first_calibration_flag:
    cal_deviations.intersections[:,:,:] += prev_cal_dev.intersections[:,:,:]

cal_deviations.export()
aerotech_exporter(cal_deviations)
