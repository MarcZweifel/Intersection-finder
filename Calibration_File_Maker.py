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

def aerotech_exporter(deviations, counts_per_unit):
    file_path = filedialog.asksaveasfilename(
        title="Export aerotech calibration file",
        initialdir="~/Desktop",
        filetypes=[("Comma separated values", ".csv")]
        )
    if not file_path:
        return
    if file_path[-4:]!=".csv":
            file_path = file_path + ".csv"
    
    with open(file_path, "w", newline="") as csv_file:
        csv_writer = csv.writer(csv_file, lineterminator="\n")
        
        for i in range(deviations.grid_size.y):
            line = []
            for j in range(deviations.grid_size.x):
                dx = round(deviations.intersections[0,i,j] * counts_per_unit, 5)
                dy = round(deviations.intersections[1,i,j] * counts_per_unit, 5)
                line.append(f"{dy:.5f},{dx:.5f},0.00000") if j<deviations.grid_size.x-1 else line.append(f"{dy:.5f},{dx:.5f},0.00000")
            csv_writer.writerow(line)

#----------------------------------------------------------------------------------------------------------------------
#----------------------------------------------------------------------------------------------------------------------

# open a file dialog to select a CSV file
file_path = filedialog.askopenfilename(title="Select the array with x and y positions and deviation from the calibration before", filetypes=[('CSV Files', '*.csv')])
if file_path:
    prev_cal_dev = grid()
    prev_cal_dev.import_points(file_path, mm_to_pixel=False)
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

fig, ax = plt.subplots()

meas_grid.show_intersections(plot_axes=True, axis=ax, color="r")
meas_grid.rotate_grid_to_x()
meas_grid.show_intersections(plot_axes=True, title="Imported (red) & rotated (blue) points", axis=ax)

plt.show()

# Define the distance between x rows and y columns

dx = input("Input the line spacing in X-direction.")
dy = input("Input the line spacing in Y-direction.")
# dx = 2.80932
# dy = 2.80932

ideal_grid = create_ideal_grid(meas_grid, dx, dy)

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

rbf3_y = Rbf(ix, iy, dy, function="multiquadric", smooth=0)
rbf3_x = Rbf(ix, iy, dx, function="multiquadric", smooth=0)

xs = np.linspace(-22.474, 22.474, 65) # some extrapolation to negative numbers
ys = np.linspace(-22.474, 22.474, 65) # some extrapolation to negative numbers
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

factor = 0.92

if first_calibration_flag:
    cal_deviations = cal_deviations.scale_grid(factor=factor)

else:
    cal_deviations.intersections[:,:,:] += prev_cal_dev.intersections[:,:,:]

cal_deviations.export()
aerotech_exporter(cal_deviations, 1458)
