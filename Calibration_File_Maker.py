import csv
import numpy as np
import tkinter as tk
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

# Function to sort an array according 2 columns. Column1 will be sorted first and then the array will be sorted again according column2
def sort_array_respectively_2_column_subarray(array, column1, column2):
    sorted_indices = np.argsort(array[:, column1])
    array = array[sorted_indices]
    # Define a tolerance for y-value proximity
    tolerance = 0.3
    # Find the differences between consecutive y-values
    diffs = np.diff(array[:, column1])
    # Find the indices where the differences exceed the tolerance
    split_indices = np.where(diffs > tolerance)[0] + 1
    # Split the original array into subarrays based on the split indices
    sub_array = np.split(array, split_indices)
    for i in range(len(sub_array)):
        sorted_indices = np.argsort(sub_array[i][:, column2])
        sub_array[i] = sub_array[i][sorted_indices]
    return sub_array

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
# TODO Replace by input() function
dx = 2.80932
dy = 2.80932

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
    # TODO Previous deviations in grid
    # dev_new = dev_old + dev_measured
    cal_deviations.intersections[:,:,:] += prev_cal_dev.intersections[:,:,:]

cal_deviations.export()
aerotech_exporter(cal_deviations, 1458)




deviation_x_y_new_cal = np.zeros((len(znew_x), 6))
for i in range(len(deviation_x_y_new_cal)):
    deviation_x_y_new_cal[i, 4] = xnew_x[i] # ideal x
    deviation_x_y_new_cal[i, 5] = ynew_x[i] # ideal y
    deviation_x_y_new_cal[i, 0] = znew_x[i] if not first_calibration_flag else faktor*znew_x[i] # dev x
    deviation_x_y_new_cal[i, 1] = znew_y[i] if not first_calibration_flag else faktor*znew_y[i] # dev y
    deviation_x_y_new_cal[i, 2] = xnew_x[i] - deviation_x_y_new_cal[i, 0] # meas x
    deviation_x_y_new_cal[i, 3] = ynew_x[i] - deviation_x_y_new_cal[i, 1] # meas y

deviation_x_y_new_cal = sort_array_respectively_2_column(deviation_x_y_new_cal, 5, 4)
# find indices where column 5 and 6 of array 2 match column 5 and 6 of array 1
ad_zero = np.zeros((len(array_first_cal_x_y), 2))
array_first_cal_x_y = np.hstack((array_first_cal_x_y, ad_zero))

for i in range(len(deviation_x_y_new_cal)):
    deviation_x_y_new_cal[i, 4] = round(deviation_x_y_new_cal[i, 4], 5)
    deviation_x_y_new_cal[i, 5] = round(deviation_x_y_new_cal[i, 5], 5)

# extract columns as separate arrays
# dx = deviation_x_y[:, 0] #deviation in x
# dy = deviation_x_y[:, 1] #deviation in y
# mx = deviation_x_y[:, 2] #measured x
# my = deviation_x_y[:, 3] #measured y
# ix = deviation_x_y[:, 4] #ideal x
# iy = deviation_x_y[:, 5] #ideal y

# dx = ix - mx
# dy = iy - my

if not first_calibration_flag:
    for i in range(len(array_first_cal_x_y)):
        array_first_cal_x_y[i, 4] = round(array_first_cal_x_y[i, 4], 5) # ideal x
        array_first_cal_x_y[i, 5] = round(array_first_cal_x_y[i, 5], 5) # ideal y

    b = 0
    a = 0
    for i in range(len(array_first_cal_x_y)):
        if a == len(deviation_x_y_new_cal[:, 5]):
            a = len(deviation_x_y_new_cal[:, 5]) - 1
        if array_first_cal_x_y[i, 5] == deviation_x_y_new_cal[a, 5]:
            if array_first_cal_x_y[i, 4] == deviation_x_y_new_cal[a, 4]:
                array_first_cal_x_y[i, 6] = deviation_x_y_new_cal[a, 0]
                array_first_cal_x_y[i, 7] = deviation_x_y_new_cal[a, 1]
                a = a + 1
                b = a

    new_cal_array_x_y = np.zeros((len(array_first_cal_x_y), 6))

else:
    new_cal_array_x_y = np.zeros((len(xnew_x), 6))

for i in range(len(new_cal_array_x_y)):
    new_cal_array_x_y[i, 0] = array_first_cal_x_y[i, 0] + array_first_cal_x_y[i, 6] # dev x
    new_cal_array_x_y[i, 1] = array_first_cal_x_y[i, 1] + array_first_cal_x_y[i, 7] # dev y
    new_cal_array_x_y[i, 4] = array_first_cal_x_y[i, 4] # ideal x
    new_cal_array_x_y[i, 5] = array_first_cal_x_y[i, 5] # ideal y
    new_cal_array_x_y[i, 2] = new_cal_array_x_y[i, 0] - new_cal_array_x_y[i, 4] # meas x
    new_cal_array_x_y[i, 3] = new_cal_array_x_y[i, 1] - new_cal_array_x_y[i, 5] # meas y

# extract columns as separate arrays
dx = new_cal_array_x_y[:, 0]  # deviation in x
dy = new_cal_array_x_y[:, 1]  # deviation in y
mx = new_cal_array_x_y[:, 2]  # measured x
my = new_cal_array_x_y[:, 3]  # measured y
ix = new_cal_array_x_y[:, 4]  # ideal x
iy = new_cal_array_x_y[:, 5]  # ideal y
# create a new figure


fig, ax = plt.subplots()
# create scatter plot of measured points
ax.scatter(mx, my, s=3, label='Measured')
# add scatter plot of ideal points
ax.scatter(ix, iy, s=3, label='Ideal')
# loop over data points and add lines
for i in range(len(new_cal_array_x_y)):
    ax.plot([mx[i], mx[i]-dx[i]], [my[i], my[i]-dy[i]], 'k-', alpha=0.5) # red solid lines
# set plot labels and legend
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.legend()
# show plot
plt.show()


np.savetxt(
    r'C:\Users\maputzer\Desktop\z_Putzer_shared\02 PhD\101_Scan_Field_Calibration\2023_04_06_Check\5_deviation_x_y_65_Nr6.csv',
    new_cal_array_x_y, delimiter=';')

new_cal_array_x_y = 1458 * new_cal_array_x_y
iter_y = int(round(math.sqrt(len(new_cal_array_x_y))))
iter_x = int(round(math.sqrt(len(new_cal_array_x_y))))
with open(
        r'C:\Users\maputzer\Desktop\z_Putzer_shared\02 PhD\101_Scan_Field_Calibration\2023_04_06_Check\5_Kalibration_65_Nr6.txt',
        'w') as file:
    a = 0
    for i in range(iter_y):
        line = ""
        for j in range(iter_x):
            if j == iter_x:
                ad_to_line = '"' + str(new_cal_array_x_y[a, 1]) + ',' + str(new_cal_array_x_y[a, 0]) + ',0.00000"'
            else:
                ad_to_line = '"' + str(new_cal_array_x_y[a, 1]) + ',' + str(new_cal_array_x_y[a, 0]) + ',0.00000",'
            line = line + ad_to_line
            a = a + 1
        file.write(line + '\n')

print("12")
