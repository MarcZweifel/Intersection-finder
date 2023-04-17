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


# Function to sort an array according 2 columns. Column1 will be sorted first and then the array will be sorted again according column2
def sort_array_respectively_2_column(array, column1, column2):
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
    array = np.empty((0, 6))
    for i in range(len(sub_array)):
        array = np.vstack((array, sub_array[i]))
    return array

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

def angle_between_vectors(v1, v2):
    dot_product = v1[0]*v2[0] + v1[1]*v2[1]
    mag_v1 = math.sqrt(v1[0]**2 + v1[1]**2)
    mag_v2 = math.sqrt(v2[0]**2 + v2[1]**2)
    cos_angle = dot_product / (mag_v1 * mag_v2)
    angle_in_radians = math.acos(cos_angle)
    angle_in_degrees = math.degrees(angle_in_radians)
    return angle_in_degrees

def rotate_matrix(matrix, angle):
    rad = math.radians(angle)
    rotation_matrix = np.array([[math.cos(rad), -math.sin(rad)],
                                [math.sin(rad), math.cos(rad)]])
    rotated_matrix = np.dot(matrix, rotation_matrix)
    rotated_positions = rotated_matrix
    return rotated_positions

# Add an interactive option to select a point and find the closest point in the positions array
def onclick(event):
    global closest_x, closest_y
    # Get the x and y coordinates of the point clicked
    x_pos = event.xdata
    y_pos = event.ydata
    if x_pos is not None and y_pos is not None:
        print(f"Selected point: ({x_pos:.2f}, {y_pos:.2f})")

        # Find the index of the closest point in the positions array
        distances = np.sqrt((positions[:, 0] - x_pos) ** 2 + (positions[:, 1] - y_pos) ** 2)
        closest_idx = np.argmin(distances)

        # Get the x and y positions of the closest point
        closest_x = positions[closest_idx, 0]
        closest_y = positions[closest_idx, 1]

        print(f"Closest point: ({closest_x:.2f}, {closest_y:.2f})")

        # Update the values of closest_x and closest_y to use it further in the program
        plt.close(fig)

def onclick_2(event):
    global closest_x, closest_y
    # Get the x and y coordinates of the point clicked
    x_pos = event.xdata
    y_pos = event.ydata
    if x_pos is not None and y_pos is not None:
        print(f"Selected point: ({x_pos:.2f}, {y_pos:.2f})")

        # Find the index of the closest point in the points array
        distances = np.sqrt((points[:, 0] - x_pos) ** 2 + (points[:, 1] - y_pos) ** 2)
        closest_idx = np.argmin(distances)

        # Get the x and y rotated_positions of the closest point
        closest_x = points[closest_idx, 0]
        closest_y = points[closest_idx, 1]

        print(f"Closest point: ({closest_x:.2f}, {closest_y:.2f})")

        # Update the values of closest_x and closest_y to use it further in the program
        plt.close(fig)

#----------------------------------------------------------------------------------------------------------------------
#----------------------------------------------------------------------------------------------------------------------


# Load the array with the ideal, the measured(fitted) and the deviation in x and y
# create a Tkinter root window
root = tk.Tk()
root.withdraw()
# open a file dialog to select a CSV file
file_path = filedialog.askopenfilename(title="Select the array with x and y positions and deviation from the calibration before", filetypes=[('CSV Files', '*.csv')])
# load the selected CSV file into a numpy array
array_first_cal_x_y = np.genfromtxt(file_path, delimiter=';')

# Load the measured positions from a new calibration
file_path = filedialog.askopenfilename(initialdir="~/Desktop", title="Select Select the file with the data of the new measurement after the previous calibration", filetypes=(("CSV files", "*.csv"),))
# Open the file and read each line
with open(file_path, 'r') as csv_file:
    csv_reader = csv.reader(csv_file)
    # Skip the header row if it exists
    next(csv_reader, None)
    # Initialize the position array
    positions = np.empty((0, 2), float)
    for row in csv_reader:
        # Extract the last and second last numbers from the row
        x_pos = float(row[-2])
        y_pos = float(row[-1]) * -1

        # Add the numbers to the position array
        positions = np.append(positions, np.array([[x_pos, y_pos]]), axis=0)


# Create a scatter plot of the positions
fig, ax = plt.subplots()
scatter = ax.scatter(positions[:, 0], positions[:, 1], s=3)
plt.xlabel("X position")
plt.ylabel("Y position")
plt.title("Choose the center point for origin (X¦Y) = (0¦0)")
fig.canvas.mpl_connect('button_press_event', onclick)
# Show the plot
plt.show()

# Number to subtract from each entry in the column
subtract_num_x = closest_x
subtract_num_y = closest_y

# Use a list comprehension to subtract the number from each entry in the column
for i in range(len(positions)):
    positions[i, 0] = positions[i, 0] - subtract_num_x
    positions[i, 1] = positions[i, 1] - subtract_num_y

# Create a scatter plot of the positions
fig, ax = plt.subplots()
scatter = ax.scatter(positions[:, 0], positions[:, 1], s=3)
plt.xlabel("X position")
plt.ylabel("Y position")
plt.title("Choose the point with the highest x value on the x-axis")
# Show the plot
fig.canvas.mpl_connect('button_press_event', onclick)
plt.show()

# Calculate the Angle between the y-axis of the theoretical grid and the real grid
v1 = [10, 0]  # vector with x-component 0 and y-component 10
v2 = [closest_x, closest_y]
angle_X1 = -angle_between_vectors(v1, v2)

# Create a scatter plot of the positions
fig, ax = plt.subplots()
scatter = ax.scatter(positions[:, 0], positions[:, 1], s=3)
plt.xlabel("X position")
plt.ylabel("Y position")
plt.title("Choose the point with the lowest x value on the x-axis")
# Show the plot
fig.canvas.mpl_connect('button_press_event', onclick)
plt.show()

# Calculate the Angle between the y-axis of the theoretical grid and the real grid
v1 = [-10, 0]  # vector with x-component 0 and y-component 10
v2 = [closest_x, closest_y]
angle_X2 = -angle_between_vectors(v1, v2)

#Gabriel
angle_X = -(angle_X2 + angle_X1)/2
rotated_positions = rotate_matrix(positions, angle_X)
#rotated_positions = positions

#np.savetxt(r'C:\Users\maputzer\Desktop\z_Putzer_shared\02 PhD\101_Scan_Field_Calibration\2023_03_16_Kalibration\5_2023_03_16_deviation_x_y_65_Nr5_Joh_rotated positions.csv', rotated_positions, delimiter=';')

# Create a scatter plot of the positions
fig, ax = plt.subplots()
ax.plot([0, closest_x], [0, closest_y], color="red")
ax.plot([-closest_x, closest_x], [0, 0], color="blue")
ax.plot([0, 0], [-closest_x, closest_x], color="blue")
scatter = ax.scatter(positions[:, 0], positions[:, 1], s=3, color="red")
scatter = ax.scatter(rotated_positions[:, 0], rotated_positions[:, 1], s=3, color="green")
plt.xlabel("X position")
plt.ylabel("Y position")
plt.title("Position Scatter Plot")
plt.show()

# Define the distance between x rows and y columns
dx = 2.80932
dy = 2.80932

sorted_indices = np.argsort(rotated_positions[:, 1])
rotated_positions = rotated_positions[sorted_indices]
# Define a tolerance for y-value proximity
tolerance = 1
# Find the differences between consecutive y-values
diffs = np.diff(rotated_positions[:, 1])
# Find the indices where the differences exceed the tolerance
split_indices = np.where(diffs > tolerance)[0] + 1
# Split the original array into subarrays based on the split indices
subarrays = np.split(rotated_positions, split_indices)

row_num = len(subarrays)
column_num = 0
for i in range(len(subarrays)):
    length = len(subarrays[i])
    if length > column_num:
        column_num = length

for i in range(len(subarrays)):
    array = subarrays[i]
    sorted_array = array[array[:,0].argsort()]
    subarrays[i] = sorted_array

deviation_x_y = np.zeros((len(rotated_positions), 6))
a = 0

highest_x = max(rotated_positions[:, 0])
lowest_x = min(rotated_positions[:, 0])
highest_y = max(rotated_positions[:, 1])
lowest_y = min(rotated_positions[:, 1])
test = highest_y + lowest_y
if test > dy/2:
    if (row_num%2) == 0:
        count_num_subarray2 = row_num / 2 - 1
    else:
        count_num_subarray2 = (row_num - 1) / 2 + 1
elif test > dy/2:
    if (row_num%2) == 0:
        count_num_subarray2 = row_num / 2 - 1
    else:
        count_num_subarray2 = (row_num - 1) / 2 - 1
else:
    if (row_num%2) == 0:
        count_num_subarray2 = row_num / 2
    else:
        count_num_subarray2 = (row_num - 1) / 2


for i in range(len(subarrays)):
    subarrays_1 = subarrays[i][:, 0]
    count_num_subarray = len(subarrays_1[subarrays_1 < - dx/2])

    for j in range(len(subarrays[i])):
        x_measured = subarrays[i][j, 0]
        x_ideal = -count_num_subarray * dx + j * dx
        deviation_x = x_ideal - x_measured
        deviation_x_y[a, 0] = deviation_x
        deviation_x_y[a, 2] = x_measured
        deviation_x_y[a, 4] = x_ideal
        y_measured = subarrays[i][j, 1]
        y_ideal = -count_num_subarray2 * dy + i * dy
        deviation_y = y_ideal - y_measured
        deviation_x_y[a, 1] = deviation_y
        deviation_x_y[a, 3] = y_measured
        deviation_x_y[a, 5] = y_ideal
        a = a + 1

# extract columns as separate arrays
dx = deviation_x_y[:, 0] #deviation in x
dy = deviation_x_y[:, 1] #deviation in y
mx = deviation_x_y[:, 2] #measured x
my = deviation_x_y[:, 3] #measured y
ix = deviation_x_y[:, 4] #ideal x
iy = deviation_x_y[:, 5] #ideal y
# create a new figure
fig, ax = plt.subplots()
# create scatter plot of measured points
ax.scatter(mx, my, s=3, label='Measured')
# add scatter plot of ideal points
ax.scatter(ix, iy, s=3, label='Ideal')
# loop over data points and add lines
for i in range(len(deviation_x_y)):
    ax.plot([mx[i], mx[i]+dx[i]], [my[i], my[i]+dy[i]], 'r-', alpha=0.5) # red solid lines
# set plot labels and legend
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.legend()
# show plot
plt.show()

rbf3_y = Rbf(deviation_x_y[:, 4], deviation_x_y[:, 5], dy, function="multiquadric", smooth=0)

xs = np.linspace(-22.474, 22.474, 65) # some extrapolation to negative numbers
ys = np.linspace(-22.474, 22.474, 65) # some extrapolation to negative numbers
xnew_y, ynew_y = np.meshgrid(xs, ys)
xnew_y = xnew_y.flatten()
ynew_y = ynew_y.flatten()
znew_y = rbf3_y(xnew_y, ynew_y)

rbf3_x = Rbf(deviation_x_y[:, 4], deviation_x_y[:, 5], dx, function="multiquadric", smooth=0)

xs = np.linspace(-22.474, 22.474, 65) # some extrapolation to negative numbers
ys = np.linspace(-22.474, 22.474, 65) # some extrapolation to negative numbers
xnew_x, ynew_x = np.meshgrid(xs, ys)
xnew_x = xnew_x.flatten()
ynew_x = ynew_x.flatten()
znew_x = rbf3_x(xnew_x, ynew_x)

# create 3D plot
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(deviation_x_y[:, 4], deviation_x_y[:, 5], dy, color="r", marker='o')
ax.scatter(xnew_y, ynew_y, znew_y, color="b", marker='o')
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
ax.scatter(deviation_x_y[:, 4], deviation_x_y[:, 5], dx, color="r", marker='o')
ax.scatter(xnew_x, ynew_x, znew_x, color="b", marker='o')
# set plot labels and title
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.set_title('3D Plot of Distance in X')
# show plot
plt.show()

deviation_x_y_new_cal = np.zeros((len(znew_x), 6))
for i in range(len(deviation_x_y_new_cal)):
    deviation_x_y_new_cal[i, 4] = xnew_x[i]
    deviation_x_y_new_cal[i, 5] = ynew_x[i]
    deviation_x_y_new_cal[i, 0] = znew_x[i]
    deviation_x_y_new_cal[i, 1] = znew_y[i]
    deviation_x_y_new_cal[i, 2] = xnew_x[i] - znew_x[i]
    deviation_x_y_new_cal[i, 3] = ynew_x[i] - znew_y[i]

deviation_x_y_new_cal = sort_array_respectively_2_column(deviation_x_y_new_cal, 5, 4)
# find indices where column 5 and 6 of array 2 match column 5 and 6 of array 1
ad_zero = np.zeros((len(array_first_cal_x_y), 2))
array_first_cal_x_y = np.hstack((array_first_cal_x_y, ad_zero))

for i in range(len(deviation_x_y_new_cal)):
    deviation_x_y_new_cal[i, 4] = round(deviation_x_y_new_cal[i, 4], 5)
    deviation_x_y_new_cal[i, 5] = round(deviation_x_y_new_cal[i, 5], 5)

for i in range(len(array_first_cal_x_y)):
    array_first_cal_x_y[i, 4] = round(array_first_cal_x_y[i, 4], 5)
    array_first_cal_x_y[i, 5] = round(array_first_cal_x_y[i, 5], 5)

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
for i in range(len(array_first_cal_x_y)):
    new_cal_array_x_y[i, 0] = array_first_cal_x_y[i, 0] + array_first_cal_x_y[i, 6]
    new_cal_array_x_y[i, 1] = array_first_cal_x_y[i, 1] + array_first_cal_x_y[i, 7]
    new_cal_array_x_y[i, 4] = array_first_cal_x_y[i, 4]
    new_cal_array_x_y[i, 5] = array_first_cal_x_y[i, 5]
    new_cal_array_x_y[i, 2] = new_cal_array_x_y[i, 0] - new_cal_array_x_y[i, 4]
    new_cal_array_x_y[i, 3] = new_cal_array_x_y[i, 1] - new_cal_array_x_y[i, 5]

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
