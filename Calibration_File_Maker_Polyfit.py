import csv
import numpy as np
import tkinter as tk
from tkinter import filedialog
import matplotlib.pyplot as plt
import math
from scipy.interpolate import interp2d
from scipy.interpolate import interp1d
from mpl_toolkits.mplot3d import Axes3D
from scipy.interpolate import Rbf

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



# Define the number of rows and columns you choose as calibration field size
row_x = 17
column_y = 17
# Define the distance between x rows and y columns
dx = 2.80932
dy = 2.80932
# Calculate the maximum x and y values
max_x = ((row_x - 1) / 2) * dx
max_y = ((column_y - 1) / 2) * dy
# Create x and y arrays
x = np.linspace(-max_x, max_x, row_x)
y = np.linspace(-max_y, max_y, column_y)
# Create a grid of x and y values
xx, yy = np.meshgrid(x, y)
# Combine the x and y arrays into a single array with two columns
points = np.column_stack((xx.ravel(), yy.ravel()))

x_num_theoretical = 17
y_num_theoretical = 17
x_distance_theoretical = 2.8093
y_distance_theoretical = 2.8093

# Create the root window
root = tk.Tk()
root.withdraw()

# Initialize the closest point variables
closest_x = None
closest_y = None

# Open a file dialog box to choose a file
file_path = filedialog.askopenfilename(initialdir="~/Desktop", title="Select a file", filetypes=(("CSV files", "*.csv"),))


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


# Divide the positions list by Faktor 1000 to get the values in mm
positions = positions

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

angle_X = (angle_X2 + angle_X1)/2

rotated_positions = rotate_matrix(positions, angle_X)

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

# Create a scatter plot of the positions
fig, ax = plt.subplots(figsize=(8, 8))
# Set the aspect ratio of the plot to 1
ax.set_aspect('equal')
scatter = ax.scatter(rotated_positions[:, 0], rotated_positions[:, 1], s=10, color="green")
scatter = ax.scatter(points[:, 0], points[:, 1], s=10, color="red")
plt.xlabel("X position")
plt.ylabel("Y position")
plt.title("Position Scatter Plot")
plt.show()

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
        count_num_subarray2 = row_num / 2 + 1
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


# create 3D plot
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(deviation_x_y[:, 4], deviation_x_y[:, 5], dy, color="r", marker='o')
ax.scatter(deviation_x_y[:, 4], deviation_x_y[:, 5], dx, color="b", marker='o')
# set plot labels and title
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.set_title('3D Plot of Distance in X')
# show plot
plt.show()

rbf3_y = Rbf(deviation_x_y[:, 4], deviation_x_y[:, 5], dy, function="multiquadric", smooth=5)

xs = np.linspace(-22.474, 22.474, 65) # some extrapolation to negative numbers
ys = np.linspace(-22.474, 22.474, 65) # some extrapolation to negative numbers
xnew_y, ynew_y = np.meshgrid(xs, ys)
xnew_y = xnew_y.flatten()
ynew_y = ynew_y.flatten()
znew_y = rbf3_y(xnew_y, ynew_y)

rbf3_x = Rbf(deviation_x_y[:, 4], deviation_x_y[:, 5], dx, function="multiquadric", smooth=5)

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

Faktor = 0.92

deviation_x_y_65 = np.zeros((len(xnew_x), 6))
for i in range(len(deviation_x_y_65)):
    deviation_x_y_65[i, 4] = xnew_x[i]
    deviation_x_y_65[i, 5] = ynew_x[i]
    deviation_x_y_65[i, 0] = Faktor * znew_x[i]
    deviation_x_y_65[i, 1] = Faktor * znew_y[i]
    deviation_x_y_65[i, 2] = xnew_x[i] - Faktor * znew_x[i]
    deviation_x_y_65[i, 3] = ynew_x[i] - Faktor * znew_y[i]

'''
# extract columns as separate arrays
dx = deviation_x_y_65[:, 0] #deviation in x
dy = deviation_x_y_65[:, 1] #deviation in y
mx = deviation_x_y_65[:, 2] #measured x
my = deviation_x_y_65[:, 3] #measured y
ix = deviation_x_y_65[:, 4] #ideal x
iy = deviation_x_y_65[:, 5] #ideal y
# create a new figure
fig, ax = plt.subplots()
# create scatter plot of measured points
ax.scatter(mx, my, s=3, label='Measured')
# add scatter plot of ideal points
ax.scatter(ix, iy, s=3, label='Ideal')
# loop over data points and add lines
for i in range(len(deviation_x_y_65)):
    ax.plot([mx[i], mx[i]+dx[i]], [my[i], my[i]+dy[i]], 'r-', alpha=0.5) # red solid lines
# set plot labels and legend
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.legend()
# show plot
'''
plt.show()

x_cal_max = 22.4739
x_cal_min = -x_cal_max
y_cal_max = x_cal_max
y_cal_min = x_cal_min

sorted_indices = np.argsort(deviation_x_y_65[:, 5])
deviation_x_y_65 = deviation_x_y_65[sorted_indices]
# Define a tolerance for y-value proximity
tolerance = 0.3
# Find the differences between consecutive y-values
diffs = np.diff(deviation_x_y_65[:, 5])
# Find the indices where the differences exceed the tolerance
split_indices = np.where(diffs > tolerance)[0] + 1
# Split the original array into subarrays based on the split indices
subdeviation_x_y_65 = np.split(deviation_x_y_65, split_indices)

for i in range(len(subdeviation_x_y_65)):
    sorted_indices = np.argsort(subdeviation_x_y_65[i][:, 4])
    subdeviation_x_y_65[i] = subdeviation_x_y_65[i][sorted_indices]

deviation_x_y_65 = np.empty((0, 6))
for i in range(len(subdeviation_x_y_65)):
    deviation_x_y_65 = np.vstack((deviation_x_y_65, subdeviation_x_y_65[i]))

# TODO Change to file dialog
np.savetxt(r'C:\Users\marcz\Desktop\2023_04_13_deviation_x_y_after_0.csv', deviation_x_y_65, delimiter=';')

deviation_x_y_65 = 1458 * deviation_x_y_65
iter_y = int(round(math.sqrt(len(deviation_x_y_65))))
iter_x = int(round(math.sqrt(len(deviation_x_y_65))))
with open(r'C:\Users\marcz\Desktop\2023_04_13_Kalibration_1.txt', 'w') as file:
    a = 0
    for i in range(iter_y):
        line = ""
        for j in range(iter_x):
            if j == iter_x:
                ad_to_line = '"' + str(deviation_x_y_65[a, 1]) + ',' + str(deviation_x_y_65[a, 0]) + ',0.00000"'
            else:
                ad_to_line = '"' + str(deviation_x_y_65[a, 1]) + ',' + str(deviation_x_y_65[a, 0]) + ',0.00000",'
            line = line + ad_to_line
            a = a + 1
        file.write(line + '\n')

print("12")
