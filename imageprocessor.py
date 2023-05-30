import cv2 as cv
import matplotlib.pyplot as plt
import matplotlib.widgets as widgets
from matplotlib.collections import PathCollection
import numpy as np
from tkinter import filedialog
import csv
from scipy.optimize import least_squares
import sys

# plt.rcParams["font.family"] = "Neue Haas Grotesk Text Pro"

debug_mode = False # Shows summation peaks during line finding

class image_size():
    """Class for storing image size in pixels."""

    def __init__(self, x_in, y_in):
        self.x = x_in
        self.y = y_in

class image():
    """Class for storing images including its size, resolution and relative scale to the original image.
    image_data: np.array
    scale: float
    resolution: float"""

    def __init__(self, image_data, scale=1, resolution=1):
        self.image_data = image_data
        self.scale = scale
        x_in = self.image_data.shape[1]
        y_in = self.image_data.shape[0]
        self.image_size = image_size(x_in, y_in)
        self.resolution = resolution
    
    def scale_image(self,factor):
        """Return a scaled by a factor copy of image instance."""

        image_data = cv.resize(self.image_data, None, fx=factor, fy=factor)
        scale = factor*self.scale
        resolution = self.resolution*factor
        return image(image_data, scale, resolution=resolution)
        
    def show_image(self, mode="gray", title = None):
        """Show the image in the specified color mode with a title."""

        if mode == "gray":
            plt.imshow(self.image_data, cmap="gray")
        elif mode == "color":
            plt.imshow(self.image_data)
        else:
            raise ValueError("mode must be 'gray' or 'color'.")
        
        if title is not None:
            plt.title(title)
        plt.show()
            
class grid():
    """Class for storing a grid, its intersection points, origin and corresponding image. Intersection points can be masked to not be considered by intersection finding algorithm.
    grid_lines: list of pixel coordinate lists of horizontal & vertical lines. Intersections are calculated automatically.
    intersections: (2 x n x m)-np.array (meshgrid format) with n intersections in y and m intersections in x-direction.
    image: corresponding image
    origin: origin of the grid (upper left corner of image)
    mask: (n+2 x m+2)-np.array filled with bools: True = ignore point; False = Consider point"""

    def __init__(self, grid_lines=None, intersections=None, image=None, origin=None, mask=None, zero_index=None, resolution=None):
        
        if origin is None:
            origin = [0, 0]
        
        if resolution is None:
            resolution = 1

        if intersections is not None:
            self.intersections = intersections
            self.intersections[0,:,:] -= origin[0]
            self.intersections[1,:,:] -= origin[1]

        elif grid_lines is not None:
            # Calculate grid data relative to origin
            grid_data = [[j-origin[i] for j in grid_lines[i]] for i in [0,1]]

            # Calculate meshgrid of intersections
            self.intersections = np.array(np.meshgrid(grid_data[0], grid_data[1]))
        
        else:
            self.intersections = np.empty((2,0,0))  

        # Save number of gridlines as image_size
        row_size = self.intersections.shape[1]
        col_size = self.intersections.shape[2]
        self.grid_size = image_size(col_size, row_size)

        # Create mask
        self.create_mask(mask)

        # Save corresponding image & origin
        self.image = image
        if self.image is not None:
            self.scale = self.image.scale
            self.resolution = self.image.resolution
        else:
            self.scale = 1
            self.resolution = resolution
        self.origin = origin

        # Grid zero point information
        self.zero_index = zero_index
        if zero_index is not None:
            self.zero = self.intersections[:, zero_index[0], zero_index[1]]
    
    def create_mask(self, mask=None, shape=None):
        shape = (self.intersections.shape[1]+2, self.intersections.shape[2]+2)
        if mask is not None:
            if type(mask) is bool:
                self.mask = np.full(shape, True)
                self.mask[1:-1, 1:-1] = mask
            else:
                self.mask = np.full((mask.shape[0]+2, mask.shape[1]+2), True)
                self.mask[1:-1, 1:-1] = mask[:,:]
        else:
            self.mask = np.full(shape, True)

    def select_intersections(self, color_mode="gray", title=None, standard_selection_mode="Select", axis=None):
        """Function for setting the intersection mask from user selection."""

        # Initialize figure & plot axis
        if axis is None:
            fig, ax = plt.subplots()
            fig.set_size_inches(9, 6)
        
        else:
            ax = axis
            fig = ax.figure

        # Define selection handler instance
        selector = point_selector(fig.canvas, self, ax, mode=standard_selection_mode)
        
        # Set plot title
        if title is not None:
            ax.set_title(title)
        
        # Plot image in the specified color mode only if it exists
        if self.image is not None:
            if color_mode == "gray":
                ax.imshow(self.image.image_data, cmap="gray")
            elif color_mode == "color":
                ax.imshow(self.image.image_data)
            else:
                raise ValueError("mode must be 'gray' or 'color'.")

        # Callback function for handling checkbox behavior
        def checkbox_callback(label):
            status = checkboxes.get_status()

            if label == "Select":
                if status == [False, False]:
                    #Select was deactivated, deselect is inactive
                    checkboxes.set_active(1)
                    selector.set_mode(label)
                elif status == [True, True]:
                    #Select was activated, deselect is active
                    checkboxes.set_active(1)
                    selector.set_mode(label)
            elif label == "Deselect":
                if status == [False, False]:
                    #Deselect was deactivated, select is inactive
                    checkboxes.set_active(0)
                    selector.set_mode(label)
                elif status == [True, True]:
                    #Deselect was activated, select is active
                    checkboxes.set_active(0)
                    selector.set_mode(label)
        
        # List for checkbox initialization
        actives = [standard_selection_mode=="Select", standard_selection_mode=="Deselect"]

        # Initialize Checkbox
        checkbox_axis = fig.add_axes([0.05, 0.4, 0.1, 0.15])
        checkboxes = widgets.CheckButtons(checkbox_axis, ["Select", "Deselect"], actives=actives)
        checkboxes.on_clicked(checkbox_callback)
        
        plt.show()        
        
    def show_intersections(self, mode="gray", title=None, plot_axes=False, axis=None, color=None, label=None):
        """Show all unmasked intersection with corresponding image in the specified color mode. Set the plot title."""
        if axis is None:
            fig, ax = plt.subplots()
        
        else:
            ax = axis
        
        # Plot image in the color mode only if it exists
        if self.image is not None:
            if mode == "gray":
                ax.imshow(self.image.image_data, cmap="gray")
            elif mode == "color":
                ax.imshow(self.image.image_data)
            else:
                raise ValueError("mode must be 'gray' or 'color'.")

        # Set title
        if title is not None:
            plt.title(title)

        color = color if color is not None else "b"

        # Extract unmasked points & plot them
        xv = self.intersections[0, :,:]
        xv = xv[np.logical_not(self.mask[1:-1, 1:-1])]
        yv = self.intersections[1, :,:]
        yv = yv[np.logical_not(self.mask[1:-1, 1:-1])]
        if label is not None:
            ax.plot(xv, yv, marker=".", color=color, linestyle="none", label=label)
        else:
            ax.plot(xv, yv, marker=".", color=color, linestyle="none")
        if plot_axes:
            ax.grid("on")

        if axis is None:
            plt.show()

    def scale_grid(self, factor=1, image=None):
        """Returns a scaled copy of the grid instance. The scaling is either determined by a specified factor or by the specified image so the grid fits the same image."""

        if image is None:
            # Scale by specified factor
            if self.image is not None:
                image = self.image.scale_image(factor)
            
        else:
            # Fit onto image (standard behavior)
            factor = image.scale/self.scale
        
        # Scale grid and origin
        intersections = self.intersections[:,:,:]*factor
        origin_scaled = [c*factor for c in self.origin]
        resolution = self.resolution*factor

        return grid(intersections=intersections, image=image, origin=origin_scaled, mask=self.mask[1:-1,1:-1], resolution=resolution, zero_index=self.zero_index)

    def activate(self, coords):
        """Activate intersections at specified row-column-coordinates. (Not fully implemented & used!)
        coords: (n x 2)-np.array, row-index in first, column-index in second column"""

        for row in coords[:,0]:
            for col in coords[:,1]:
                self.mask[row+1][col+1] = False
    
    def copy(self):
        """Return a copy of the grid instance."""

        copy = grid(intersections=self.intersections, image=self.image, origin=self.origin)
        return copy

    def get_active(self):
        """Return (n x 4)-np.array containing x-y- and row-column-coordinates of unmasked intersections."""

        mask = self.mask[1:-1, 1:-1]
        result = self.intersections[:,np.logical_not(mask)]
        rows = np.array([i[0] for i in np.ndindex(mask.shape) if not mask[i]])
        columns = np.array([i[1] for i in np.ndindex(mask.shape) if not mask[i]])
        return np.stack((result[0], result[1], rows, columns), axis=1)
    
    def points_as_list(self):
        """Return all intersection row-column-coordinates as an (n x 2)-np.array."""
        xv = self.intersections[0,:,:].flatten()
        yv = self.intersections[1,:,:].flatten()
        rows = np.array([i[0] for i in np.ndindex(self.intersections.shape[1:])])
        columns = np.array([i[1] for i in np.ndindex(self.intersections.shape[1:])])
        return np.stack((xv, yv, rows, columns), axis=1)

    def export(self):
        """Export the unmasked intersections coordinates to a csv-file. The coordinates are in mm relative to the upper left corner of the image."""

        resolution = self.resolution

        # File dialoge for user selection of the file path
        file_path = filedialog.asksaveasfilename(
            title="Export points",
            initialdir="~/Desktop",
            filetypes=[("Comma separated values", ".csv")]    
        )

        # User hit cancel
        if not file_path:
            return

        # Append .csv to the path to not get a generic file
        if file_path[-4:]!=".csv":
            file_path = file_path + ".csv"

        # Write data to file in required format
        with open(file_path, "w", newline="\n") as csv_file:
            csv_writer = csv.writer(csv_file)
            csv_writer.writerow([str(self.resolution)])
            csv_writer.writerow([
                "row", "column", "zero point", "Min", "Max", "X", "Y"])
            points = self.get_active()
            
            filler = np.zeros_like(points[:,0])
            
            zero_point = np.logical_and(
                np.equal(points[:,2], self.zero_index[0]),
                np.equal(points[:,3], self.zero_index[1]))

            output = np.stack((points[:,2], points[:,3], zero_point, filler, filler, points[:,0]/resolution, points[:,1]/resolution), axis=1)
            csv_writer.writerows(output)

    def import_points(self, file_path, mm_to_pixel=True, switch_y_direction=False):
        with open(file_path, "r", newline="\n") as csv_file:
            csv_reader = csv.reader(csv_file)
            # Read resolution
            resolution = float(next(csv_reader, None)[0])
            # Skip header
            next(csv_reader, None)

            # Read points and its indices in the grid in csv-rows
            points = np.empty((0,2), float)
            indices = np.empty((0,2), int)
            for row in csv_reader:
                if mm_to_pixel:
                    xpoint = float(row[5])*resolution
                    ypoint = float(row[6])*resolution
                else:
                    xpoint = float(row[5])
                    ypoint = float(row[6])
                    resolution = 1

                if switch_y_direction:
                    ypoint = -ypoint

                points = np.append(points, np.array([[xpoint, ypoint]]), axis=0)
                indices = np.append(indices, np.array([row[0:2]], dtype=float).astype(int), axis=0)
                if bool(int(float(row[2]))):
                    zero_index = np.array(row[0:2], dtype=float).astype(int)
        
        # Use minimal amount of masked points
        row_min = indices[:,0].min()
        row_max = indices[:,0].max()
        col_min = indices[:,1].min()
        col_max = indices[:,1].max()
        shape = (2 ,row_max-row_min+1, col_max-col_min+1)
        mask = np.full(shape[1:], True)
        intersections = np.full(shape, np.NaN)
        for point, [row, column] in zip(points, indices):
            intersections[:,row-row_min, column-col_min] = point
            mask[row-row_min, column-col_min] = False
        zero_index[0] -= row_min
        zero_index[1] -= col_min
        self.__init__(intersections=intersections, image=self.image, mask=mask, zero_index=zero_index, resolution=resolution)
        if self.image is not None:
            self.place_zero(title="Place the zero point of the grid on the image.")

    def pick_zero(self, color_mode="gray", title=None):
        """Function to pick the zero point of the grid from existing intersection points."""
        
        def on_pick(event):
            # Get the x and y coordinates of the point clicked
            x_pos = event.mouseevent.xdata
            y_pos = event.mouseevent.ydata

            points = self.get_active()[:,(0,1)]
            indices = self.get_active()[:,(2,3)].astype(int)

            if x_pos is not None and y_pos is not None:
                #print(f"Selected point: ({x_pos:.2f}, {y_pos:.2f})")

                # Find the index of the closest point in the positions array
                distances = np.sqrt((points[:, 0] - x_pos) ** 2 + (points[:, 1] - y_pos) ** 2)
                closest_list_idx = np.argmin(distances)
                self.zero_index = indices[closest_list_idx,:]
                self.zero = points[closest_list_idx,:]
                print(f"Zero point at ({self.zero[0]}, {self.zero[1]}) with index ({self.zero_index[0]}, {self.zero_index[1]})")

        # Cancel if no intersection points exist
        if self.intersections is None:
            return
        
        self.compress()

        fig, ax = plt.subplots()
        fig.set_size_inches(9, 6)
        
        # Set plot title
        if title is not None:
            ax.set_title(title)
        
        # Plot image in the specified color mode only if it exists
        if self.image is not None:
            if color_mode == "gray":
                ax.imshow(self.image.image_data, cmap="gray")
            elif color_mode == "color":
                ax.imshow(self.image.image_data)
            else:
                raise ValueError("mode must be 'gray' or 'color'.")
        
        points = self.get_active()[:,(0,1)]
        indices = self.get_active()[:,(2,3)]

        # points = self.get_active()[:,(0,1)]
        # indices = self.get_active()[:,(2,3)]

        ax.scatter(points[:,0], points[:,1], marker=".", color="b", picker=True)
        fig.canvas.mpl_connect("pick_event", on_pick)

        plt.show()

    def place_zero(self, title=None, color_mode="gray"):
        """Function to place the zero point of the grid to the position where the mouse click happened."""

        def on_click(event):
            canvas = event.canvas
            tool_mode = str(canvas.toolbar.mode)

            if tool_mode != "":
                return

            x_pos = event.xdata
            y_pos = event.ydata
            ax = event.inaxes
            fig = canvas.figure
            
            self.zero = [x_pos, y_pos]
            if self.zero_index is not None:
                # Only if zero_index exists
                # Calculate zero_point deviation from click position
                delta = self.zero[:] - self.intersections[:,self.zero_index[0], self.zero_index[1]]
                # Move all intersection points
                self.intersections[0,:,:] += delta[0]
                self.intersections[1,:,:] += delta[1]

                # TODO get plotting to work
                points = self.get_active()[:,0:2]
                point_artist = ax.scatter(points[:,0], points[:,1], marker=".", color="b", animated=True)
                canvas.draw()

                print(f"Zero point at ({self.intersections[0,self.zero_index[0], self.zero_index[1]]}, {self.intersections[1,self.zero_index[0], self.zero_index[1]]})")


        fig, ax = plt.subplots()
        fig.set_size_inches(9, 6)

        # Set plot title
        if title is not None:
            ax.set_title(title)
        
        # Plot image in the specified color mode only if it exists
        if self.image is not None:
            if color_mode == "gray":
                ax.imshow(self.image.image_data, cmap="gray")
            elif color_mode == "color":
                ax.imshow(self.image.image_data)
            else:
                raise ValueError("mode must be 'gray' or 'color'.")
            
        

        fig.canvas.mpl_connect("button_press_event", on_click)
        plt.show()

    def compress(self):
        mask = self.mask[1:-1, 1:-1]
        rows_keep = np.count_nonzero(mask, axis=1)
        rows_keep = np.flatnonzero(np.logical_not(rows_keep == self.intersections.shape[2]))
        columns_keep = np.count_nonzero(mask, axis=0)
        columns_keep = np.flatnonzero(np.logical_not(columns_keep == self.intersections.shape[1]))
        self.intersections = self.intersections[:,rows_keep, :]
        self.intersections = self.intersections[:, :, columns_keep]
        mask = mask[rows_keep,:]
        mask = mask[:, columns_keep]
        self.create_mask(mask)
    
    def move_origin_to_zero(self):
        self.origin[:] = self.zero[:]
        self.intersections[0,:,:] -= self.zero[0]
        self.intersections[1,:,:] -= self.zero[1]

    def get_x_axis_points(self):
        mask = self.get_mask()[self.zero_index[0],:]
        return  self.intersections[:,self.zero_index[0],np.logical_not(mask)]

    def rotate_grid_to_x(self):
        def fun(params, x, y):
            return params[0]*x + params[1] - y
        
        params_init = [0,0]
        points = self.get_x_axis_points()
        x = points[0]
        y = points[1]
        result = least_squares(fun, x0=params_init, args=(x,y))
        rad = np.arctan(result.x[0])

        rotation_matrix = np.array([
            [np.cos(rad), np.sin(rad)],
            [-np.sin(rad), np.cos(rad)]])
        
        for [row, column] in np.ndindex(self.intersections.shape[1:]):
            vector = self.intersections[:,row,column]
            self.intersections[:,row, column] = np.dot(rotation_matrix, vector)

    def set_grid_from_list(self, list):
        pass

    def get_mask(self):
        return self.mask[1:-1, 1:-1]

    def evaluate_for_active(self, func1, func2=None):
        mask = self.get_mask()
        intersections = self.intersections.copy()
        for i, j in np.ndindex((self.grid_size.y, self.grid_size.x)):
            x, y = intersections[:,i,j]
            if func2 is None:
                intersections[:,i,j] = func1(x, y) if not mask[i,j] else intersections[:,i,j]
            else:
                intersections[0,i,j] = func1(x, y) if not mask[i,j] else x
                intersections[1,i,j] = func2(x, y) if not mask[i,j] else y
        
        return grid(intersections=intersections, mask=mask, zero_index=self.zero_index)

class point_selector():
    """Class for handling the intersection selection process.
    canvas: matplotlib.FigureCanvas of the whole figure
    grid: custom grid class
    plot_axis: matplotlib.Axes where the image is plotted to
    mode: 'Select' or 'Deselect' initial selection mode"""

    def __init__(self, canvas, grid, plot_axis, mode = None):
        self.canvas = canvas
        self.figure = self.canvas.figure

        # Define event connection handles
        self._press_id = self.canvas.mpl_connect(
            "button_press_event",
            self.on_press
        )
        self._release_id = self.canvas.mpl_connect(
            "button_release_event",
            self.on_release
        )
        self._draw_id = self.canvas.mpl_connect(
            "draw_event",
            self.on_draw
        )
        self._select_id = None
        self._leave_id = None

        self.grid = grid

        # Mouse input data
        self.start = None # Start position in figure coordinates
        self.start_data = None # Start position in plot coordinates
        self.finish_data = None # End position in plot coordinates

        # Axis handles
        self.plot_axis = plot_axis # Axis to be plotted in
        self.event_axes = None # Axis where mouse event started

        self.selected = [] # list of selected points

        self.mode = mode

        self.points = grid.points_as_list()
        self.point_colors = None
        
        
        # Initialize point colors based on grid mask
        self.set_colors()

        # Define point plot artist for handling drawing commands. Use blitting
        self.point_plot = self.plot_axis.scatter(self.points[:,0], self.points[:,1], c=self.point_colors, marker=".", animated=True)
        
        # Set up legend as help for the user. Don't draw the placeholder points.
        placeholder_point_r = self.plot_axis.scatter([0],[0], c="r", animated=True)
        placeholder_point_b = self.plot_axis.scatter([0],[0], c="b", animated=True)
        self.figure.legend(
            [placeholder_point_r, placeholder_point_b],
            ["Unselected", "Selected"],
            loc="right")

    def on_draw(self, event):
        """Drawing event callback function to draw points when entire figure is redrawn."""

        self.background = self.canvas.copy_from_bbox(self.figure.bbox)
        self.draw_points()

    def on_axis_leave(self, event):
        """Callback function when mouse leaves plotting axis to avoid errors."""

        # Clip finish data to last mouse coordinates that where inside the plotting axis.
        if self.event_axes[0].in_axes(event):
            self.finish_data = (event.xdata, event.ydata)

    def on_press(self, event):
        """Callback function for mouse press event. Start selection process."""

        # Check if matplotlib navigation tools are deactivated.
        tool_mode = str(self.canvas.toolbar.mode)
        if tool_mode!="":
            return
        
        # Terminate if not left mouse button pressed or mouse is outside the figure canvas.
        if (event.button != 1
                or event.x is None or event.y is None):
            return
        
        # Selection mode has to be specified
        if self.mode is None:
            return

        # Connect a mouse drag event
        self._select_id = self.canvas.mpl_connect(
            "motion_notify_event",
            self.on_drag
        )

        # Connect a axis leave event
        self._leave_id = self.canvas.mpl_connect(
            "axes_leave_event",
            self.on_axis_leave
        )

        # Extract the axis the press event happened in
        self.event_axes = [a for a in self.figure.get_axes() if a.in_axes(event)]
        if not self.event_axes:
            return

        # Set the selection start positions
        self.start = (event.x, event.y)
        self.start_data = (event.xdata, event.ydata)

    def on_drag(self, event):
        """Callback function for mouse drag events. Is continuously called during mouse drag."""

        start = self.start

        # Only continue if drag event happens in axis where the initial button press happened.
        if self.event_axes:
            axes = self.event_axes[0]
        else:
            return

        # Clip the selection box data to the axis borders
        (x1, y1), (x2, y2) = np.clip(
            [start, [event.x, event.y]], axes.bbox.min, axes.bbox.max)

        # Draw the selection box
        self.draw_rubberband(event, x1, y1, x2, y2)

    def on_release(self, event):
        """Callback function for a mouse release event."""

        # Check if matplotlib navigation tools are inactive
        tool_mode = str(self.canvas.toolbar.mode)
        if tool_mode!="":
            return
        
        # Check if a mouse drag callback is registered
        if self._select_id is None or self.start is None:
            return

        # Disconnect registered selection callbacks and remove selection rectangle
        self.canvas.mpl_disconnect(self._select_id)
        self.canvas.mpl_disconnect(self._leave_id)
        self.remove_rubberband()

        # Set finish data only if mouse is inside the plotting axis
        if self.event_axes[0].in_axes(event):
            self.finish_data = (event.xdata, event.ydata)

        # Process selected points
        self.find_selected()
        self.set_colors()
        self.update()

        # Reset
        self.start = None
        self.start_data, self.finish_data = None, None

    def draw_rubberband(self, event, x0, y0, x1, y1):
        """Function for drawing the selection rectangle."""

        self.remove_rubberband()
        height = self.canvas.figure.bbox.height
        y0 = height - y0
        y1 = height - y1
        self.lastrect = self.canvas._tkcanvas.create_rectangle(x0, y0, x1, y1)

    def remove_rubberband(self):
        """Function for removing the selection rectangle."""

        if hasattr(self, "lastrect"):
            self.canvas._tkcanvas.delete(self.lastrect)
            del self.lastrect
    
    def find_selected(self):
        """Function for finding points inside the selection rectangle."""

        if self.start_data is None or self.finish_data is None:
            return
        
        # Sort data
        x1, y1 = self.start_data
        x2, y2 = self.finish_data
        xmin, xmax = min(x1, x2), max(x1, x2)
        ymin, ymax = min(y1, y2), max(y1, y2)

        # Check all intersection points if their coordinates are inside the selection intervals
        for row in range(self.grid.intersections.shape[1]):
            for col in range(self.grid.intersections.shape[2]):
                x = self.grid.intersections[0, row, col]
                y = self.grid.intersections[1, row, col]
                if xmin<=x<=xmax and ymin<=y<=ymax:
                    self.selected.append([row, col])
        
        # Set the grids point mask according to the selection mode
        if self.mode == "Select":
            for point in self.selected:
                self.grid.mask[point[0]+1, point[1]+1] = False
        elif self.mode == "Deselect":
            for point in self.selected:
                self.grid.mask[point[0]+1, point[1]+1] = True
        else:
            print(self.selected)

        print(self.selected)

        # Reset
        self.selected = []

    def set_mode(self, new_mode):
        """Function for setting the selection mode and error handling."""

        if new_mode=="Select" or new_mode=="Deselect" or new_mode is None:
            self.mode = new_mode
        else:
            raise ValueError("Mode must be 'Select', 'Deselect' or None")
    
    def set_colors(self):
        """Function for creating the point color array used for plotting."""

        self.point_colors = np.array(["red" if masked else "blue" for masked in self.grid.mask[1:-1, 1:-1].flatten()])

    def draw_points(self):
        """Function for drawing the points in the corresponding colors."""

        self.point_plot.set_color(self.point_colors)
        self.point_plot.draw(self.canvas.get_renderer())

    def update(self):
        """Function for handling draw events using blitting."""

        self.canvas.restore_region(self.background) # Restore background
        self.draw_points()
        self.canvas.blit(self.figure.bbox)
        self.canvas.flush_events() # Process all pending UI events

class process():
    """This base class is a general processor object for image processing."""

    def __init__(self):
        """Defines images from before and after the process."""

        self.before = None
        self.after = None

        self.allowed_keys = ()
    
    def process(self):
        """Method prototype for the processing routine."""
        pass

    def load(self, image):
        """Loads the image input into the processor class."""

        self.before = image

    def set_parameters(self, **kwargs):
        """Method prototype for changing the processor parameters."""
        
        self.__dict__.update((k, v) for k, v in kwargs.items() if k in self.allowed_keys)

    def get_result(self):
        return self.after

class binarization(process):
    """This class is the processor object for calculating a binary image with values 0 = black or 255 = white. If the image is grayscale, threshold is a single int. If it's a color image threshold is a list of 3 ints. If inverted is True all color values below the threshold are set to white."""

    def __init__(self, threshold=127, mode="gray", inverted=False):
        super().__init__()

        self.threshold = threshold
        self.allowed_keys = ("threshold", "mode")
        self.mode = mode
        self.inverted = inverted

        # Map inversion statements to opencv syntax
        self.inversion_map = {
            True : cv.THRESH_BINARY_INV,
            False : cv.THRESH_BINARY
        }
    
    def process(self):
        """Processes binarized image if it's loaded into the processor."""

        try:
            if self.mode == "gray":
                image_data = cv.threshold(
                    self.before.image_data,
                    self.threshold, 255,
                    self.inversion_map[self.inverted])[1]
                self.after = image(image_data, scale=self.before.scale, resolution=self.before.resolution)

            elif self.mode == "color":
                # Some error prevention
                if len(self.inverted)!=3:
                    self.inverted = [self.inverted[0] for i in range(3)]
                if len(self.threshold)!=3:
                    self.threshold = [self.threshold[0] for i in range(3)]
                
                # Individual processing of rgb channels
                image_data_b = cv.threshold(self.before.image_data[:,:,0], self.threshold[0], 255, self.inversion_map[self.inverted[0]])[1]
                image_data_g = cv.threshold(self.before.image_data[:,:,1], self.threshold[1], 255, self.inversion_map[self.inverted[1]])[1]
                image_data_r = cv.threshold(self.before.image_data[:,:,2], self.threshold[2], 255, self.inversion_map[self.inverted[2]])[1]

                # Take maximum pixel value of all binarized color channels
                image_data = np.maximum(np.maximum(image_data_b, image_data_g), image_data_r)
                self.after = image(image_data, scale=self.before.scale, resolution=self.before.resolution)
            else:
                raise ValueError("Binarizer mode needs to be 'gray' or 'color'.")

        
        except:
            print("No image is loaded into binarization processor")

class bilateral_filtering(process):
    """This class is the processor object for the bilateral image filter. Currently unused."""

    def __init__(self, d=9, sigma_color=50):
        super().__init__()
        self.d = d                      # Filter radius
        self.sigma_color = sigma_color  # Color "radius"  

        self.allowed_keys = ("d", "sigma_color")
    
    def process(self):
        """Processes filtered image if it's loaded into the processor."""

        try:
            image_data = cv.bilateralFilter(self.before.image_data, self.d, self.sigma_color, None)
            self.after = image(image_data, scale=self.before.scale, resolution=self.before.resolution)
        
        except:
            print("No image is loaded into bilateral filter")

class rough_grid_finding(process):
    """This subclass roughly finds a grid of lines following the algorithm of Jim Green (https://ntrs.nasa.gov/citations/19940023403). Line_thickness should be generously thicker than lines.
    field_size in mm ; Length over which the pixel values are summed up
    line_thickness in um ; Search width in which one line is contained. Should be generously thicker than the actual line.
    threshold: Relative threshold for peak detection
    num_lines: Number of lines the algorithm is supposed to find
    """

    def __init__(self, field_size=1, line_thickness=80, threshold = 0.8, num_lines=None):
        super().__init__()

        # Standard values
        self.field_size = field_size
        self.line_thickness = line_thickness
        self.threshold = threshold
        self.num_lines = num_lines

        self.allowed_keys = ("field_size", "line_thickness", "threshold", "num_lines")
    
    def process(self):
        """Finds the grid of lines."""

        # Convert mm to pixels
        resolution = self.before.resolution
        self.pixel_field_size = int(self.field_size*resolution)
        self.line_thickness = int(resolution*self.line_thickness/1000)

        # Find lines and save them
        x_lines = self.find_lines('mh', pixel_field_size=self.pixel_field_size)
        y_lines = self.find_lines('mv', pixel_field_size=self.pixel_field_size)
        self.after = grid(grid_lines=[x_lines, y_lines], image=self.before)

    def find_lines(self, location, pixel_field_size):
        """Finds the lines by searching in the specified location. Location can be:
        "mh": middle horizontal,
        "mv": middle vertical,
        "n": north,
        "e": east,
        "s": south,
        "w": west"""

        field_size = pixel_field_size

        # Check if image is loaded
        if self.before is None:
            print("Load an image!")
            return

        # Map location keywords to get image size in summation direction
        axis_size_dir = {
            "mh":"y",
            "mv":"x",
            "n":"y",
            "e":"x",
            "s":"y",
            "w":"x"
        }[location]

        size = getattr(
            self.before.image_size, 
            axis_size_dir, 
            "Specifiy location as 'mh', 'mv', 'n', 'e', 's', 'w'!")

        if location=="mh" or location=="mv":
            # Extract field in the middle

            middle = (size-1)//2
            result = [middle-field_size//2, middle+field_size//2]

            if location == "mh":
                field = self.before.image_data[result[0]:result[1], :]

            elif location == "mv":
                field = self.before.image_data[:, result[0]:result[1]]
        
        elif location=="n" or location=="w":
            # Extract field at 0 pixel indices
            result = [0, field_size-1]

            if location == "n":
                field = self.before.image_data[result[0]:result[1],:]

            elif location == "w":
                field = self.before.image_data[:, result[0]:result[1]] 
        
        elif location=="e" or location=="s":
            # Extract field at -1 pixel indices
            result = [size-1-field_size, size-1]

            if location == "s":
                field = self.before.image_data[result[0]:result[1],:]

            elif location == "e":
                field = self.before.image_data[:, result[0]:result[1]] 

        # Indirectly map location keyword to summation direction
        axis_num = {"x":1, "y":0}[axis_size_dir]
        sum = np.sum(field, axis=axis_num)
        if max(sum) != 0:
            sum = sum/max(sum) # Normalize sum array

        # Plot sum array if in debug mode
        global debug_mode

        if debug_mode:
            plt.plot(sum)
            plt.plot([0, len(sum)-1], [self.threshold, self.threshold])
            plt.title(location)
            plt.show()

        # Invert location keyword map for iterating over summation array
        axis_size_dir = {
            "mh":"x",
            "mv":"y",
            "n":"x",
            "e":"y",
            "s":"x",
            "w":"y"
        }[location]

        size = getattr(
            self.before.image_size, 
            axis_size_dir, 
            "Specifiy location as 'mh', 'mv', 'n', 'e', 's', 'w'!")

        c = 0 # counter
        temp = [0, 0]

        result = [] # List of peak center indices
        spans = [] # List of peak spans

        while c < size:
            # Approach peak from one side
            if sum[c] > self.threshold:
                temp[0] = c
                for j in range(c+self.line_thickness, c-1, -1):
                    # Approach peak from the opposite side
                    if j >= size:
                        # Don't leave the image
                        continue
                    if sum[j] > self.threshold:
                        temp[1] = j
                        break
                spans.append(temp[:])
                result.append((temp[0] + temp[1])/2) # Middle of the peak
                c = c + self.line_thickness - 1 # Continue searching after the peak span
            c = c + 1
        
        if self.num_lines is None:
            # Return all the found peaks
            return np.array(result)
        
        # Find the maximum value in each peak span
        span_max = np.zeros((len(spans),1))
        for i in range(len(spans)):
            lower = spans[i][0]
            upper = spans[i][1]

            # Some error handling if span includes only one value
            if lower==upper:
                span_max[i] = sum[lower]
                continue

            span_max[i] = np.max(sum[lower:upper])
        
        # Find the num_lines highest peaks
        partial_result = []
        for i in range(self.num_lines):
            try:
                index = np.argmax(span_max)
                partial_result.append(result[index])
                span_max = np.delete(span_max, index)

            except ValueError:
                break
        
        return np.array(partial_result)
 
class sequence(process):
    """This class is the processor object for a sequential combination of processes. It functions the same way a singular process would."""

    def __init__(self, sequence=[]):

        super().__init__()
        # Sequence saved as a list of processes.
        self.sequence = sequence
    
    def process(self):
        """Processes the image after the whole sequence."""
        if not self.sequence: # If sequence is empty
            return

        # Process sequence starting at index 0
        self._process_partial_sequence(0)

    def _process_element(self, idx):
        """Processes a single sequence element."""

        self._transfer(idx)
        self.sequence[idx].process()

    def _process_partial_sequence(self, idx):
        """Processes the sequence starting from specified index."""

        for c in range(idx, len(self.sequence)):
            self._process_element(c)
        
        self.after = self.sequence[-1].get_result()

    def _transfer(self, idx):
        """Transfers the result from previous process to process with specified index."""
        if not self.sequence:
            return

        if idx==0:
            # Load self.before for first element
            self.sequence[0].load(self.before)
        
        else:
            self.sequence[idx].load(self.sequence[idx-1].get_result()) # Load result from previous process

    def append(self, process):
        """Appends a process at the end of the sequence and loads it with the result of the last entry."""

        self.sequence.append(process)
        self._transfer(-1)

    def pop(self, idx, recalculate):
        """Removes the process with specified index from the sequence and recalculates the following sequence if recalculate is True."""

        self.sequence.pop(idx)
        self-_transfer(idx)

        if recalculate:
            _process_partial_sequence(idx)

    def insert(self, idx, process, recalculate):
        """Inserts a process at the specified index and recalculates the rest of the sequence if recalculate is True."""

        self.sequence.insert(idx, process)
        self._transfer(idx)

        if recalculate:
            self._process_partial_sequence(idx)
    
    def set_parameters(self, idx, **kwargs):
        """Set the parameters of the process at the specified index."""

        self.sequence[idx].set_parameters(**kwargs)

class subdividing(process):
    """This class subdivides a grid and corresponding image such that each subimage contains one intersection point. The border of the local image is given by the middle between two gridpoints, the global image edge or the symmetrical distance from the gridpoint. If the optional field size argument is given it controls the side length of the local image."""

    def __init__(self, field_size=1.66):
        super().__init__()
        # before as grid with image
        # after as list of local images
        
        self.field_size = field_size # optional argument

        self.allowed_keys = ("field_size")

    def process(self):
        # Initialize self.after array
        col_size = self.before.grid_size.x
        row_size = self.before.grid_size.y
        self.after = np.full((row_size, col_size), None)

        # Convert from mm to pixels
        resolution = self.before.image.resolution
        self.pixel_field_size = int(self.field_size*resolution)

        # Iterate over all intersections
        for row in range(self.before.grid_size.y):
            for col in range(self.before.grid_size.x):
                if self.pixel_field_size is None:
                    # If field size argument is not specified, split in middle between points
                    span = self.find_split([row, col])
                else:
                    span = self.find_field([row, col], self.pixel_field_size)
                
                # Span array contains slicing indices for image extraction
                # Span is None if intersection point is masked
                if span is not None:
                    subimage = self.extract_subimage(span)
                    suborigin = span[:,0] # Origin at upper left corner of the subimage
                    gridpoint = [[self.before.intersections[i,row, col]] for i in [0,1]]
                    self.after[row, col] = grid(grid_lines=gridpoint, image=subimage, origin=suborigin)
    
    def find_split(self, gridpoint):
        """Function to find the span array of the specified intersection if the global image is split between two points respectively."""

        # TODO Reimplement this function to find the next unmasked point and split in the middle instead of only considering the 8 surrounding points.

        result = np.zeros((2, 2)) # [[xspan], [yspan]]
        row = gridpoint[0]
        col = gridpoint[1]

        # Mask index is +1 due to border mask
        mask_row = row+1
        mask_col = col+1

        # Return None if point is masked
        point_mask = self.before.mask[mask_row, mask_col]
        if point_mask:
            return None

        # Extract 3x3 mask around point
        kernel_mask = np.copy(self.before.mask[mask_row-1:mask_row+2, mask_col-1:mask_col+2])
        point = self.before.intersections[:, row, col]
        
        # Map for accessing the 3x3 mask considering the split direction.
        idx = {
            0:[1,0, 1,-1],
            1:[0,1, -1,1]
        }

        # Loop to find the split in horizontal=0 and vertical=1 direction
        for c in [0,1]:
            # Standard field size if all surrounding points are masked
            image_size = getattr(self.before.image.image_size, {0:"x", 1:"y"}[c])
            grid_size = getattr(self.before.grid_size, {0:"x", 1:"y"}[c])
            field_size = image_size//(grid_size+1)

            if not kernel_mask[idx[c][0],idx[c][1]]:
                # If east or north point is not masked, append midpoint to result
                point_min = self.before.intersections[c, row-c, col-1+c]
                result[c, 0] = (point[c] + point_min)//2
    
                if kernel_mask[idx[c][2],idx[c][3]]:
                    # If opposite point is masked do symmetric field expansion
                    result[c, 1] = 2*point[c]-result[c, 0]
    
            if not kernel_mask[idx[c][2],idx[c][3]]:
                # If west or south point is not masked, append midpoint to result
                point_max = self.before.intersections[c, row+c, col+1-c]
                result[c, 1] = (point[c] + point_max)//2
    
                if kernel_mask[idx[c][0],idx[c][1]]:
                    # If opposite point is masked do symmetric field expansion
                    result[c, 0] = 2*point[c]-result[c, 1]
            
            if kernel_mask[idx[c][0],idx[c][1]] and kernel_mask[idx[c][2],idx[c][3]]:
                # If all direct neighbors are masked apply standard field size
                result[c, 0] = point[c]-field_size//2
                result[c, 1] = point[c]+field_size//2
            
            # Clip the found spans at the image border
            result[c, 0] = self.fit_to_image(result[c, 0], image_size)
            result[c, 1] = self.fit_to_image(result[c, 1], image_size)
            
        return result.astype(int)
    
    def find_field(self, gridpoint, field_size):
        """Function to find the span array of the specified intersection point if local images are extracted with equal field sizes symmetrically around the point."""
        
        result = np.zeros((2,2)) # [[x_span], [y_span]]

        row = gridpoint[0]
        col = gridpoint[1]

        # Mask index is +1 due to border mask
        mask_row = row+1
        mask_col = col+1

        # Return None if point is masked
        if self.before.mask[mask_row, mask_col]:
            return None

        point = self.before.intersections[:,row, col].astype(int)

        # Loop over slicing directions
        for c in [0, 1]:
            image_size = getattr(self.before.image.image_size, {0:"x", 1:"y"}[c])
            
            # Calculate upper and lower span limit
            max = point[c]+field_size//2
            min = point[c]-field_size//2

            # Clip the found span at the image border
            result[c, 0] = self.fit_to_image(min, image_size)
            result[c, 1] = self.fit_to_image(max, image_size)
        
        return result
    
    def fit_to_image(self, index, size):
        """Function to limit the specified index to be inside the specified image size."""

        if index < 0:
            index = 0
        
        elif index >= size:
            index = size-1
        
        return index

    def extract_subimage(self, span):
        """Function to extract the subimage in the specified Coordinate span. Returns the subimage"""

        x_min = int(span[0, 0])
        x_max = int(span[0, 1])
        y_min = int(span[1, 0])
        y_max = int(span[1, 1])
        
        image_data = self.before.image.image_data[y_min:y_max, x_min:x_max]
        return image(image_data=image_data, scale=self.before.scale, resolution=self.before.image.resolution)

class vector_intersection(process):
    """This class finds where lines cross the edges of an extracted subimage and refines the intersection position using vector geometry.
    field_size in mm ; width over which pixels are summed up
    line_thickness in um ; Search width in which one line is contained. Should be generously thicker than the actual line.
    threshold: Relative threshold for peak detection"""

    def __init__(self, field_size=0.4, line_thickness=80, threshold= 0.6):
        super().__init__()
        self.field_size = field_size
        self.line_thickness = line_thickness
        self.threshold = threshold

    def process(self):
        # Convert from mm to pixels
        resolution = self.before.image.resolution
        self.pixel_field_size = int(self.field_size*resolution)
        self.pixel_line_thickness = int(self.line_thickness*resolution/1000)

        # Calculating the local intersection coordinates as floats
        edge_points = self.find_edge_points()
        result = self.find_intersection(edge_points)

        # Set result also considering the origin point
        self.after = grid(
            grid_lines=result, image=self.before.image)
        self.after.origin = self.before.origin

    def find_edge_points(self):
        """Function to find where lines cross the edges of the local image. Returns a (4x2)-np.array of edge point coordinates"""

        # Use line finder from rough_grid_finding class. Only find 1 line, otherwise program might give an error.
        linefndr = rough_grid_finding(
            field_size=self.pixel_field_size, threshold=self.threshold, line_thickness=self.pixel_line_thickness, num_lines=1)
        linefndr.load(self.before.image)

        # Find line crossings at four different edges.
        n_x = linefndr.find_lines("n", pixel_field_size=self.pixel_field_size)[0]
        n_y = 0
        s_x = linefndr.find_lines("s", pixel_field_size=self.pixel_field_size)[0]
        s_y = self.before.image.image_size.y-1
        e_y = linefndr.find_lines("e", pixel_field_size=self.pixel_field_size)[0]
        e_x = self.before.image.image_size.x-1
        w_y = linefndr.find_lines("w", pixel_field_size=self.pixel_field_size)[0]
        w_x = 0

        # Plot local image and found line crossings for debugging purposes
        global debug_mode
        if debug_mode:
            
            plt.imshow(self.before.image.image_data, cmap="gray")
            plt.scatter([n_x, s_x, e_x, w_x],[n_y, s_y, e_y, w_y])
            plt.show()

        return np.array([
            [n_x, n_y],
            [s_x, s_y],
            [e_x, e_y],
            [w_x, w_y]
        ])

    def find_intersection(self, edge_points): 
        """Function to find a Line–line intersection from given edge points. https://en.wikipedia.org/wiki/Line%E2%80%93line_intersection"""

        x1, x2, x3, x4 = edge_points[:,1] # Switched because of inverted y-axis
        y1, y2, y3, y4 = edge_points[:,0]

        py = ((x1 * y2 - y1 * x2) * (x3 - x4) - (x1 - x2) * (x3 * y4 - y3 * x4)) / (
                (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4))
        px = ((x1 * y2 - y1 * x2) * (y3 - y4) - (y1 - y2) * (x3 * y4 - y3 * x4)) / (
                (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4))
        return [[px], [py]]

class recombining(process):
    """Class for recombining an array of refined intersection points to a big grid.
    orig_grid is the unrefined grid."""

    def __init__(self, orig_grid):
        super().__init__()

        self.orig_grid = orig_grid
        self.allowed_keys = ("orig_grid")

    def process(self):
            self.after = self.orig_grid

            for row in range(len(self.before)):
                for col in range(len(self.before[row])):
                    if self.before[row, col] is None:
                        continue

                    # If a point has been refined replace it in the original grid by the refined point
                    grid = self.before[row, col]
                    point = np.add(grid.intersections[:, 0, 0], grid.origin) # Add the origin offset to the local grid coordinates

                    self.after.intersections[:,row, col] = point
