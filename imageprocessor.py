import cv2 as cv
import matplotlib.pyplot as plt
import matplotlib.widgets as widgets
import numpy as np
from tkinter import filedialog
import csv

debug_mode = False

class image_like():
    def __init__(self, image_data, scale=1):
        self.image_data = image_data
        self.scale = scale

class image_size():
    def __init__(self, x_in, y_in):
        self.x = x_in
        self.y = y_in

class image(image_like):
    def __init__(self, image_data, scale=1):
        super().__init__(image_data, scale)
        x_in = self.image_data.shape[1]
        y_in = self.image_data.shape[0]
        self.image_size = image_size(x_in, y_in)
        self.resolution = 1
    
    def scale_image(self,factor):
        image_data = cv.resize(self.image_data, None, fx=factor, fy=factor)
        scale = factor*self.scale
        return image(image_data, scale)
        
    def show_image(self, mode="gray", title = None):
        if mode == "gray":
            plt.imshow(self.image_data, cmap="gray")
        elif mode == "color":
            plt.imshow(self.image_data)
        else:
            raise ValueError("mode must be 'gray' or 'color'.")
        
        if title is not None:
            plt.title(title)
        plt.show()
            
class grid(image_like):
    """This subclass saves the grid line coordinated and calculates its intersection points together with the corresponding image class object."""

    def __init__(self, grid_lines, image=None, origin=[0,0], mask=None):
        super().__init__(grid_lines)

        self.grid_data = [[j-origin[i] for j in self.image_data[i]] for i in [0,1]]
        del self.image_data
        self.intersections = np.array(np.meshgrid(self.grid_data[0], self.grid_data[1])) #

        row_size = self.intersections.shape[1]
        col_size = self.intersections.shape[2]

        self.grid_size = image_size(col_size, row_size)

        if mask is None:
            self.mask = np.full((row_size+2, col_size+2), True)
            self.mask[:,(0, -1)] = True
            self.mask[(0, -1), :] = True
        else:
            self.mask = mask

        self.image = image
        self.scale = self.image.scale

        self.origin = origin # [0,0] = top result[c, 0], positive directions are result[c, ] & down

    def select_intersections(self, color_mode="gray", title=None, standard_selection_mode="Select", use_mask = False):
        fig, ax = plt.subplots()
        selector = point_selector(fig.canvas, self, ax, mode=standard_selection_mode)
        if title is not None:
            ax.set_title(title)
        if self.image is not None:
            if color_mode == "gray":
                ax.imshow(self.image.image_data, cmap="gray")
            elif color_mode == "color":
                ax.imshow(self.image.image_data)
            else:
                raise ValueError("mode must be 'gray' or 'color'.")

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
        
        actives = [standard_selection_mode=="Select", standard_selection_mode=="Deselect"]

        checkbox_axis = fig.add_axes([0.05, 0.4, 0.1, 0.15])
        checkboxes = widgets.CheckButtons(checkbox_axis, ["Select", "Deselect"], actives=actives)
        checkboxes.on_clicked(checkbox_callback)
        
        plt.show()        
        
    def show_intersections(self, mode="gray", title=None):
        if self.image is not None:
            if mode == "gray":
                plt.imshow(self.image.image_data, cmap="gray")
            elif mode == "color":
                plt.imshow(self.image.image_data)
            else:
                raise ValueError("mode must be 'gray' or 'color'.")
        
        if title is not None:
            plt.title(title)

        xv = self.intersections[0, :,:]
        xv = xv[np.logical_not(self.mask[1:-1, 1:-1])]

        yv = self.intersections[1, :,:]
        yv = yv[np.logical_not(self.mask[1:-1, 1:-1])]

        plt.plot(xv, yv, marker=".", color="r", linestyle="none")
        plt.show()

    def scale_grid(self, factor=1, image=None):
        
        if image is None:
            image = self.image.scale_image(factor)
            
        else:
            factor = image.scale/self.scale
        
        grid_data = [[j*factor for j in c] for c in self.grid_data]
        origin_scaled = [c*factor for c in self.origin]

        return grid(grid_data, image, origin=origin_scaled, mask=self.mask)
    
    def activate(self, coords):
        
        for row in coords[:,0]:
            for col in coords[:,1]:
                self.mask[row+1][col+1] = False
    
    def copy(self):
        copy = grid(self.grid_data, image=self.image)
        copy.origin = self.origin
        return copy

    def get_active(self):
        result = self.intersections[:,np.logical_not(self.mask[1:-1, 1:-1])]
        return np.stack((result[0], result[1]), axis=1)
    
    def points_as_list(self):
        return np.stack((self.intersections[0,:,:].flatten(), self.intersections[1,:,:].flatten()), axis=1)

    def export(self):

        file_path = filedialog.asksaveasfilename(
            title="Export points",
            initialdir="~/Desktop",
            filetypes=[("Comma separated values", ".csv")]    
        )
        file_path = file_path + ".csv"
        with open(file_path, "w", newline="\n") as csv_file:
            csv_writer = csv.writer(csv_file)
            csv_writer.writerow([
                "","Area","Mean","Min", "Max", "X", "Y"
            ])
            points = self.get_active()/240.4333333
            
            filler = np.zeros_like(points[:,0])

            output = np.stack((filler, filler, filler, filler, filler, points[:,0], points[:,1]), axis=1)
            csv_writer.writerows(output)

class point_selector():

    def __init__(self, canvas, grid, plot_axis, mode = None):
        self.canvas = canvas
        self.figure = self.canvas.figure

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
        self._enter_id = None
        self.grid = grid

        self.start = None
        self.start_data = None
        self.finish_data = None
        self.plot_axis = plot_axis
        self.event_axes = None
        self.selected = []

        self.mode = mode

        self.points = grid.points_as_list()

        self.point_colors = None
        
        
        # Define color list
        self.set_colors()

        # Define point plot artists
        self.point_plot = self.plot_axis.scatter(self.points[:,0], self.points[:,1], c=self.point_colors, marker=".", animated=True)

        #self.update()

    def on_draw(self, event):
        self.background = self.canvas.copy_from_bbox(self.figure.bbox)
        self.draw_points()


    def on_axis_leave(self, event):
        if self.event_axes[0].in_axes(event):
            self.finish_data = (event.xdata, event.ydata)

    def on_press(self, event):
        tool_mode = str(self.canvas.toolbar.mode)

        if tool_mode!="":
            return
        
        if (event.button != 1
                or event.x is None or event.y is None):
            return
        
        if self.mode is None:
            return

        self._select_id = self.canvas.mpl_connect(
            "motion_notify_event",
            self.on_drag
        )
        self._leave_id = self.canvas.mpl_connect(
            "axes_leave_event",
            self.on_axis_leave
        )

        self.event_axes = [a for a in self.figure.get_axes() if a.in_axes(event)]
        if not self.event_axes:
            return

        self.start = (event.x, event.y)
        self.start_data = (event.xdata, event.ydata)

    def on_drag(self, event):
        start = self.start
        axes = self.event_axes[0]

        (x1, y1), (x2, y2) = np.clip(
            [start, [event.x, event.y]], axes.bbox.min, axes.bbox.max)

        self.draw_rubberband(event, x1, y1, x2, y2)

    def on_release(self, event):
        tool_mode = str(self.canvas.toolbar.mode)

        if tool_mode!="":
            return
        
        if self._select_id is None or self.start is None:
            return

        self.canvas.mpl_disconnect(self._select_id)
        self.canvas.mpl_disconnect(self._leave_id)
        self.remove_rubberband()

        if self.event_axes[0].in_axes(event):
            self.finish_data = (event.xdata, event.ydata)

        self.find_selected()
        self.set_colors()
        self.update()

        self.start = None
        self.start_data, self.finish_data = None, None

    def draw_rubberband(self, event, x0, y0, x1, y1):
        self.remove_rubberband()
        height = self.canvas.figure.bbox.height
        y0 = height - y0
        y1 = height - y1
        self.lastrect = self.canvas._tkcanvas.create_rectangle(x0, y0, x1, y1)

    def remove_rubberband(self):
        if hasattr(self, "lastrect"):
            self.canvas._tkcanvas.delete(self.lastrect)
            del self.lastrect
    
    def find_selected(self):
        if self.start_data is None or self.finish_data is None:
            return
        
        x1, y1 = self.start_data
        x2, y2 = self.finish_data
        xmin, xmax = min(x1, x2), max(x1, x2)
        ymin, ymax = min(y1, y2), max(y1, y2)

        for row in range(self.grid.intersections.shape[1]):
            for col in range(self.grid.intersections.shape[2]):
                x = self.grid.intersections[0, row, col]
                y = self.grid.intersections[1, row, col]
                if xmin<=x<=xmax and ymin<=y<=ymax:
                    self.selected.append([row, col])
        
        if self.mode == "Select":
            for point in self.selected:
                self.grid.mask[point[0]+1, point[1]+1] = False
        elif self.mode == "Deselect":
            for point in self.selected:
                self.grid.mask[point[0]+1, point[1]+1] = True
        else:
            print(self.selected)

        print(self.selected)

        self.selected = []

    def set_mode(self, new_mode):
        if new_mode=="Select" or new_mode=="Deselect" or new_mode is None:
            self.mode = new_mode
        else:
            raise ValueError("Mode must be 'Select', 'Deselect' or None")
    
    def set_colors(self):
        self.point_colors = np.array(["red" if masked else "blue" for masked in self.grid.mask[1:-1, 1:-1].flatten()])

    def draw_points(self):
        self.point_plot.set_color(self.point_colors)
        self.point_plot.draw(self.canvas.get_renderer())

    def update(self):
        self.canvas.restore_region(self.background)
        self.draw_points()
        self.canvas.blit(self.figure.bbox)
        self.canvas.flush_events()


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
    """This subclass is the processor object for calculating a binary image with values 0 = black or 255 = white.
        If the image is grayscale, threshold is a single int. If it's a color image threshold is a list of 3 ints.
        """

    def __init__(self, threshold=127, mode="gray", inverted=False):
        super().__init__()

        # Standard values
        self.threshold = threshold
        self.allowed_keys = ("threshold", "mode")
        self.mode = mode
        self.inverted = inverted

        self.inversion_map = {
            True : cv.THRESH_BINARY_INV,
            False : cv.THRESH_BINARY
        }
    
    def process(self):
        """Processes binarized image if it's loaded into the processor."""
        try:
            if self.mode == "gray":
                image_data = cv.threshold(self.before.image_data, self.threshold, 255, self.inversion_map[self.inverted])[1]
                self.after = image(image_data, scale=self.before.scale)
            elif self.mode == "color":
                if len(self.inverted)!=3:
                    self.inverted = [self.inverted for i in range(3)]
                if len(self.threshold)!=3:
                    self.threshold = [self.threshold for i in range(3)]
                image_data_b = cv.threshold(self.before.image_data[:,:,0], self.threshold[0], 255, self.inversion_map[self.inverted[0]])[1]
                image_data_g = cv.threshold(self.before.image_data[:,:,1], self.threshold[1], 255, self.inversion_map[self.inverted[1]])[1]
                image_data_r = cv.threshold(self.before.image_data[:,:,2], self.threshold[2], 255, self.inversion_map[self.inverted[2]])[1]
                image_data = np.maximum(np.maximum(image_data_b, image_data_g), image_data_r)
                self.after = image(image_data, scale=self.before.scale)
            else:
                raise ValueError("Binarizer mode needs to be 'gray' or 'color'.")

        
        except:
            print("No image is loaded into binarization processor")

class bilateral_filtering(process):
    """This subclass is the processor object for the bilateral image filter."""

    def __init__(self, d=9, sigma_color=50):
        """Adds the standard filter parameters."""

        super().__init__()
        self.d = d                      # Filter radius
        self.sigma_color = sigma_color  # Color "radius"  

        self.allowed_keys = ("d", "sigma_color")
    
    def process(self):
        """Processes filtered image if it's loaded into the processor."""

        try:
            image_data = cv.bilateralFilter(self.before.image_data, self.d, self.sigma_color, None)
            self.after = image(image_data, scale=self.before.scale)
        
        except:
            print("No image is loaded into bilateral filter")

class rough_grid_finding(process):
    """This subclass roughly finds a grid of lines following the algorithm of Jim Green (https://ntrs.nasa.gov/citations/19940023403). However, the summation fields are located in the middle of the image."""

    def __init__(self, field_size=200, line_thickness=20, threshold = 0.9, num_lines=None):
        super().__init__()

        # Standard values
        self.field_size = field_size
        self.line_thickness = line_thickness
        self.line_distance = 400
        self.threshold = threshold
        self.num_lines = num_lines

        self.allowed_keys = ("field_size", "line_thickness", "threshold", "num_lines")
    
    def process(self):
        """Finds the grid of lines."""

        x_lines = self.find_lines('mh')
        y_lines = self.find_lines('mv')
        self.after = grid([x_lines, y_lines], image=self.before)

    def find_lines(self, location):
        """Finds the lines by searching in the specified location. Location can be "mh", "mv", "n", "e", "s", "w". """

        if self.before is None:
            print("Load an image!")
            return

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
        
        axis_num = {"x":1, "y":0}[axis_size_dir]

        if location=="mh" or location=="mv":
            
            middle = (size-1)//2
            result = [middle-self.field_size//2, middle+self.field_size//2]

            if location == "mh":
                field = self.before.image_data[result[0]:result[1], :]

            elif location == "mv":
                field = self.before.image_data[:, result[0]:result[1]]
        
        elif location=="n" or location=="w":
            result = [0, self.field_size-1]

            if location == "n":
                field = self.before.image_data[result[0]:result[1],:]

            elif location == "w":
                field = self.before.image_data[:, result[0]:result[1]] 
        
        elif location=="e" or location=="s":
            result = [size-1-self.field_size, size-1]

            if location == "s":
                field = self.before.image_data[result[0]:result[1],:]

            elif location == "e":
                field = self.before.image_data[:, result[0]:result[1]] 


        sum = np.sum(field, axis=axis_num)
        sum = sum/max(sum)

        global debug_mode

        if debug_mode:
            plt.plot(sum)
            plt.plot([0, len(sum)], [self.threshold, self.threshold])
            plt.title(location)
            plt.show()

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

        c = 0
        temp = [0, 0]

        result = []
        spans = []

        while c < size:
            if sum[c] > self.threshold:
                temp[0] = c
                for j in range(c+self.line_thickness, c-1, -1):
                    if j >= size:
                        continue
                    if sum[j] > self.threshold:
                        temp[1] = j
                        break
                spans.append(temp[:])
                result.append((temp[0] + temp[1])/2)
                c = c + self.line_thickness - 1
            c = c + 1
        
        if self.num_lines is None:
            return np.array(result)
        
        span_max = np.zeros((len(spans),1))
        for i in range(len(spans)):
            lower = spans[i][0]
            upper = spans[i][1]

            if lower==upper:
                span_max[i] = sum[lower]
                continue

            span_max[i] = np.max(sum[lower:upper])
        
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
    """This subclass is the processor object for a sequential combination of processes."""

    def __init__(self, sequence=[]):
        """Creates a list containing the sequential processes."""

        super().__init__()
        self.sequence = sequence
    
    def process(self):
        """Processes the image after the whole sequence."""
        if not self.sequence: # If sequence is empty
            return

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
            self.sequence[0].load(self.before) # Load sequence input into appended process
        
        else:
            self.sequence[idx].load(self.sequence[idx-1].get_result()) # Load result from previous process into appended process

    def append(self, process):
        """Appends a process at the end of the sequence."""

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
        self.sequence[idx].set_parameters(**kwargs)

class subdividing(process):
    """This subclass subdivides an image with the given grid points. The border of the local image is given by the middle between two gridpoints, the global image edge or the symmetrical distance from the gridpoint. If the optional field size argument controls the side length of the local image if it is given."""

    def __init__(self, field_size=None):
        super().__init__()
        # before as grid with image
        # after as list of local images

        self.field_size = field_size # optional argument

        self.allowed_keys = ("field_size")

    def process(self):
        col_size = self.before.grid_size.x
        row_size = self.before.grid_size.y

        self.after = np.full((row_size, col_size), None)

        for row in range(self.before.grid_size.y):
            for col in range(self.before.grid_size.x):
                
                if self.field_size is None:
                    span = self.find_split([row, col])
                else:
                    span = self.find_field([row, col], self.field_size)
                
                if span is not None:
                    subimage = self.extract_subimage(span)
                    suborigin = span[:,0]
                    gridpoint = [[self.before.intersections[i,row, col]] for i in [0,1]]
                    self.after[row, col] = grid(gridpoint, image=subimage, origin=suborigin)
    
    def find_split(self, gridpoint):
        result = np.zeros((2, 2)) # [xspan, yspan]
        row = gridpoint[0]
        col = gridpoint[1]

        mask_row = row+1
        mask_col = col+1
        point_mask = self.before.mask[mask_row, mask_col]
        
        if point_mask:
            return None

        kernel_mask = np.copy(self.before.mask[mask_row-1:mask_row+2, mask_col-1:mask_col+2])
        point = self.before.intersections[:, row, col]
        
        idx = {
            0:[1,0, 1,-1],
            1:[0,1, -1,1]
        }

        for c in [0,1]:
            
            image_size = getattr(self.before.image.image_size, {0:"x", 1:"y"}[c])
            grid_size = getattr(self.before.grid_size, {0:"x", 1:"y"}[c])

            field_size = image_size//(grid_size+1)

            if not kernel_mask[idx[c][0],idx[c][1]]:
                point_min = self.before.intersections[c, row-c, col-1+c]
                result[c, 0] = (point[c] + point_min)//2
    
                if kernel_mask[idx[c][2],idx[c][3]]:
                    result[c, 1] = 2*point[c]-result[c, 0]
    
            if not kernel_mask[idx[c][2],idx[c][3]]:
                point_max = self.before.intersections[c, row+c, col+1-c]
                result[c, 1] = (point[c] + point_max)//2
    
                if kernel_mask[idx[c][0],idx[c][1]]:
                    result[c, 0] = 2*point[c]-result[c, 1]
            
            if kernel_mask[idx[c][0],idx[c][1]] and kernel_mask[idx[c][2],idx[c][3]]:
                result[c, 0] = point[c]-field_size//2
                result[c, 1] = point[c]+field_size//2
            
            
            result[c, 0] = self.fit_to_image(result[c, 0], image_size)
            result[c, 1] = self.fit_to_image(result[c, 1], image_size)
            
        return result.astype(int)
        
    
    def find_field(self, gridpoint, field_size):
        result = np.zeros((2,2)) # [x_span, y_span]

        row = gridpoint[0]
        col = gridpoint[1]

        mask_row = row+1
        mask_col = col+1

        if self.before.mask[mask_row, mask_col]:
            return None

        point = self.before.intersections[:,row, col].astype(int)

        for c in [0, 1]:
            image_size = getattr(self.before.image.image_size, {0:"x", 1:"y"}[c])
            
            max = point[c]+field_size//2
            min = point[c]-field_size//2

            result[c, 0] = self.fit_to_image(min, image_size)
            result[c, 1] = self.fit_to_image(max, image_size)
        
        return result
    
    def fit_to_image(self, index, size):

        if index < 0:
            index = 0
        
        elif index >= size:
            index = size-1
        
        return index

    def extract_subimage(self, result):        
        x_min = int(result[0, 0])
        x_max = int(result[0, 1])
        y_min = int(result[1, 0])
        y_max = int(result[1, 1])
        
        image_data = self.before.image.image_data[y_min:y_max, x_min:x_max]
        return image(image_data=image_data, scale=self.before.scale)

class vector_intersection(process):

    def __init__(self, field_size=50, line_thickness=15, threshold= 0.9):
        super().__init__()
        # Before is subgrid with approximate point
        # After is same subgrid with refined point
        self.field_size = field_size
        self.line_thickness = line_thickness
        self.threshold = threshold
        pass

    def process(self):
        edge_points = self.find_edge_points()
        result = self.find_intersection(edge_points)
        self.after = grid(
            result, image=self.before.image)
        
        self.after.origin = self.before.origin

    def find_edge_points(self):
        linefndr = rough_grid_finding(
            field_size=self.field_size, threshold=self.threshold, line_thickness=self.line_thickness, num_lines=1)
        linefndr.load(self.before.image)

        n_x = linefndr.find_lines("n")[0]
        n_y = 0
        s_x = linefndr.find_lines("s")[0]
        s_y = self.before.image.image_size.y-1
        e_y = linefndr.find_lines("e")[0]
        e_x = self.before.image.image_size.x-1
        w_y = linefndr.find_lines("w")[0]
        w_x = 0

        global debug_mode

        if debug_mode:
            plt.imshow(self.before.image.image_data, cmap="gray")
            plt.scatter([n_x, e_x, s_x, w_x], [n_y, e_y, s_y, w_y])
            plt.show()

        return np.array([
            [n_x, n_y],
            [s_x, s_y],
            [e_x, e_y],
            [w_x, w_y]
        ])

    def find_intersection(self, edge_points): 
        """https://en.wikipedia.org/wiki/Line%E2%80%93line_intersection.
        function to find a Line–line intersection"""
        x1, x2, x3, x4 = edge_points[:,1]
        y1, y2, y3, y4 = edge_points[:,0]

        px = ((x1 * y2 - y1 * x2) * (x3 - x4) - (x1 - x2) * (x3 * y4 - y3 * x4)) / (
                (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4))
        py = ((x1 * y2 - y1 * x2) * (y3 - y4) - (y1 - y2) * (x3 * y4 - y3 * x4)) / (
                (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4))
        return [[py], [px]]

class recombining(process):

    def __init__(self, orig_grid=None):
        super().__init__()

        self.after = orig_grid

        self.allowed_keys = ("orig_grid")

    
    def process(self):
            gridpoints = np.full_like(self.after.intersections, None)

            for row in range(len(self.before)):
                for col in range(len(self.before[row])):
                    if self.before[row, col] is None:
                        continue

                    grid = self.before[row, col]
                    point = np.add(grid.intersections[:, 0, 0], grid.origin)

                    self.after.intersections[:,row, col] = point
