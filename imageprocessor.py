import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np


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
        
    def show_image(self):
        plt.imshow(self.image_data, "gray")
        plt.show()
            
class grid(image_like):
    """This subclass saves the grid line coordinated and calculates its intersection points together with the corresponding image class object."""

    def __init__(self, grid_lines, image=None, origin=[0,0]):
        super().__init__(grid_lines)

        self.grid_data = [[j-origin[i] for j in self.image_data[i]] for i in [0,1]]
        del self.image_data
        self.intersections = np.array(np.meshgrid(self.grid_data[0], self.grid_data[1])) #

        row_size = self.intersections.shape[1]
        col_size = self.intersections.shape[2]

        self.grid_size = image_size(col_size, row_size)

        self.mask = np.full((row_size+2, col_size+2), True)
        self.mask[:,(0, -1)] = True
        self.mask[(0, -1), :] = True

        self.image = image
        self.scale = self.image.scale

        self.origin = origin # [0,0] = top result[c, 0], positive directions are result[c, ] & down

    def select_intersections(self):
        fig, ax = plt.subplots()
        if self.image is not None:
            ax.imshow(self.image.image_data, cmap="gray")
        ax.plot(self.intersections[0], self.intersections[1], marker=".", color="r", linestyle="none")
        
        result = []

        def on_press(epress):
            result.append([epress.xdata, epress.ydata])
            

        def on_release(erelease):
            result.append([erelease.xdata, erelease.ydata])
            x1, y1 = result[0]
            x2, y2 = result[1]
            
            print(x1, y1)
            print(x2, y2)
            xmin, xmax = min(x1, x2), max(x1, x2)
            ymin, ymax = min(y1, y2), max(y1, y2)

            # Select the points within the box
            selected = []

            for row in range(self.intersections.shape[1]):
                for col in range(self.intersections.shape[2]):
                    x = self.intersections[0, row, col]
                    y = self.intersections[1, row, col]

                    if xmin<=x<=xmax and ymin<=y<=ymax:
                        selected.append([row, col])
            
            selected = np.array(selected)

            self.activate(selected)         

        # Connect the onselect function to the figure
        fig.canvas.mpl_connect('button_press_event', on_press)
        fig.canvas.mpl_connect('button_release_event', on_release)
        plt.show()

    def show_intersections(self):
        if self.image is not None:
            plt.imshow(self.image.image_data, cmap="gray")
        
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

        return grid(grid_data, image, origin=origin_scaled)
    
    def activate(self, coords):
        
        for row in coords[:,0]:
            for col in coords[:,1]:
                self.mask[row+1][col+1] = False
    
    def copy(self):
        copy = grid(self.grid_data, image=self.image)
        copy.origin = self.origin
        return copy

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
    """This subclass is the processor object for calculating a binary image with values 0 = black or 255 = white."""

    def __init__(self, threshold=127):
        super().__init__()

        # Standard values
        self.threshold = threshold
        self.allowed_keys = ("threshold")
    
    def process(self):
        """Processes binarized image if it's loaded into the processor."""
        try:
            image_data = cv.threshold(self.before.image_data, self.threshold, 255, cv.THRESH_BINARY)[1]
            self.after = image(image_data, scale=self.before.scale)
        
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

    def __init__(self, field_size=200, line_thickness=20, threshold = 10000):
        super().__init__()

        # Standard values
        self.field_size = field_size
        self.line_thickness = line_thickness
        self.line_distance = 400
        self.threshold = threshold

        self.allowed_keys = ("field_size", "line_thickness", "threshold")
    
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

        # plt.plot(sum)
        # plt.plot([0, len(sum)], [self.threshold, self.threshold])
        # plt.show()

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

        while c < size:
            if sum[c] > self.threshold:
                temp[0] = c
                for j in range(c+self.line_thickness*3, c, -1):
                    if j >= size:
                        continue
                    if sum[j] > self.threshold:
                        temp[1] = j
                        break
                
                result.append((temp[0] + temp[1])/2)
                c = c + self.line_thickness*3 - 1
            c = c + 1
        
        return np.array(result)
 

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
        point = self.before.intersections[:, row, col].astype(int)
        
        idx = {
            0:[1,0, 1,-1],
            1:[0,1, -1,1]
        }

        for c in [0,1]:
            
            image_size = getattr(self.before.image.image_size, {0:"x", 1:"y"}[c])
            grid_size = getattr(self.before.grid_size, {0:"x", 1:"y"}[c])

            field_size = image_size/(grid_size+1)

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
            
            if kernel_mask[idx[c][1],idx[c][2]] and kernel_mask[idx[c][2],idx[c][3]]:
                result[c, 0] = point[c]-field_size//2
                result[c, 1] = point[c]+field_size//2
            
            
            result[c, 0] = self.fit_to_image(result[c, 0], image_size)
            result[c, 1] = self.fit_to_image(result[c, 1], image_size)
            return result
        
    
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

    def __init__(self, field_size=50, line_thickness=30, threshold= 10000):
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
            field_size=self.field_size, threshold=self.threshold, line_thickness=self.line_thickness)
        linefndr.load(self.before.image)
        
        n_x = linefndr.find_lines("n")[0]
        n_y = 0
        s_x = linefndr.find_lines("s")[0]
        s_y = self.before.image.image_size.y-1
        e_y = linefndr.find_lines("e")[0]
        e_x = self.before.image.image_size.x-1
        w_y = linefndr.find_lines("w")[0]
        w_x = 0

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
