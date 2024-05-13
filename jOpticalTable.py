import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import Arc, Rectangle
from matplotlib.lines import Line2D

class Tools:
    """
    Tools contains functions that help with drawing and aligning optics correctly.
    
    Tools is normally not called by the end user, but it is used in other classes
    to ensure optics are placed and rotated correctly.
    """
    def inch_to_mm(x):
        """Convert inches to millimetre."""
        return x*25.4

    def mm_to_inch(x):
        """Convert millimetre to inches."""
        return x/25.4

    def deg_to_rad(x):
        """Convert degrees to radians."""
        return x * (np.pi/180)

    def rotate_point(point, angle):
        """
        Rotate a point by an angle theta around the origin.
        
        Used for rotating optics around their centre. Uses a simple 2D rotation
        matrix (written out in longhand).
        
        Parameters
        ----------
            point : tuple
                Coordinates (x,y) of the point to be rotated relative to (0,0).
            angle : float
                Angle in degrees to rotate the point by.
        Returns
        ----------
            point_rot : tuple
                Coordinates of the rotated point (x_rot, y_rot).
        """
        x_rot = point[0]*np.cos(Tools.deg_to_rad(angle)) - \
            point[1]*np.sin(Tools.deg_to_rad(angle))
        y_rot = point[0]*np.sin(Tools.deg_to_rad(angle)) + \
            point[1]*np.cos(Tools.deg_to_rad(angle))
        point_rot = (x_rot, y_rot)
        return point_rot

    def sind(x):
        """Return sine of an angle in degrees."""
        return np.sin(Tools.deg_to_rad(x))

    def cosd(x):
        """Return cosine of an angle in degrees."""
        return np.cos(Tools.deg_to_rad(x))

    def get_midpoint(line):
        """
        Return the midpoint of a line.
        
        Line is loaded as a tuple of points ( (x1, y1), (x2, y2) )
        
        Parameters
        ----------
            line : tuple
                Contains two points (as tuples (x,y)) that define the line.
        
        Returns
        ----------
            midpoint : tuple
                The midpoint of the line (midpoint_x, midpoint_y)
        """
        point1 = line[0]
        point2 = line[1]
        midpoint_x = 0.5*(point1[0] + point2[0])
        midpoint_y = 0.5*(point1[1] + point2[1])
        midpoint = (midpoint_x, midpoint_y)
        return midpoint

    def get_label_coords(label_pos, x, y, size, labelpad):
        """
        Returns the coordinates where the label for an optic should be.
        
        Calculates them based on the position and size of the optic. User defined
        labelpad can also be used to further move the label. 
        
        Label position is determined by the label_pos parameter.

        Parameters
        ----------
        label_pos : string
            String determining label position relative to the optic. Allowable
            values are 'top', 'bottom', 'left', 'right'.
        x : float
            x-coordinate of the optical element being labelled.
        y : float
            y-coordinate of the optical element being labelled.
        size : float
            size of optical element being labelled.
        labelpad : float
            additional space to put between optic and label.

        Returns
        -------
        label_x : float
            x-coordinate of label position.
        label_y : TYPE
            y-coordinate of label position.

        """
        offset = (size*0.5) + labelpad
        if label_pos == 'top':
            label_x = x
            label_y = y + offset
        elif label_pos == 'bottom':
            label_x = x
            label_y = y - offset
        elif label_pos == 'left':
            label_x = x - offset
            label_y = y
        elif label_pos == 'right':
            label_x = x + offset
            label_y = y
        else:
            raise ValueError(
                'Invalid label position - should be "top", "bottom", "left", \
                    or "right".')
        return label_x, label_y

class OpticalElement:
    """
    Base class for an optical element in an optical schematic.

    Attributes
    ----------
    x : float
        The x-coordinate of the center of the optical element.
    y : float
        The y-coordinate of the center of the optical element.
    size : float
        The size of the optical element (could represent diameter, length, etc.).
    angle : float
        The angle (in radians) of the principle line from the positive x-axis.

    Methods
    -------
    update_position(x, y):
        Update the position of the optical element.
    update_angle(angle):
        Update the orientation of the optical element based on the principle line.
    get_position():
        Return the current position of the optical element.
    get_angle():
        Return the current angle of the principle line.
    """

    def __init__(self, x, y, size, angle, show_principle_line=False, label=""):
        """
        Initializes the OpticalElement with the specified position, size, and angle.

        Parameters
        ----------
        x : float
            x-coordinate of the optical element.
        y : float
            y-coordinate of the optical element.
        size : float
            Size of the optical element.
        angle : float
            Angle of the principle line from the x-axis in degrees.
        show_principle_line : bool, optional
            If True, draws the principle line and a perpendicular line at the center.
        """
        self.x = x
        self.y = y
        self.size = size
        self.angle = np.radians(angle)
        self.principle_line = np.array([np.cos(angle), np.sin(angle)])
        self.show_principle_line = show_principle_line  # Store the indicator for drawing lines
        self.patches = []  # Initialize an empty list to hold graphical patches

        self.add_label(label)

    def update_position(self, x, y):
        """
        Update the position of the optical element.

        Parameters
        ----------
        x : float
            New x-coordinate of the optical element.
        y : float
            New y-coordinate of the optical element.
        """
        self.x = x
        self.y = y

    def update_angle(self, angle):
        """
        Update the orientation of the optical element based on the principle line.

        Parameters
        ----------
        angle : float
            New angle of the principle line in degrees.
        """
        self.angle = angle
        self.principle_line = np.array([np.cos(angle), np.sin(angle)])

    def get_position(self):
        """
        Return the current position of the optical element.

        Returns
        -------
        tuple of float
            The x and y coordinates of the optical element.
        """
        return (self.x, self.y)

    def get_angle(self):
        """
        Return the current angle of the principle line.

        Returns
        -------
        float
            The angle of the principle line in degrees.
        """
        return np.degrees(self.angle)

    def add_patch(self, patch):
        """
        Add a graphical patch to the element's list of patches.

        Parameters
        ----------
        patch : matplotlib.patches.Patch
            The patch to be added.
        """
        self.patches.append(patch)

    def add_label(self, label):
        """
        Adds a label near the element. The specific offset and position can be adjusted.
        """
        offset = 1.3  # This determines how far from the element the label appears
        label_x = self.x - offset * self.size / 2. * np.sin(self.angle)
        label_y = self.y + offset * self.size / 2. * np.cos(self.angle)
        label = mpl.text.Text(label_x, label_y, label, rotation=np.degrees(self.angle), fontsize=10, ha='center', va='center')
        self.patches.append(label)

    def update_angle(self, input_angle):
        """
        Virtual method to update the beam angle upon interaction with this optical element.
        By default, it returns the input angle unchanged.
        """
        return input_angle  # Default behavior: angle doesn't change

class OpticalTable:
    """
    Class representing an optical table where optical elements are placed and visualized.

    Attributes
    ----------
    width : float
        Width of the optical table in user-defined units (e.g., meters, centimeters).
    height : float
        Height of the optical table in user-defined units.
    fig_size : tuple
        Size of the figure that represents the optical table visually.

    Methods
    -------
    __init__(width, height, fig_size=(10, 6)):
        Initializes an OpticalTable with specified dimensions and figure size.
    """

    def __init__(self, width, height, fig_size=(10, 6)):
        """
        Initialize the OpticalTable with specific dimensions and figure size.

        Parameters
        ----------
        width : float
            The width of the optical table (e.g., in meters or centimeters).
        height : float
            The height of the optical table.
        fig_size : tuple, optional
            The size of the figure that represents the table, defaults to (10, 6) inches.
        """
        self.width = width
        self.height = height
        self.fig_size = fig_size
        self.fig, self.ax = plt.subplots(figsize=self.fig_size)
        self.ax.set_xlim(0, self.width)
        self.ax.set_ylim(0, self.height)
        self.ax.set_aspect('equal')
        # self.ax.grid(True, linestyle='none', marker='.')

        self.beams = {}

    def add_beam(self, name, beam):
        self.beams[name] = beam

    def draw_beam(self, name):
        if name in self.beams:
            self.beams[name].draw_beam(self.ax)

    def add_element(self, element):
        """
        Adds an optical element to the table and draws its graphical components.

        Parameters
        ----------
        element : OpticalElement
            The optical element to be added to the table.
        """
        # Iterate through the graphical patches associated with the element and add them to the axes
        for patch in element.patches:
            self.ax.add_artist(patch)

        if element.show_principle_line:
            self._show_principle_line(element)

        # Optionally, update the plot display or make other adjustments
        self.ax.figure.canvas.draw_idle()

    def add_element_along_beam(self, beam_name, distance, element_class, size, angle, **kwargs):
        """
        Adds an optical element along the path of a specified beam.

        Parameters:
        beam_name : str
            The name of the beam along which the element is to be placed.
        element_class : class
            The class of the optical element to be instantiated.
        distance : float
            The distance from the last point on the beam's path to place the new element.
        **kwargs : dict
            Additional parameters necessary for initializing the optical element.
        """
        if beam_name not in self.beams:
            print("Beam not found.")
            return  # Optionally, handle this with exceptions or error logging

        beam = self.beams[beam_name]
        last_x, last_y = beam.path[-1]
        new_x = last_x + distance * np.cos(beam.angle)
        new_y = last_y + distance * np.sin(beam.angle)

        # Create a new element instance with the computed position and additional parameters
        new_element = element_class(new_x, new_y, size, angle, **kwargs)
        
        # Update the beam with the new element's position and possibly modified direction
        beam.add_path_point(new_x, new_y)
        new_angle = new_element.update_angle(beam.angle)
        beam.update_direction(new_angle)

        # Add the new element to the table
        self.add_element(new_element)

        return new_element

    def _show_principle_line(self, element):
        end_point = (element.x + np.cos(element.angle), element.y + np.sin(element.angle))
        self.ax.add_line(plt.Line2D([element.x, end_point[0]], [element.y, end_point[1]], 
                                    linestyle='dashed', color='red', linewidth=.5))

        perp_angle = element.angle + np.pi / 2
        half_size = element.size / 2
        perp_start = (element.x + np.cos(perp_angle) * half_size, element.y + np.sin(perp_angle) * half_size)
        perp_end = (element.x - np.cos(perp_angle) * half_size, element.y - np.sin(perp_angle) * half_size)
        self.ax.add_line(plt.Line2D([perp_start[0], perp_end[0]], [perp_start[1], perp_end[1]], 
                                    linestyle='dashed', color='blue', linewidth=.5))

class LaserBeam:
    def __init__(self, start_x, start_y, angle):
        """
        Initialize a laser beam with a starting position and angle.
        Angle is given in degrees relative to the positive x-axis.
        """
        self.start_x = start_x
        self.start_y = start_y
        self.angle = np.radians(angle)
        self.path = [(start_x, start_y)]  # Store points through which the beam passes

    def update_direction(self, new_angle):
        """
        Update the direction of the laser beam.
        """
        self.angle = new_angle

    def add_path_point(self, x, y):
        """
        Add a new point to the beam's path.
        """
        self.path.append((x, y))

    def draw_beam(self, ax):
        x_vals, y_vals = zip(*self.path)
        ax.plot(x_vals, y_vals, linestyle='--', color='red')

    def get_angle(self):
        return np.degrees(self.angle)

################################
###     Optical Elements     ###
################################

class Mirror(OpticalElement):
    """
    Represents a mirror in an optical setup, using a rectangle to visually depict it.

    Inherits from OpticalElement and adds specific properties that are relevant to mirrors,
    such as reflection properties.

    Attributes
    ----------
    reflectivity : float
        Reflectivity coefficient of the mirror, ranging from 0 (no reflection) to 1 (perfect reflection).
    """

    def __init__(self, x, y, size, angle, reflectivity=1.0, show_principle_line=False, label=""):
        """
        Initializes a Mirror with specified position, size, angle, and reflectivity.

        Parameters
        ----------
        x : float
            x-coordinate of the mirror's center.
        y : float
            y-coordinate of the mirror's center.
        size : float
            Size of the mirror, which could relate to the diameter or side length depending on shape.
        angle : float
            Angle of the mirror's principle line from the x-axis in degrees.
        reflectivity : float, optional
            Reflectivity of the mirror, defaults to 1.0 for perfect reflection.
        show_principle_line : bool, optional
            If True, shows the principle line indicating the normal to the mirror's surface.
        """
        super().__init__(x, y, size, angle, show_principle_line, label)
        self.reflectivity = reflectivity

        # Create a graphical representation of the mirror, a rectangle here
        self.create_patches()

    def create_patches(self):
        """
        Creates a graphical patch representing the mirror and adds it to the patches list.
        The center point is aligned with one of the long edges of the rectangle.
        """
        # Width of the mirror (considering as a thin strip for simplicity)
        width = self.size / 10  # Thin width of the mirror
        height = self.size  # Length of the mirror

        # Calculate the position of the lower left corner of the rectangle so that one of the long edges
        # is centered at (x, y) and lies along the principle line.
        lower_left_x = self.x - height / 2 * np.sin(self.angle)  # Adjust position to move the edge to the center point
        lower_left_y = self.y + height / 2 * np.cos(self.angle)  # Adjust position to move the edge to the center point

        # Create a rectangle to represent the mirror
        # The angle needs to be converted from radians to degrees for matplotlib rotation
        rotation_deg = np.degrees(self.angle)  # Convert radians to degrees for the rotation argument
        mirror_rect = Rectangle((lower_left_x, lower_left_y), width, height, angle=180.+rotation_deg, color='gray', fill=True)
        self.patches.append(mirror_rect)

    def update_angle(self, input_angle):
        """
        Reflects the incoming beam angle based on the mirror's orientation.

        Parameters:
        input_angle : float
            The angle of the incoming laser beam in degrees.

        Returns:
        float
            The angle of the reflected beam in degrees.
        """
        # Calculate the angle of incidence with respect to the normal (mirror's principle line)
        # Mirror's angle is assumed to be the angle of the normal to the reflecting surface
        # Convert both angles to radians for computation
        normal_angle_radians = self.angle
        input_angle_radians = (input_angle + np.pi) % (2.*np.pi)

        # Angle of incidence is the difference between the beam angle and the mirror's normal
        angle_of_incidence = input_angle_radians - normal_angle_radians

        # Reflect the beam by negating the incidence angle relative to the normal
        reflected_angle_radians = normal_angle_radians - angle_of_incidence

        # Convert back to degrees to maintain consistency in interface
        return reflected_angle_radians

class PlanoConvexLens(OpticalElement):
    """
    Represents a plano-convex lens in an optical setup, where the flat surface is at the center
    and the principle line points towards the convex side.

    Attributes
    ----------
    curvature_radius : float
        The radius of curvature of the convex surface.
    """

    def __init__(self, x, y, size, angle, curvature_radius=0.4, show_principle_line=False, label=""):
        """
        Initializes a PlanoConvexLens with specified position, size, angle, and curvature.

        Parameters
        ----------
        x : float
            x-coordinate of the lens's center at the flat surface.
        y : float
            y-coordinate of the lens's center at the flat surface.
        size : float
            Size of the lens, typically representing the diameter.
        angle : float
            Angle of the principle line from the x-axis in degrees, pointing towards the convex side.
        curvature_radius : float
            Radius of curvature of the lens's convex surface.
        show_principle_line : bool, optional
            If True, shows the principle line indicating the direction towards the convex surface.
        """
        super().__init__(x, y, size, angle, show_principle_line, label)
        self.curvature_radius = curvature_radius
        self.create_patches()

    def create_patches(self):
        """
        Creates graphical representations using Line2D for the plano-convex lens and adds them to the patches list.
        The flat surface is represented as a straight line, and the convex surface as a curved line.
        """
        # Parameters for the visual representation
        lens_thickness = self.size / 10  # Thickness of the lens
        half_diameter = self.size / 2   # Half of the lens diameter

        # Calculate endpoints of the flat surface line
        flat_start_x = self.x - half_diameter * np.sin(self.angle)
        flat_start_y = self.y + half_diameter * np.cos(self.angle)
        flat_end_x = self.x + half_diameter * np.sin(self.angle)
        flat_end_y = self.y - half_diameter * np.cos(self.angle)

        # Flat surface line
        flat_line = Line2D([flat_start_x, flat_end_x], [flat_start_y, flat_end_y],
                           linewidth=1, color='gray')
        self.patches.append(flat_line)

        # Calculate endpoints of the convex surface line
        convex_start_x = flat_end_x
        convex_start_y = flat_end_y
        convex_end_x = flat_end_x + lens_thickness * np.cos(self.angle)
        convex_end_y = flat_end_y - lens_thickness * np.sin(self.angle)

        # Create the convex surface as an arc
        arc_angle = 180  # Half-circle to represent the convex shape
        arc = Arc((self.x, self.y), width=self.curvature_radius, height=self.size,
                  theta1=270, theta2=90, angle=np.degrees(self.angle), color='gray', linewidth=1)
        self.patches.append(arc)











