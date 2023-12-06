import math

def distance(x1, y1, x2, y2):
    """
    Calculates the Euclidean distance between two points.

    Args:
        x1: The x-coordinate of the first point.
        y1: The y-coordinate of the first point.
        x2: The x-coordinate of the second point.
        y2: The y-coordinate of the second point.

    Returns:
        The Euclidean distance between the two points.
    """
    return math.sqrt((x2 - x1)**2 + (y2 - y1)**2)

def find_closest_point(x1, y1, points):
    """
    Finds the point in a list of points that is closest to a given point.

    Args:
        x1: The x-coordinate of the reference point.
        y1: The y-coordinate of the reference point.
        points: A list of points in the format [(x, y)].

    Returns:
        The point in the list that is closest to the reference point.
    """
    closest_point = None
    closest_distance = float('inf')
    for x2, y2, id in points:
        dist = distance(x1, y1, x2, y2)
        if dist < closest_distance:
            closest_point = (x2, y2, id)
            closest_distance = dist
    return closest_point

def get_middle_coords(x1, y1, x2, y2):
    """
    Calculates the middle coordinates between two points.

    Args:
        x1: The x-coordinate of the first point.
        y1: The y-coordinate of the first point.
        x2: The x-coordinate of the second point.
        y2: The y-coordinate of the second point.

    Returns:
        A tuple containing the middle x and y coordinates.
    """
    middle_x = (x1 + x2) / 2
    middle_y = (y1 + y2) / 2
    return middle_x, middle_y