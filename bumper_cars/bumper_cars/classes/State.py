class State:
    """
    Represents the state of an object in a 2D space.
    
    Attributes:
        x (float): The x-coordinate of the object.
        y (float): The y-coordinate of the object.
        v (float): The longitudinal velocity of the object.
        yaw (float): The yaw angle of the object wrt world.
        omega (float): The angular velocity of the object wrt world.
    """
    def __init__(self, x=0.0, y=0.0, v=0.0, yaw=0.0, omega=0.0):
        """
        Initialize the State object.

        Args:
            x (float): The x-coordinate of the state. Default is 0.0.
            y (float): The y-coordinate of the state. Default is 0.0.
            v (float): The longitudinal velocity of the state. Default is 0.0.
            yaw (float): The yaw angle of the state wrt world. Default is 0.0.
            omega (float): The angular velocity of the state wrt world. Default is 0.0.
        """
        self.x = x
        self.y = y
        self.v = v
        self.yaw = yaw
        self.omega = omega
