import pyautogui

class MouseController:
    """
    This class allows to control the mouse pointer. It uses the pyautogui library.      
    """

    def __init__(self, precision, speed, failsafe):
        """
        Initializes a new instance of the mouse pointer controller. 
        Mouse movement precision determines how much the mouse moves (can be 'high', 'low', and 'medium').
        The speed defines how fast it moves (can be 'fast', 'slow', and 'medium').   
        The failsafe parameter can be used to change the PyAutoGUI's failsafe mode. When set to true,
        the exception will be raised once the mouse pointer reaches the screen corner.
        """
        #precision_dict={'high':100, 'low':1000, 'medium':500}
        precision_dict={'high':70, 'low':150, 'medium':100}
        #speed_dict={'fast':1, 'slow':10, 'medium':5}
        speed_dict={'fast':0.01, 'slow':0.12, 'medium':0.05}

        self.precision=precision_dict[precision]
        self.speed=speed_dict[speed]

        pyautogui.FAILSAFE = failsafe

    def move(self, x, y):
        """
        Moves the mouse pointer. Call this function with the x and y output of the gaze estimation model.
        """
        pyautogui.moveRel(x*self.precision, -1*y*self.precision, duration=self.speed)
        #pyautogui.moveRel(x*self.precision, -1*y*self.precision, duration=0)
