import numpy as np
from pyna.gc.electromagnetics import B_from_TF_coils, B_from_circular_coil, E_from_circular_capacitor, B_from_circular_capacitor_charging
from pyna.gc.EBField import EBField

class LoopBetweenCapacitor(EBField):
    def __init__(self, 
            I_TF_coils=100e5, 
            I_coil=10e5, 
            radius_coil=1.0, 
            radius_capacitor=100.0, 
            I_capacitor=lambda t: 0e-8 * np.sin(t/1e6), 
            Q_capacitor=lambda t:-0e-2 * np.cos(t/1e6), 
            d_capacitor=3.0):
        """Initialize the LoopBetweenCapacitor class with given parameters.

        Args:
            I_TF_coils (float, optional): Current in the TF coils [A]. Defaults to 100e5.
            I_coil (float, optional): Current in the coil [A]. Defaults to 1e5.
            radius_coil (float, optional): Radius of the coil [m]. Defaults to 1.0.
            radius_capacitor (float, optional): Radius of the capacitor plates [m]. Defaults to 100.0.
            I_capacitor (function, optional): Function of current in the capacitor [A]. Defaults to lambda t: 0e-8 * np.sin(t/1e6).
            Q_capacitor (function, optional): Function of charge of the capacitor [C]. Defaults to lambda t: -0e-2 * np.cos(t/1e6).
            d_capacitor (float, optional): Distance between the capacitor plates [m]. Defaults to 3.0.
        """
        self.I_TF_coils = I_TF_coils
        self.I_coil = I_coil
        self.radius_coil = radius_coil
        self.radius_capacitor = radius_capacitor
        self.I_capacitor = I_capacitor
        self.Q_capacitor = Q_capacitor
        self.d_capacitor = d_capacitor

    def E_at(self, x, y, z, t):
        E = E_from_circular_capacitor(x, y, z, Q=self.Q_capacitor(t), a=self.radius_capacitor, d=self.d_capacitor)    
        return E

    def B_at(self, x, y, z, t):
        B = B_from_TF_coils(x, y, z, I=self.I_TF_coils)
        B += B_from_circular_coil(x, y, z, I=self.I_coil, a=self.radius_coil)
        B += B_from_circular_capacitor_charging(x, y, z, I=self.I_capacitor(t), a=self.radius_capacitor, d=self.d_capacitor)
        return B

    