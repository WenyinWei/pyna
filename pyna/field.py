class RegualrCylindricalGridField:
    def __init__(self, R, Z, Phi, BR, BZ, BPhi) -> None:
        self._R = R
        self._Z = Z
        self._Phi = Phi
        self._BR = BR
        self._BZ = BZ
        self._BPhi = BPhi

    @property
    def R(self):
        return self._R
    @property
    def Z(self):
        return self._Z
    @property
    def Phi(self):
        return self._Phi
    @property
    def BR(self):
        return self._BR
    @property
    def BZ(self):
        return self._BZ
    @property
    def BPhi(self):
        return self._BPhi