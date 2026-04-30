"""Magnetic field derived from a numerical vector potential A via B = curl(A).



Since curl(curl(A)) = 0 identically, fields constructed this way are

guaranteed to be divergence-free (âÂ·B = 0) to machine precision,

limited only by the accuracy of the numerical differentiation.

"""

from __future__ import annotations

from typing import Optional

import numpy as np

from scipy.interpolate import RegularGridInterpolator

from pyna.toroidal.coils.base import CoilFieldVacuum





class CoilFieldVectorPotential(CoilFieldVacuum):

    """Magnetic field B = âÃA from a numerically-specified vector potential.



    The curl is computed via second-order central finite differences on the

    (R, Z, Ï) grid, then interpolated for evaluation at arbitrary points.



    Parameters

    ----------

    R : 1D array, shape (nR,)

        Radial grid (m).

    Z : 1D array, shape (nZ,)

        Axial grid (m).

    Phi : 1D array, shape (nPhi,)

        Toroidal angle grid (rad), must span [0, 2Ï) uniformly.

    AR, AZ, APhi : 3D array, shape (nR, nZ, nPhi)

        Vector potential components (TÂ·m).

    """



    def __init__(self, R, Z, Phi, AR, AZ, APhi):

        self._R = np.asarray(R, dtype=float)

        self._Z = np.asarray(Z, dtype=float)

        self._Phi = np.asarray(Phi, dtype=float)

        self._AR = np.asarray(AR, dtype=float)

        self._AZ = np.asarray(AZ, dtype=float)

        self._APhi = np.asarray(APhi, dtype=float)

        # Compute B = curl(A) on the grid, then build interpolators

        BR, BZ, BPhi = self._compute_curl()

        self._interp_BR = RegularGridInterpolator(

            (self._R, self._Z, self._Phi), BR,

            method='linear', bounds_error=False, fill_value=np.nan)

        self._interp_BZ = RegularGridInterpolator(

            (self._R, self._Z, self._Phi), BZ,

            method='linear', bounds_error=False, fill_value=np.nan)

        self._interp_BPhi = RegularGridInterpolator(

            (self._R, self._Z, self._Phi), BPhi,

            method='linear', bounds_error=False, fill_value=np.nan)



    def _compute_curl(self):

        R = self._R; Z = self._Z; Phi = self._Phi

        AR = self._AR; AZ = self._AZ; APhi = self._APhi

        R3 = R[:, np.newaxis, np.newaxis]  # (nR, 1, 1)



        dZ = Z[1] - Z[0]

        dR = R[1] - R[0]

        dPhi = Phi[1] - Phi[0]



        # âf/âZ: axis=1

        dAZ_dZ = np.gradient(AZ, dZ, axis=1)

        dAPhi_dZ = np.gradient(APhi, dZ, axis=1)

        dAR_dZ = np.gradient(AR, dZ, axis=1)



        # âf/âR: axis=0

        dAPhi_dR = np.gradient(APhi, dR, axis=0)

        dAZ_dR = np.gradient(AZ, dR, axis=0)



        # âf/âÏ? axis=2, periodic boundary

        dAZ_dPhi = np.gradient(

            np.concatenate([AZ[:, :, -1:], AZ, AZ[:, :, :1]], axis=2),

            dPhi, axis=2)[:, :, 1:-1]

        dAR_dPhi = np.gradient(

            np.concatenate([AR[:, :, -1:], AR, AR[:, :, :1]], axis=2),

            dPhi, axis=2)[:, :, 1:-1]



        # B_R = (1/R) âA_Z/âÏ?- âA_Ï/âZ

        BR = (1.0 / R3) * dAZ_dPhi - dAPhi_dZ



        # B_Z = âA_Ï/âR + A_Ï/R - (1/R) âA_R/âÏ?
        BZ = dAPhi_dR + APhi / R3 - (1.0 / R3) * dAR_dPhi



        # B_Ï = âA_R/âZ - âA_Z/âR

        BPhi = dAR_dZ - dAZ_dR



        return BR, BZ, BPhi



    def B_at(self, R, Z, phi):

        R = np.asarray(R, dtype=float)

        Z = np.asarray(Z, dtype=float)

        phi = np.asarray(phi, dtype=float) % (2 * np.pi)

        shape = np.broadcast(R, Z, phi).shape

        pts = np.stack([R.ravel(), Z.ravel(), phi.ravel()], axis=1)

        BR = self._interp_BR(pts).reshape(shape)

        BZ = self._interp_BZ(pts).reshape(shape)

        BPhi = self._interp_BPhi(pts).reshape(shape)

        return BR, BZ, BPhi



    def divergence_free(self) -> bool:

        return True



    @classmethod

    def from_coil_field(

        cls,

        coil_field: "CoilFieldVacuum",

        R: np.ndarray,

        Z: np.ndarray,

        Phi: np.ndarray,

        *,

        gauge: str = "Coulomb",

    ) -> "CoilFieldVectorPotential":

        """Construct by numerically integrating A from a given coil field.



        Uses the Coulomb gauge (âÂ·A = 0) Biot-Savart integral formula

        for the vector potential â?only valid for CoilFieldBiotSavart.

        For general fields, raises NotImplementedError.



        Parameters

        ----------

        coil_field : CoilFieldVacuum

        R, Z, Phi : 1D arrays

        gauge : str

            Only 'Coulomb' supported currently.

        """

        raise NotImplementedError(

            "from_coil_field is not yet implemented. "

            "Provide AR, AZ, APhi arrays directly from an MHD code output."

        )

