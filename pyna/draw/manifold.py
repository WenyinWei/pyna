from pyna.topo.manifold import accumulate_s_from_RZ_arr

def line_colored_by_s(fig, ax, W1d_RZ, norm=None, cmap="BrBG"):
    import numpy as np
    from matplotlib.collections import LineCollection
    from matplotlib.colors import ListedColormap, FuncNorm, PowerNorm
    
    # def _forward(x):
    #     return x**(2.0)
    # def _inverse(x):
    #     return x**(1/2.0)
    
    W1d_s = accumulate_s_from_RZ_arr(W1d_RZ)
    ptsnum = len(W1d_s)
    segments = np.empty( (ptsnum-1, 2, 2,) )
    segments[:,0,:] = W1d_RZ[:-1,:]
    segments[:,1,:] = W1d_RZ[1:,:]

    # Create a continuous norm to map from data points to colors
    if norm is None:
        norm = PowerNorm( gamma=1.0, vmin=-0.3*W1d_s.max(), vmax=W1d_s.max(), )
        # norm = FuncNorm( (_forward, _inverse), vmin=W1d_s.min(), vmax=W1d_s.max(), )
    lc = LineCollection(segments, cmap=cmap, norm=norm)
    # Set the values used for colormapping
    lc.set_array(W1d_s)
#     lc.set_linewidth(2)
    line = ax.add_collection(lc)
    # fig.colorbar(line, ax=ax, extend="max")
    
    return line, lc
