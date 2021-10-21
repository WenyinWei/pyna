import numpy as np
from matplotlib.lines import Line2D

class Line2DExtender:
    def __init__(self, line:Line2D):
        self.line = line
        self.xs = list(line.get_xdata())
        self.ys = list(line.get_ydata())
        self.cid_mouse_press = line.figure.canvas.mpl_connect('button_press_event', self)
        self.cid_key_press = line.figure.canvas.mpl_connect('key_press_event', self.finish)

    def __call__(self, event):
#         print('click', event)
        if event.inaxes!=self.line.axes: return
        self.xs.append(event.xdata)
        self.ys.append(event.ydata)
        self.line.set_data(self.xs, self.ys)
        self.line.figure.canvas.draw()
        
    def finish(self, event):
        if event.inaxes!=self.line.axes: return
        if event.key == 'enter':
            self.xs.append(self.xs[0])
            self.ys.append(self.ys[0])
            self.line.set_data(self.xs, self.ys)
            self.line.figure.canvas.draw()
            self.line.figure.canvas.mpl_disconnect(self.cid_mouse_press)
            self.line.figure.canvas.mpl_disconnect(self.cid_key_press)
            

def uniformly_scattered_points_in_polygon(polygon:Line2DExtender, ny=40):
    if polygon.xs[-1] != polygon.xs[0]:
        raise ValueError("The polygon is not closed in x coordinate.")
    if polygon.ys[-1] != polygon.ys[0]:
        raise ValueError("The polygon is not closed in y coordinate.")
    poly_edge_num = len(polygon.xs) - 1
    
    ymin, ymax = min(polygon.ys), max(polygon.ys)
    ygap = ymax - ymin
    dy = ygap / ny
    dx = dy
    yarr = np.linspace(ymin+dy, ymax, num=ny, endpoint=False)
    
    scattered_points_x = []
    scattered_points_y = []
    def line_of_two_points(x1, y1, x2, y2):
        '''
        return a, b, c of line ax + by + c = 0
        '''
        if abs(y2-y1)> abs(x2-x1): # the line is steeo, let x = k2 * y + b
            k2 = (x2 - x1) / (y2 - y1)
            c = x1 - k2*y1
            return 1, -k2, -c
        else: # let y = k1 * x + b
            k1 = (y2 - y1) / (x2 - x1)
            c = y1 - k1*x1
            return -k1, 1, -c
    for y in yarr:
        intersected_points_x = list()
        for i in range(poly_edge_num):
            if (polygon.ys[i] - y)*(polygon.ys[i+1] - y) < 0: # the segment intersects with y = {y} horizontal line.
                seg_a, seg_b, seg_c = line_of_two_points(
                    polygon.xs[i], polygon.ys[i],
                    polygon.xs[i+1], polygon.ys[i+1])
                intersected_points_x.append( -(seg_b*y+seg_c) / seg_a )
        intersected_points_x.sort()
        if len(intersected_points_x)%2 != 0:
            print(intersected_points_x)
            raise RuntimeError(f"The polygon is strange that we don't have even number of intersected points with y={y} line.")
        
        for i in range(int(len(intersected_points_x)/2)):
            horizon_seg_xbeg = intersected_points_x[2*i]
            horizon_seg_xend = intersected_points_x[2*i+1]
            scat_x = horizon_seg_xbeg
            while True:
                scattered_points_x.append(scat_x)
                scattered_points_y.append(y)
                scat_x+= dx
                if scat_x > horizon_seg_xend:
                    break
        
    return scattered_points_x, scattered_points_y
    
    