import matplotlib.pyplot as plt
import numpy as np
from matplotlib.widgets import Slider, RadioButtons

class plotter(object):
    def hzfunc(self,label):
        hzdict = {'Kalman ON': 1, 'Kalman OFF': 0}
        self.kalman_on = hzdict[label]
    
    def num_datapoints_update(self, val):
        self.num_datapoints_plot = int(val)  

    def __init__(self, sensor_locations=None):
        self.fig = plt.figure(figsize=(12, 12))

        self.input_fig = plt.figure(figsize=(5, 5))
        axcolor = 'lightgoldenrodyellow'
        self.rax = plt.axes([0.33, 0.6, 0.30, 0.30], facecolor=axcolor)
        self.radio = RadioButtons(self.rax, ('Kalman ON', 'Kalman OFF'))
        
        self.radio.on_clicked(self.hzfunc)
        self.kalman_on = 1

        self.ax_slider = plt.axes([0.35, 0.15, 0.65, 0.03])
        self.num_datapoints_slider = Slider(
            ax=self.ax_slider,
            label='Number of Datapoints',
            valmin=50,
            valmax=1000,
            valinit=300,
        )
        self.num_datapoints_slider.on_changed(self.num_datapoints_update)
        self.num_datapoints_plot = 300

        self.x_y_fig = plt.figure(figsize=(7, 7))
        # y_z_fig = plt.figure(figsize=(7, 7))
        self.x_z_fig = plt.figure(figsize=(7, 7))
        self.ax = self.fig.gca(projection='3d')
        self.x_y = self.x_y_fig.gca()
        # y_z = y_z_fig.gca()
        self.x_z = self.x_z_fig.gca()

        

        self.ax.set_title("3D Plane Movement")
        self.ax.set_xlabel('x(cm)')
        self.ax.set_ylabel('y(cm)')
        self.ax.set_zlabel('z(cm)')
        self.ax.set_xlim([-80, 80])
        self.ax.set_ylim([-10, 70])
        self.ax.set_zlim([-10, 50])
        Xs = 1e2 * sensor_locations[:, 0]
        Ys = 1e2 * sensor_locations[:, 1]
        Zs = 1e2 * sensor_locations[:, 2]

        self.x_y.set_title("X-Y 2D Plane Movement")
        self.x_y.set_xlabel('x(cm)')
        self.x_y.set_ylabel('y(cm)')
        self.x_y.set_xlim([-80, 80])
        self.x_y.set_ylim([-10, 70])

        # y_z.set_title("Y-Z 2D Plane Movement")
        # y_z.set_xlabel('y(cm)')
        # y_z.set_ylabel('z(cm)')
        # y_z.set_xlim([-10, 70])
        # y_z.set_ylim([-10, 50])

        self.x_z.set_title("X-Z 2D Plane Movement")
        self.x_z.set_xlabel('x(cm)')
        self.x_z.set_ylabel('z(cm)')
        self.x_z.set_xlim([-80, 80])
        self.x_z.set_ylim([-10, 50])

        XXs = Xs
        YYs = Ys
        ZZs = Zs
        self.ax.scatter(XXs, YYs, ZZs, c='r', s=1, alpha=0.5)
        self.x_y.scatter(XXs, YYs, c='r', s=1, alpha=0.5)
        # y_z.scatter(YYs, ZZs, c='r', s=1, alpha=0.5)
        self.x_z.scatter(XXs, ZZs, c='r', s=1, alpha=0.5)

        t = 0
        
        (self.magnet_pos,) = self.ax.plot(t / 100.0 * 5, t / 100.0 * 5, t /
                                100.0 * 5, linewidth=3, animated=True, label="Predicted Location")
        (self.x_y_pos, ) = self.x_y.plot(t / 100.0 * 5, t / 100.0 * 5, linewidth=3, animated=True, label="Predicted Location")
        # (y_z_pos, ) = y_z.plot(t / 100.0 * 5, t / 100.0 * 5, linewidth=3, animated=True)
        (self.x_z_pos, ) = self.x_z.plot(t / 100.0 * 5, t / 100.0 * 5, linewidth=3, animated=True, label="Predicted Location")


        (self.magnet_pos_gt,) = self.ax.plot(t / 100.0 * 5, t / 100.0 * 5, t /
                                100.0 * 5, linewidth=3, animated=True, label="Ground Truth Location")
        self.ax.legend()
        (self.x_y_pos_gt, ) = self.x_y.plot(t / 100.0 * 5, t / 100.0 * 5, linewidth=3, animated=True, label="Ground Truth Location")
        self.x_y.legend()
        # (y_z_pos, ) = y_z.plot(t / 100.0 * 5, t / 100.0 * 5, linewidth=3, animated=True)
        (self.x_z_pos_gt, ) = self.x_z.plot(t / 100.0 * 5, t / 100.0 * 5, linewidth=3, animated=True, label="Ground Truth Location")
        self.x_z.legend()

        plt.show(block=False)
        plt.pause(0.1)
        self.bg = self.fig.canvas.copy_from_bbox(self.fig.bbox)
        self.x_yg = self.x_y_fig.canvas.copy_from_bbox(self.x_y_fig.bbox)
        # y_zg = y_z_fig.canvas.copy_from_bbox(y_z_fig.bbox)
        self.x_zg = self.x_z_fig.canvas.copy_from_bbox(self.x_z_fig.bbox)
        self.ax.draw_artist(self.magnet_pos)
        self.x_y.draw_artist(self.x_y_pos)
        # y_z.draw_artist(y_z_pos)
        self.x_z.draw_artist(self.x_z_pos)
        self.ax.draw_artist(self.magnet_pos_gt)
        self.x_y.draw_artist(self.x_y_pos_gt)
        # y_z.draw_artist(y_z_pos_gt)
        self.x_z.draw_artist(self.x_z_pos_gt)
        self.fig.canvas.blit(self.fig.bbox)
        # timer = Timer(text=f"frame elapsed time: {{: .5f}}")
    
    def plot(self, gt_data=None, result=None):
        # gt_data = np.array(gt_data)
        self.fig.canvas.restore_region(self.bg)
        self.x_y_fig.canvas.restore_region(self.x_yg)
        # y_z_fig.canvas.restore_region(y_zg)
        self.x_z_fig.canvas.restore_region(self.x_zg)
        # update the artist, neither the canvas state nor the screen have
        x = result[:, 0] 
        y = result[:, 1] 
        z = result[:, 2]
        # x = x_smooth
        # y = y_smooth
        # z = z_smooth
        xx = x
        yy = y
        zz = z

        x_gt = gt_data[:, 0] * 100
        y_gt = gt_data[:, 1] * 100
        z_gt = gt_data[:, 2] * 100
        xx_gt = x_gt
        yy_gt = y_gt
        zz_gt = z_gt
        
        self.magnet_pos.set_xdata(xx)
        self.magnet_pos.set_ydata(yy)
        self.magnet_pos.set_3d_properties(zz, zdir='z')

        self.magnet_pos_gt.set_xdata(xx_gt)
        self.magnet_pos_gt.set_ydata(yy_gt)
        self.magnet_pos_gt.set_3d_properties(zz_gt, zdir='z')

        self.x_y_pos.set_xdata(xx)
        self.x_y_pos.set_ydata(yy)
        # y_z_pos.set_xdata(yy)
        # y_z_pos.set_ydata(zz)
        self.x_z_pos.set_xdata(xx)
        self.x_z_pos.set_ydata(zz)

        self.x_y_pos_gt.set_xdata(xx_gt)
        self.x_y_pos_gt.set_ydata(yy_gt)
        # y_z_pos_gt.set_xdata(yy_gt)
        # y_z_pos_gt.set_ydata(zz_gt)
        self.x_z_pos_gt.set_xdata(xx_gt)
        self.x_z_pos_gt.set_ydata(zz_gt)

        # re-render the artist, updating the canvas state, but not the screen
        self.ax.draw_artist(self.magnet_pos)
        self.x_y.draw_artist(self.x_y_pos)
        # y_z.draw_artist(y_z_pos)
        self.x_z.draw_artist(self.x_z_pos)

        self.ax.draw_artist(self.magnet_pos_gt)
        self.x_y.draw_artist(self.x_y_pos_gt)
        # y_z.draw_artist(y_z_pos)
        self.x_z.draw_artist(self.x_z_pos_gt)

        # copy the image to the GUI state, but screen might not changed yet
        self.fig.canvas.blit(self.fig.bbox)
        self.x_y_fig.canvas.blit(self.x_y_fig.bbox)
        # y_z_fig.canvas.blit(y_z_fig.bbox)
        self.x_z_fig.canvas.blit(self.x_z_fig.bbox)
        # flush any pending GUI events, re-painting the screen if needed
        self.fig.canvas.flush_events()
        self.x_y_fig.canvas.flush_events()
        # y_z_fig.canvas.flush_events()
        self.x_z_fig.canvas.flush_events()
        # timer.stop()