import time
import cigvis
import os
import segyio
from skimage import exposure
from typing import List, Dict, Tuple, Union
import re
import matplotlib.pyplot as plt
import numpy as np
import viser
from PIL import Image, ImageDraw
import imageio.v3 as iio
from packaging import version
import sys
from matplotlib.colors import LinearSegmentedColormap

from seismic_toolkit import SeismicVolume

import cigvis
from cigvis import colormap
from cigvis.visernodes import (
    VolumeSlice,
    PosObserver,
    SurfaceNode,
    MeshNode,
    LogPoints,
    LogLineSegments,
    LogBase,
    Server,
)
from cigvis.meshs import surface2mesh
import cigvis.utils as utils
from cigvis.utils import surfaceutils
from itertools import combinations


class ViserPlotIntegrated():
    
    def __init__(self,
                 server,
                 data_path,
                 iline_xline_data=(None, None),
                 color_map_data=None,
                 mask_path=None,
                 iline_xline_mask=(None, None),
                 color_map_mask=None,
                 tridimensional_plot_path=None,
                 color_map_tridimensional_plot=None,
                 iline_xline_tridimensional_plot=(None, None)
                ):

        self.server = server
        self.data_path = data_path
        self.tridimensional_plot_path = tridimensional_plot_path
        self.mask_path = mask_path
        self.color_map_data = color_map_data
        self.color_map_mask = color_map_mask
        self.color_map_tridimensional_plot = color_map_tridimensional_plot
        self.iline_xline_data = iline_xline_data
        self.iline_xline_mask = iline_xline_mask
        self.iline_xline_tridimensional_plot = iline_xline_tridimensional_plot

        self.data = self.load_data(self.data_path, self.iline_xline_data[0], self.iline_xline_data[1])
        self.data = self.data[:, 3:, :]
            
        if self.mask_path:
            self.mask = self.load_data(self.mask_path, self.iline_xline_mask[0], self.iline_xline_mask[1])
            self.mask = self.mask[:, 3:, :]
        
        if self.tridimensional_plot_path:
            self.tridimensional_plot = self.load_data(self.tridimensional_plot_path, self.iline_xline_tridimensional_plot[0], self.iline_xline_tridimensional_plot[1])
            self.tridimensional_plot  = self.tridimensional_plot[:, 3:, :]
            #self.tridimensional_plot[:, :, 718:] = np.where(self.tridimensional_plot[:, :, 718:] == 1, 0, self.tridimensional_plot[:, :, 718:])
            self.tridimensional_plot = np.argwhere(self.tridimensional_plot == 1)

    def load_data(self, data_file, iline, xline):
    
        if data_file[-3:] == 'npy':
            data = SeismicVolume.from_numpy(data_file).compute()
        elif data_file[-3:] == 'sgy' or data_file[-4:] == 'segy':
            try:
                data = SeismicVolume.from_segy(data_file, iline=iline, xline=xline).compute()
            except:
                print('''Check the data path, if this is correct, try the next hint.The parameters 'iline' and 'xline' are the fields where segy searches for information about the inlines and crosslines of the data.
        The default value for segy data is iline=189 and xline=193, but Petrobras generally uses iline=21 and xline=25. Try with these values for solve it.
        ENDING PROGRAM.''')
                sys.exit()
        else:
            print('Format error. Expected format for data_file: npy, sgy or segy. ENDING PROGRAM.')
            sys.exit()
            
        return data
    
    def create_nodes(self):

        nodes = []
        
        if self.color_map_data is not None:
            data_node = self.create_slices(self.data, cmap=self.color_map_data)
        else:
            data_node = self.create_slices(self.data)

        if self.mask_path is not None and self.color_map_mask is not None:
            cmap_mask = colormap.set_alpha_except_min(self.color_map_mask, alpha=0.5)
            self.add_mask(data_node, self.mask, cmaps=cmap_mask, excpt=0)
        if self.mask_path is not None and self.color_map_mask is None:
            cmap_mask = colormap.set_alpha_except_min('viridis', alpha=0.5)
            self.add_mask(data_node, self.mask, cmaps=cmap_mask, excpt=0)
        
        if self.tridimensional_plot_path is not None and self.color_map_tridimensional_plot is not None:
            tridimensional_plot_node = self.create_point_cloud(self.tridimensional_plot, shape=(self.data.shape[0], self.data.shape[1]), cmap=self.color_map_tridimensional_plot)
            nodes.extend(tridimensional_plot_node)
        if self.tridimensional_plot_path is not None and self.color_map_tridimensional_plot is None:
            #blue_cmap = LinearSegmentedColormap.from_list('blue_cmap', ['blue', 'blue'])
            #tridimensional_plot_node = self.create_point_cloud(self.tridimensional_plot, shape=(self.data.shape[0], self.data.shape[1]), cmap=blue_cmap)
            tridimensional_plot_node = self.create_point_cloud(self.tridimensional_plot, shape=(self.data.shape[0], self.data.shape[1]))
            nodes.extend(tridimensional_plot_node)

        nodes.extend(data_node)

        #server = self.create_server(12345)
        self.plot3D(nodes, server=self.server)
            
    def apply_contrast(self, data):
        min_value = np.min(data)
        data = np.nan_to_num(data, nan=min_value)
        p5, p95 = np.percentile(data, (5, 95))
        data = exposure.rescale_intensity(data, in_range=(p5, p95), out_range=(0, 1))
        return data
        
    def create_slices(self,
                      volume: np.ndarray,
                      pos: Union[List, Dict] = None,
                      clim: List = None,
                      cmap: str = 'Petrel',
                      nancolor=None,
                      intersection_lines: bool = True,
                      line_color='white',
                      line_width=1,
                      **kwargs) -> List:

        # set pos
        # ni, nx, nt = volume.shape
        shape, _ = utils.get_shape(volume, cigvis.is_line_first())
        nt = shape[2]
        if pos is None:
            pos = dict(x=[0], y=[0], z=[nt - 1])
        if isinstance(pos, List):
            assert len(pos) == 3
            if isinstance(pos[0], List):
                x, y, z = pos
            else:
                x, y, z = [pos[0]], [pos[1]], [pos[2]]
            pos = {'x': x, 'y': y, 'z': z}
        assert isinstance(pos, Dict)
    
        if clim is None:
            clim = utils.auto_clim(volume)
    
        nodes = []
        for axis, p in pos.items():
            for i in p:
                nodes.append(
                    VolumeSlice(
                        volume,
                        axis,
                        i,
                        cmap,
                        clim,
                        nancolor=nancolor,
                        **kwargs,
                    ))
    
        if intersection_lines:
            observer = PosObserver(color=line_color, width=line_width)
            for node in nodes:
                observer.link_image(node)
    
        return nodes
    
    
    def add_mask(self,
                 nodes: List,
                 volumes: Union[List, np.ndarray],
                 clims: Union[List, Tuple] = None,
                 cmaps: Union[str, List] = None,
                 alpha=None,
                 excpt=None,
                 **kwargs) -> List:

        if not isinstance(volumes, List):
            volumes = [volumes]
    
        for volume in volumes:
            # TODO: check shape as same as base image
            utils.check_mmap(volume)
    
        if clims is None:
            clims = [utils.auto_clim(v) for v in volumes]
        if not isinstance(clims[0], (List, Tuple)):
            clims = [clims]
    
        if cmaps is None:
            raise ValueError("'cmaps' cannot be 'None'")
        if not isinstance(cmaps, List):
            cmaps = [cmaps] * len(volumes)
        if not isinstance(alpha, List):
            alpha = [alpha] * len(volumes)
        if not isinstance(excpt, List):
            excpt = [excpt] * len(volumes)
        for i in range(len(cmaps)):
            cmaps[i] = colormap.get_cmap_from_str(cmaps[i])
            if alpha[i] is not None:
                cmaps[i] = colormap.fast_set_cmap(cmaps[i], alpha[i], excpt[i])
    
        for node in nodes:
            if not isinstance(node, VolumeSlice):
                continue
            for i in range(len(volumes)):
                node.add_mask(
                    volumes[i],
                    cmaps[i],
                    clims[i],
                )
    
        return nodes
    
    def create_point_cloud(self,
                            tridimensional_plotpoints: Union[List, np.ndarray],
                            tridimensional_plotpoints_type: str = 'point',
                            cmap: str = 'jet',
                            clim: List = None,
                            width: float = 0.1,
                            point_shape: str = 'circle',
                            **kwargs,
    ):
        if not isinstance(tridimensional_plotpoints, List):
            tridimensional_plotpoints = [tridimensional_plotpoints]
    
        nodes = []
        for log in tridimensional_plotpoints:
            assert log.ndim == 2 and log.shape[1] in [3, 4, 6, 7]
            points = log[:, :3]
            values = None
            colors = None
            if log.shape[1] == 3:
                values = log[:, 2]
    
            #colors = np.ones((points.shape[0], 4))  # RGBA
            #colors[:, 3] = 1.0  # opacidade total
    
            tridimensional_plotpoints = LogPoints
            nodes.append(
                tridimensional_plotpoints(
                    points,
                    values,
                    colors,
                    cmap,
                    clim,
                    width,
                    point_shape=point_shape,
                ))
    
        return nodes
    
    def plot3D(self,
                nodes,
                axis_scales=[1, 1, 1],
                fov=30,
                look_at=None,
                wxyz=None,
                position=None,
                server=None,
                run_app=True,
                **kwargs,
              ):
        
        if server is None:
            server = Server(label='cigvis-viser', port=8080, verbose=False)
        server.reset()
        server.init_from_nodes(nodes, axis_scales, fov, look_at, wxyz, position)
    
        if run_app and not cigvis.is_running_in_notebook():
            try:
                while True:
                    time.sleep(0.1)
            except KeyboardInterrupt:
                server.stop()
                del server
                print("Execution interrupted")
    
    
    def run(self):
        try:
            while True:
                time.sleep(0.1)
        except KeyboardInterrupt:
            print("Execution interrupted")
    
    
    def link_servers(self, servers):
        """
        Linking Multiple Server Instances to Each Other
        """
        if not all(isinstance(s, Server) for s in servers):
            raise ValueError("Each element must be instance of `Server`.")
        
        for s1, s2 in combinations(servers, 2):
            s1.link(s2)
            s2.link(s1)
