import os
import sys

import imageio
import numpy as np

import utils


class VideoRecorder(object):
    def __init__(self, root_dir, height=256, width=256, fps=25, dmc_env = False, env_name = None):
        if not os.path.exists(root_dir):
            os.makedirs(root_dir)
        self.save_dir = utils.make_dir(root_dir, 'video') if root_dir else None
        self.height = height
        self.width = width
        self.fps = fps
        self.frames = []
        self.dmc_env = dmc_env
        self.env_name = env_name
        
    def init(self, enabled=True):
        self.frames = []
        self.enabled = self.save_dir is not None and enabled

        self.num_recorded_frames = 0

    def record(self, env):
        if self.enabled:
            if self.dmc_env:
                frame = env.physics.render(height=self.height,
                                        width=self.width,
                                        camera_id=0)
            else:
                if self.env_name in ['AntMazeSmall-v0', "PointUMaze-v0", "PointSpiralMaze-v0", "PointNMaze-v0"]:
                    frame = env.render(mode='rgb_array', camera_name='topview') 
                elif self.env_name in ['sawyer_peg_push', 'sawyer_peg_pick_and_place']:                    
                    frame = env.render(mode='rgb_array', camera_name='corner3')
                else:
                    frame = env.render(mode='rgb_array') 

            self.frames.append(frame)
            self.num_recorded_frames+=1
        
    def save(self, file_name):
        if self.enabled:
            path = os.path.join(self.save_dir, file_name)
            imageio.mimsave(path, self.frames, fps=self.fps)
            self.num_recorded_frames = 0

