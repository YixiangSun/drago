from __future__ import annotations

from minigrid.minigrid_env import MiniGridEnv
from minigrid.core.grid import Grid
from minigrid.core.mission import MissionSpace
from minigrid.core.world_object import Goal, Wall, WorldObj
import numpy as np

from typing import TYPE_CHECKING, Tuple

import numpy as np

from minigrid.core.constants import (
    COLOR_TO_IDX,
    COLORS,
    IDX_TO_COLOR,
    IDX_TO_OBJECT,
    OBJECT_TO_IDX,
)
from minigrid.utils.rendering import (
    fill_coords,
    point_in_circle,
    point_in_line,
    point_in_rect,
)

if TYPE_CHECKING:
    from minigrid.minigrid_env import MiniGridEnv

Point = Tuple[int, int]

class Block(WorldObj):
    def __init__(self, color: str = "grey"):
        super().__init__("wall", color)

    def see_behind(self):
        return False

    def render(self, img):
        c = COLORS[self.color]

        # Outline
        fill_coords(img, point_in_rect(0.12, 0.88, 0.12, 0.88), c)
        fill_coords(img, point_in_rect(0.18, 0.82, 0.18, 0.82), (0, 0, 0))

        # Horizontal slit
        fill_coords(img, point_in_rect(0.16, 0.84, 0.47, 0.53), c)

class NewFourRoomsEnv(MiniGridEnv):

    """

    ## Registered Configurations

    - `MiniGrid-FourRooms-New`

    """

    def __init__(self, agent_poses=None, goal_poses=None, task_idx=0, episode_length=100, 
                 size=27, dense_reward=False, goal_radius=12, put_blocks=False,
                   **kwargs):
        self.size = size
        agent_pos = agent_poses[task_idx]
        self._agent_default_pos = (self.size // 2, self.size // 2) if agent_pos == "middle"\
            else tuple(agent_pos)
        self._goal_default_poses = goal_poses
        self.current_goal_pos = tuple(goal_poses[task_idx])
        self.dense_reward = dense_reward
        self.goal_radius = goal_radius
        self.put_blocks = put_blocks
        self.blue_block = Block("blue")
        self.green_block = Block("green")
        self.red_block = Block("red")
        self.purple_block = Block("purple")
        self.yellow_block = Block("yellow")
        self.blue_poses = [[6,6],[20,3],[9,16],[20,20]]
        self.red_poses = [[10,9],[22,10],[4,19],[25,23]]
        self.purple_poses = [[10,4],[17,5],[7,23],[16,17]]
        self.yelloe_poses = [[5,2],[17,2],[4,8],[20,8],[4,15],[23,16],[5,22],[18,23]]


        mission_space = MissionSpace(mission_func=self._gen_mission)

        super().__init__(
            mission_space=mission_space,
            width=self.size,
            height=self.size,
            max_steps=episode_length,
            **kwargs,
        )

    @staticmethod
    def _gen_mission():
        return "reach the goal"

    def _gen_grid(self, width, height):
        # Create the grid
        self.grid = Grid(width, height)

        # Generate the surrounding walls
        self.grid.horz_wall(0, 0)
        self.grid.horz_wall(0, height - 1)
        self.grid.vert_wall(0, 0)
        self.grid.vert_wall(width - 1, 0)

        room_w = width // 2
        room_h = height // 2

        # For each row of rooms
        for j in range(0, 2):
            # For each column
            for i in range(0, 2):
                xL = i * room_w
                yT = j * room_h
                xR = xL + room_w
                yB = yT + room_h

                # Bottom wall and door
                if i + 1 < 2:
                    self.grid.vert_wall(xR, yT, room_h)
                    if j == 0:
                        pos1 = (xR, yB - 1)
                        pos2 = (xR, yB)
                        self.grid.set(*pos1, None)
                        self.grid.set(*pos2, None)
                    else:
                        pos = (xR, yT + 1)
                        self.grid.set(*pos, None)

                # Bottom wall and door
                if j + 1 < 2:
                    self.grid.horz_wall(xL, yB, room_w)
                    if i == 0:
                        pos = (xR - 1, yB)
                        self.grid.set(*pos, None)
                    else:
                        pos = (xL + 1, yB)
                        self.grid.set(*pos, None)
        pos = (self.size // 2, self.size // 2)
        self.grid.set(*pos, None)

        # place the blocks
        if self.put_blocks:
            for i, pos in enumerate(self.blue_poses):
                self.grid.set(*pos, self.blue_block)
            for i, pos in enumerate(self.red_poses):
                self.grid.set(*pos, self.red_block)
            for i, pos in enumerate(self.purple_poses):
                self.grid.set(*pos, self.purple_block)
            for i, pos in enumerate(self.yelloe_poses):
                self.grid.set(*pos, self.yellow_block)

        # Randomize the player start position and orientation
        if self._agent_default_pos is not None:
            self.agent_pos = self._agent_default_pos
            self.grid.set(*self._agent_default_pos, None)
            # assuming random start direction
            self.agent_dir = self._rand_int(0, 4)
        else:
            self.place_agent()

        if self._goal_default_poses is not None:
            for goal_pos in self._goal_default_poses:
                goal = Goal()
                self.put_obj(Goal(), *goal_pos)
                goal.init_pos, goal.cur_pos = goal_pos
        else:
            self.place_obj(Goal())
    
    def step(self, action):

        # initialize reward
        reward = 0
        middle = (self.size // 2, self.size // 2)
        dist_to_middle = np.linalg.norm(np.array(self.agent_pos) - np.array(middle))
        dist_to_goal = np.linalg.norm(np.array(self.agent_pos) - np.array(self.current_goal_pos))

        obs, _, _, truncated, info = super().step(action)

        new_dist_to_middle = np.linalg.norm(np.array(self.agent_pos) - np.array(middle))
        new_dist_to_goal = np.linalg.norm(np.array(self.agent_pos) - np.array(self.current_goal_pos))


        if (self.agent_pos[0] - middle[0]) * (self.current_goal_pos[0] - middle[0]) > 0\
            and (self.agent_pos[1] - middle[1]) * (self.current_goal_pos[1] - middle[1]) >= 0:
            if self.dense_reward:
                reward = max(0, 1 - np.linalg.norm(np.array(self.agent_pos) -\
                np.array(self.current_goal_pos))/self.goal_radius) *\
                    int(new_dist_to_goal < dist_to_goal or tuple(self.current_goal_pos) == self.agent_pos) +\
                2 * int(tuple(self.current_goal_pos) == self.agent_pos)
            else:
                reward = 1 * int(tuple(self.current_goal_pos) == self.agent_pos)

        # No terminate condition
        return obs, reward, False, truncated, info