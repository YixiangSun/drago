from __future__ import annotations

from gymnasium.envs.registration import register

from minigrid import minigrid_env, wrappers
from minigrid.core import roomgrid
from minigrid.core.world_object import Wall
# from minigrid.envs.wfc.config import WFC_PRESETS

import custom_minigrid_env
from custom_minigrid_env import NewFourRoomsEnv

__version__ = "2.3.1"

register(
        id="MiniGrid-FourRooms-New",
        entry_point="custom_minigrid_env.NewFourRoomsEnv:NewFourRoomsEnv",
    )