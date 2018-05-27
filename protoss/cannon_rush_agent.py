from pysc2.agents import base_agent
from pysc2.lib import actions
from pysc2.lib import features

# Functions
_BUILD_FORGE = actions.FUNCTIONS.Build_Forge_screen.id
_BUILD_PYLON = actions.FUNCTIONS.Build_Pylon_screen.id
_NO_OP = actions.FUNCTIONS.no_op.id
_SELECT_POINT = actions.FUNCTIONS.select_point.id
_TRAIN_MARINE = actions.FUNCTIONS.Train_Marine_quick.id
_RALLY_UNITS_MINIMAP = actions.FUNCTIONS.Rally_Units_minimap.id
_SELECT_ARMY = actions.FUNCTIONS.select_army.id
_ATTACK_MINIMAP = actions.FUNCTIONS.Attack_minimap.id

# Features
_PLAYER_RELATIVE = features.SCREEN_FEATURES.player_relative.index
_UNIT_TYPE = features.SCREEN_FEATURES.unit_type.index

# Parameters
_PLAYER_SELF = 1
_SUPPLY_USED = 3
_SUPPLY_MAX = 4
_NOT_QUEUED = [0]
_QUEUED = [1]


class CannonRushAgent(base_agent.BaseAgent):

    def step(self, obs):
        super(CannonRushAgent, self).step(obs)

        if obs.last():
            return actions.FunctionCall(_NO_OP, [])

        if obs.first():
            return actions.FunctionCall(_NO_OP, [])

        return actions.FunctionCall(_NO_OP, [])
