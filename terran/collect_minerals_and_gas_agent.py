import random

import numpy as np
import pandas as pd
from pysc2.agents import base_agent
from pysc2.lib import actions
from pysc2.lib import features

import QLearning
import os
from lib import Player as p
from lib import CumulativeScores as cs
from lib import unit_type_constants as units

COMMAND_CENTER_PIXELS = 0
SUPPLY_DEPOT_PIXELS = 72
SCV_PIXELS = 0

# Functions
_BUILD_COMMAND_CENTER = actions.FUNCTIONS.Build_CommandCenter_screen.id
_BUILD_SUPPLY_DEPOT = actions.FUNCTIONS.Build_SupplyDepot_screen.id
_BUILD_REFINERY = actions.FUNCTIONS.Build_Refinery_screen.id
_NOOP = actions.FUNCTIONS.no_op.id
_HARVEST_GATHER = actions.FUNCTIONS.Harvest_Gather_screen.id
_TRAIN_SCV = actions.FUNCTIONS.Train_SCV_quick.id
_SELECT_IDLE_WORKER = actions.FUNCTIONS.select_idle_worker.id
_SELECT_POINT = actions.FUNCTIONS.select_point.id
_RALLY_UNITS_MINIMAP = actions.FUNCTIONS.Rally_Units_minimap.id
_SELECT_ARMY = actions.FUNCTIONS.select_army.id
_ATTACK_MINIMAP = actions.FUNCTIONS.Attack_minimap.id
_SELECT_CONTROL_GROUP = actions.FUNCTIONS.select_control_group.id

_CONTROL_GROUP_RECALL = 0
_CONTROL_GROUP_SET = 1

# Features
_PLAYER_RELATIVE = features.SCREEN_FEATURES.player_relative.index
_UNIT_TYPE = features.SCREEN_FEATURES.unit_type.index

# Parameters
_PLAYER_SELF = 1
_SUPPLY_USED = 3
_SUPPLY_MAX = 4
_ARMY_SUPPLY = 5

_NOT_QUEUED = [0]
_QUEUED = [1]
_SELECT_ALL = [2]


_NOTHING = 0


ACTION_DO_NOTHING = 'donothing'
ACTION_BUILD_SUPPLY_DEPOT = 'buildsupplydepot'
ACTION_BUILD_CC = 'buildcommandcenter'
ACTION_BUILD_SCV_START_CC = 'buildscvstartcc'
ACTION_BUILD_SCV_NEW_CC = 'buildscvnewcc'
ACTION_MINE_MINERALS_LEFT = 'minemineralsleft'
ACTION_MINE_MINERALS_RIGHT = 'minemineralsright'
ACTION_BUILD_REFINERY_TOP_LEFT = 'buildrefinerytopleft'
ACTION_BUILD_REFINERY_TOP_RIGHT = 'buildrefinerytopright'
ACTION_BUILD_REFINERY_BOTTOM_LEFT = 'buildrefinerybottomleft'
ACTION_BUILD_REFINERY_BOTTOM_RIGHT = 'buildrefinerybottomright'
ACTION_MINE_GAS_TOP_LEFT = 'minegastopleft'
ACTION_MINE_GAS_TOP_RIGHT = 'minegastopright'
ACTION_MINE_GAS_BOTTOM_LEFT = 'minegasbottomleft'
ACTION_MINE_GAS_BOTTOM_RIGHT = 'minegasbottomright'

left_mins = [(13, 14), (10, 18), (6, 29), (11, 40), (13, 51)]
right_mins = [(69, 16), (74, 19), (77, 30), (73, 39), (69, 50)]
depots = [(5, 5), (77, 5), (77, 60), (5, 60), (30, 5), (37, 5), (44, 5), (51, 5), (30, 60), (37, 60), (44, 60), (51, 60)]
geysers = [(19, 8), (64, 8), (19, 60), (64, 60)]
new_cc_spot = [(50, 32)]
start_cc = None
BUILDERS_CTRL_GROUP = 1

# Reduced action set
smart_actions = [
    ACTION_DO_NOTHING,
    ACTION_BUILD_SUPPLY_DEPOT,
    ACTION_BUILD_CC,
    ACTION_BUILD_SCV_START_CC,
    ACTION_BUILD_SCV_NEW_CC,
    ACTION_MINE_MINERALS_LEFT,
    ACTION_MINE_MINERALS_RIGHT
    # ACTION_BUILD_REFINERY_TOP_LEFT,
    # ACTION_BUILD_REFINERY_TOP_RIGHT,
    # ACTION_BUILD_REFINERY_BOTTOM_LEFT,
    # ACTION_BUILD_REFINERY_BOTTOM_RIGHT,
    # ACTION_MINE_GAS_TOP_LEFT,
    # ACTION_MINE_GAS_TOP_RIGHT,
    # ACTION_MINE_GAS_BOTTOM_LEFT,
    # ACTION_MINE_GAS_BOTTOM_RIGHT
]

DATA_FILE = 'CollectMineralsAndGasAgent_data'


def print_screen_to_file(obs, filename='/home/kenn/Development/sc2-bot/CustomAgents/screen_write.txt'):
    with open(filename, 'w+') as f:
        for layer in obs.observation['screen']:
            for row in layer:
                for value in row:
                    f.write(str(value) + ',')
                f.write('\n')
            f.write('\n')


def get_smart_action(action_id):
    return smart_actions[action_id]


def get_action_id(smart_action):
    return smart_actions.index(smart_action)


def select_control_group(group_id):
    return actions.FunctionCall(_SELECT_CONTROL_GROUP, [[_CONTROL_GROUP_RECALL], [group_id]])


def set_control_group(group_id):
    return actions.FunctionCall(_SELECT_CONTROL_GROUP, [[_CONTROL_GROUP_SET], [group_id]])


def set_supply_depot_pixels(value):
    if SUPPLY_DEPOT_PIXELS == 0:
        global SUPPLY_DEPOT_PIXELS
        SUPPLY_DEPOT_PIXELS = value


def set_command_center_pixels(value):
    if COMMAND_CENTER_PIXELS == 0:
        global COMMAND_CENTER_PIXELS
        COMMAND_CENTER_PIXELS = value


def set_scv_pixels(value):
    if SCV_PIXELS == 0:
        global SCV_PIXELS
        SCV_PIXELS = value


def set_start_command_center_xy(xs, ys):
    if start_cc is None:
        global start_cc
        start_cc = (xs.mean(), ys.mean())


def get_empty_spots(unit_type):
    return (unit_type == _NOTHING).nonzero()


def get_command_centers(unit_type):
    return (unit_type == units.TERRAN_COMMANDCENTER).nonzero()


def get_supply_depots(unit_type):
    return (unit_type == units.TERRAN_SUPPLYDEPOT).nonzero()


def get_scvs(unit_type):
    return (unit_type == units.TERRAN_SCV).nonzero()


def get_mineral_fields(unit_type):
    return (unit_type == units.NEUTRAL_MINERALFIELD).nonzero()


def get_vespene_geysers(unit_type):
    return (unit_type == units.NEUTRAL_VESPENEGEYSER).nonzero()


def get_refineries(unit_type):
    return (unit_type == units.TERRAN_REFINERY).nonzero()


def get_approx_scvs_in_rectangle_count(unit_type, upper_left, lower_right):
    scv_ys, scv_xs = get_scvs(unit_type)
    xs_in = len([x for x in scv_xs if upper_left[0] <= x <= lower_right[0]])
    ys_in = len([y for y in scv_ys if upper_left[1] <= y <= lower_right[1]])
    return max(int(xs_in / SCV_PIXELS), int(ys_in / SCV_PIXELS))


def get_supply_depot_amount(unit_type):
    supply_depot_y, supply_depot_x = get_supply_depots(unit_type)
    return 0 if SUPPLY_DEPOT_PIXELS == 0 else round(len(supply_depot_y) / SUPPLY_DEPOT_PIXELS)


def get_command_center_amount(unit_type):
    commandcenter_y, commandcenter_x = get_command_centers(unit_type)
    return 0 if COMMAND_CENTER_PIXELS == 0 else round(len(commandcenter_y) / COMMAND_CENTER_PIXELS)


def get_difference(xs1, ys1, xs2, ys2):
    points1 = [(xs1[i], ys1[i]) for i in range(0, len(ys1))]
    points2 = [(xs2[i], ys2[i]) for i in range(0, len(ys2))]
    points_diff = [pair for pair in points1 if pair not in points2]
    xs = [pair[0] for pair in points_diff]
    ys = [pair[1] for pair in points_diff]
    return np.array(ys), np.array(xs)


class CollectMineralsAndGas(base_agent.BaseAgent):

    def __init__(self):
        super(CollectMineralsAndGas, self).__init__()

        data = ['SP Att.', 'SP Att.F', '#Depots', 'Ref. Att.', 'Ref. Att.F', '#Refineries', 'CC Att.', 'CC Att.f', 'Mins', 'Gas', 'IdleSCVs', 'CCs', 'Supply', 'Score']
        with open('/home/kenn/Development/sc2-bot/CustomAgents/scores.txt', 'w+') as f:
            f.write('{0[0]:<15}{0[1]:<15}{0[2]:<15}{0[3]:<15}{0[4]:<15}{0[5]:<15}{0[6]:<15}{0[7]:<15}{0[8]:<15}{0[9]:<15}{0[10]:<15}{0[11]:<15}{0[12]:<15}{0[13]:<15}\n'.format(data))

        self.qlearn = QLearning.QLearningTable(actions=list(range(len(smart_actions))))

        self.move_number = 0

        self.previous_action = None
        self.previous_state = None
        self.unit_type = None

        self.supply_depots = 0
        self.refineries = 0

        self.previous_collected_minerals_rate = 0
        self.previous_collected_vespene_rate = 0

        self.build_supply_depot_attempts = 0
        self.build_supply_depot_attempts_failed = 0

        self.build_cc_attempts = 0
        self.build_cc_attempt_failed = 0

        self.build_refinery_attempts = 0
        self.build_refinery_attempts_failed = 0

        self.builder_iterator = 0
        self.invocations = 0

        self.initializing = 0

        if os.path.isfile(DATA_FILE + '.gz'):
            self.qlearn.q_table = pd.read_pickle(DATA_FILE + '.gz', compression='gzip')

    def step(self, obs):
        super(CollectMineralsAndGas, self).step(obs)

        self.unit_type = obs.observation['screen'][_UNIT_TYPE]

        if obs.last():
            return self.handle_last_action(obs)

        if obs.first():
            return self.handle_first_action(obs)

        if self.initializing < 2:
            return self.handle_initial_action(obs)

        if self.move_number == 0:
            self.update_buildings_built()

            supply_used = p.get_food_used(obs)
            supply_cap = p.get_food_cap(obs)

            scvs_left = get_approx_scvs_in_rectangle_count(self.unit_type, (0, 0), (41, 83))
            scvs_right = get_approx_scvs_in_rectangle_count(self.unit_type, (42, 0), (83, 83))

            current_state = np.zeros(6)
            # Available supply min to reduce state space
            current_state[0] = min(supply_cap - supply_used, 7)
            # Detect how many times we have 100 minerals up to maximum 4 times (Afford a CC) to reduce our state space
            current_state[1] = min(int(p.get_minerals(obs) / 100.0), 4)
            # Any idle workers min to reduce state space
            current_state[2] = min(p.get_idle_worker_count(obs), 3)
            # scvs left side to scvs right side difference
            scv_side_diff = scvs_left - scvs_right
            current_state[3] = max(-7, scv_side_diff) if scv_side_diff < 0 else min(8, scv_side_diff)
            # Number of CCs
            current_state[4] = get_command_center_amount(self.unit_type)
            # Number of supply depots
            current_state[5] = get_supply_depot_amount(self.unit_type)

            if self.previous_action is not None:
                # reward = self.get_reward(obs)
                self.qlearn.learn(str(self.previous_state), self.previous_action, 0, str(current_state))

            disallowed_actions = self.get_disallowed_actions(obs)

            rl_action = self.qlearn.choose_action(str(current_state), disallowed_actions)

            action_name = get_smart_action(int(rl_action))
            action_function, actual_action = self.apply_action(obs, action_name)

            action_id = rl_action
            if actual_action is None:
                action_function = actions.FunctionCall(_NOOP, [])
                action_id = get_action_id(ACTION_DO_NOTHING)

            self.previous_state = current_state
            self.previous_action = action_id

            self.move_number += 1
            return action_function
        elif self.move_number == 1:
            action_name = get_smart_action(self.previous_action)
            action_function, adjusted_action = self.apply_action(obs, action_name)

            if adjusted_action is None:
                action_function = actions.FunctionCall(_NOOP, [])
                self.previous_action = get_action_id(ACTION_DO_NOTHING)

            self.move_number += 1
            return action_function
        elif self.move_number == 2:
            action_name = get_smart_action(self.previous_action)
            action_function, adjusted_action = self.apply_action(obs, action_name)

            if adjusted_action is None:
                action_function = actions.FunctionCall(_NOOP, [])
                self.previous_action = get_action_id(ACTION_DO_NOTHING)

            self.move_number = 0
            return action_function

        return actions.FunctionCall(_NOOP, [])

    def apply_action(self, obs, action_name):
        action_function, chosen_action = self.do_nothing_action(action_name)
        if chosen_action is None:
            action_function, chosen_action = self.build_supply_depot_action_predefined(obs, action_name)
        if chosen_action is None:
            action_function, chosen_action = self.build_command_center_action(obs, action_name)
        # if chosen_action is None:
        #     action_function, chosen_action = self.build_refinery_action(obs, action_name, 0)
        # if chosen_action is None:
        #     action_function, chosen_action = self.build_refinery_action(obs, action_name, 1)
        # if chosen_action is None:
        #     action_function, chosen_action = self.build_refinery_action(obs, action_name, 2)
        # if chosen_action is None:
        #     action_function, chosen_action = self.build_refinery_action(obs, action_name, 3)
        if chosen_action is None:
            action_function, chosen_action = self.build_scv_action_cc_start(obs, action_name)
        if chosen_action is None:
            action_function, chosen_action = self.build_scv_action_cc_new(obs, action_name)
        if chosen_action is None:
            action_function, chosen_action = self.mine_minerals_left_action(obs, action_name)
        if chosen_action is None:
            action_function, chosen_action = self.mine_minerals_right_action(obs, action_name)
        # if chosen_action is None:
        #     action_function, chosen_action = self.mine_gas_action(obs, action_name, 0)
        # if chosen_action is None:
        #     action_function, chosen_action = self.mine_gas_action(obs, action_name, 1)
        # if chosen_action is None:
        #     action_function, chosen_action = self.mine_gas_action(obs, action_name, 2)
        # if chosen_action is None:
        #     action_function, chosen_action = self.mine_gas_action(obs, action_name, 3)

        return action_function, chosen_action

    def get_disallowed_actions(self, obs):
        disallowed_actions = []

        refinery_y, refinery_x = get_refineries(self.unit_type)
        geyser_top_left = geysers[0]
        if geyser_top_left[0] in refinery_x and geyser_top_left[1] in refinery_y:
            disallowed_actions.append(get_action_id(ACTION_BUILD_REFINERY_TOP_LEFT))

        geyser_top_right = geysers[1]
        if geyser_top_right[0] in refinery_x and geyser_top_right[1] in refinery_y:
            disallowed_actions.append(get_action_id(ACTION_BUILD_REFINERY_TOP_RIGHT))

        geyser_bottom_left = geysers[2]
        if geyser_bottom_left[0] in refinery_x and geyser_bottom_left[1] in refinery_y:
            disallowed_actions.append(get_action_id(ACTION_BUILD_REFINERY_BOTTOM_LEFT))

        geyser_bottom_right = geysers[3]
        if geyser_bottom_right[0] in refinery_x and geyser_bottom_right[1] in refinery_y:
            disallowed_actions.append(get_action_id(ACTION_BUILD_REFINERY_BOTTOM_RIGHT))

        if self.supply_depots >= len(depots):
            disallowed_actions.append(get_action_id(ACTION_BUILD_SUPPLY_DEPOT))

        if p.get_food_cap(obs) == p.get_food_used(obs):
            disallowed_actions.append(get_action_id(ACTION_BUILD_SCV_START_CC))
            disallowed_actions.append(get_action_id(ACTION_BUILD_SCV_NEW_CC))

        if get_command_center_amount(self.unit_type) == 2:
            disallowed_actions.append(get_action_id(ACTION_BUILD_CC))

        return disallowed_actions

    def update_buildings_built(self):
        if self.previous_action is not None:
            action_name = get_smart_action(self.previous_action)
            if action_name == ACTION_BUILD_SUPPLY_DEPOT:
                self.supply_depots += 1
            if action_name == ACTION_BUILD_REFINERY_TOP_LEFT:
                self.refineries += 1
            if action_name == ACTION_BUILD_REFINERY_TOP_RIGHT:
                self.refineries += 1
            if action_name == ACTION_BUILD_REFINERY_BOTTOM_LEFT:
                self.refineries += 1
            if action_name == ACTION_BUILD_REFINERY_BOTTOM_RIGHT:
                self.refineries += 1

    def get_reward(self, obs):
        collection_rate_mins = cs.get_collection_rate_minerals(obs)
        collection_rate_gas = cs.get_collection_rate_vespene(obs)

        reward = 0
        if collection_rate_mins > self.previous_collected_minerals_rate:
            reward += (collection_rate_mins - self.previous_collected_minerals_rate)
        if collection_rate_gas > self.previous_collected_vespene_rate:
            reward += (collection_rate_gas - self.previous_collected_vespene_rate)

        self.previous_collected_minerals_rate = collection_rate_mins
        self.previous_collected_vespene_rate = collection_rate_gas

        return reward

    def do_nothing_action(self, action):
        if action != ACTION_DO_NOTHING:
            return None, None
        return actions.FunctionCall(_NOOP, []), action

    def build_supply_depot_action_predefined(self, obs, action):
        if action != ACTION_BUILD_SUPPLY_DEPOT or self.supply_depots >= len(depots):
            return None, None

        if self.move_number == 0:
            self.build_supply_depot_attempts += 1
            action_function = select_control_group(BUILDERS_CTRL_GROUP)
            if action_function is None:
                return None, None
            else:
                return action_function, action
        elif self.move_number == 1:
            if _BUILD_SUPPLY_DEPOT in obs.observation['available_actions']:
                target = [depots[self.supply_depots][0], depots[self.supply_depots][1]]
                return actions.FunctionCall(_BUILD_SUPPLY_DEPOT, [_NOT_QUEUED, target]), action
        elif self.move_number == 2:
            action_function = self.gather_minerals(obs, left_mins)
            if action_function is None:
                return None, None
            return action_function, action

        return None, None

    def build_command_center_action(self, obs, action):
        if action != ACTION_BUILD_CC or get_command_center_amount(self.unit_type) == len(new_cc_spot) + 1:
            return None, None

        if self.move_number == 0:
            self.build_cc_attempts += 1
            action_function = select_control_group(BUILDERS_CTRL_GROUP)
            if action_function is None:
                return None, None
            return action_function, action
        elif self.move_number == 1:
            if _BUILD_COMMAND_CENTER in obs.observation['available_actions']:
                target = [new_cc_spot[0][0], new_cc_spot[0][1]]
                return actions.FunctionCall(_BUILD_COMMAND_CENTER, [_NOT_QUEUED, target]), action
            self.build_cc_attempt_failed += 1
        elif self.move_number == 2:
            action_function = self.gather_minerals(obs, left_mins)
            if action_function is None:
                return None, None
            return action_function, action

        return None, None

    def build_refinery_action(self, obs, action, choice):
        if action != ACTION_BUILD_REFINERY_TOP_LEFT and choice == 0:
            return None, None
        if action != ACTION_BUILD_REFINERY_TOP_RIGHT and choice == 1:
            return None, None
        if action != ACTION_BUILD_REFINERY_BOTTOM_LEFT and choice == 2:
            return None, None
        if action != ACTION_BUILD_REFINERY_BOTTOM_RIGHT and choice == 3:
            return None, None

        if self.move_number == 0:
            self.build_refinery_attempts += 1
            action_function = select_control_group(BUILDERS_CTRL_GROUP)
            if action_function is None:
                return None, None
            else:
                return action_function, action
        elif self.move_number == 1:
            if _BUILD_REFINERY in obs.observation['available_actions']:
                target = [geysers[choice][0], geysers[choice][1]]
                return actions.FunctionCall(_BUILD_REFINERY, [_NOT_QUEUED, target]), action
            self.build_refinery_attempts_failed += 1
        elif self.move_number == 2:
            action_function = self.gather_minerals(obs, left_mins)
            if action_function is None:
                return None, None
            return action_function, action

        return None, None

    def build_scv_action_cc_start(self, obs, action):
        if action != ACTION_BUILD_SCV_START_CC:
            return None, None

        if self.move_number == 0:
            target = [start_cc[0], start_cc[1]]
            return actions.FunctionCall(_SELECT_POINT, [_NOT_QUEUED, target]), action
        elif self.move_number == 1:
            if _TRAIN_SCV in obs.observation['available_actions']:
                return actions.FunctionCall(_TRAIN_SCV, [_QUEUED]), action
        elif self.move_number == 2:
            return actions.FunctionCall(_NOOP, []), action

        return None, None

    def build_scv_action_cc_new(self, obs, action):
        if action != ACTION_BUILD_SCV_NEW_CC or get_command_center_amount(self.unit_type) < 2:
            return None, None

        if self.move_number == 0:
            cc_y, cc_x = get_command_centers(self.unit_type)
            if new_cc_spot[0][0] in cc_x and new_cc_spot[0][1] in cc_y:
                target = [new_cc_spot[0][0], new_cc_spot[0][1]]
                return actions.FunctionCall(_SELECT_POINT, [_NOT_QUEUED, target]), action
            return None, None
        elif self.move_number == 1:
            if _TRAIN_SCV in obs.observation['available_actions']:
                return actions.FunctionCall(_TRAIN_SCV, [_QUEUED]), action
        elif self.move_number == 2:
            return actions.FunctionCall(_NOOP, []), action

        return None, None

    def mine_minerals_left_action(self, obs, action):
        if action != ACTION_MINE_MINERALS_LEFT:
            return None, None

        if self.move_number == 0:
            if _SELECT_IDLE_WORKER in obs.observation['available_actions']:
                return actions.FunctionCall(_SELECT_IDLE_WORKER, [_SELECT_ALL]), action
        elif self.move_number == 1:
            action_function = self.gather_minerals(obs, left_mins)
            if action_function is None:
                return None, None
            else:
                return action_function, action
        elif self.move_number == 2:
            return actions.FunctionCall(_NOOP, []), action

        return None, None

    def mine_minerals_right_action(self, obs, action):
        if action != ACTION_MINE_MINERALS_RIGHT:
            return None, None
        if self.move_number == 0:
            if _SELECT_IDLE_WORKER in obs.observation['available_actions']:
                return actions.FunctionCall(_SELECT_IDLE_WORKER, [_SELECT_ALL]), action
        elif self.move_number == 1:
            action_function = self.gather_minerals(obs, right_mins)
            if action_function is None:
                return None, None
            else:
                return action_function, action
        elif self.move_number == 2:
            return actions.FunctionCall(_NOOP, []), action

        return None, None

    def mine_gas_action(self, obs, action, choice):
        if action != ACTION_MINE_GAS_TOP_LEFT and choice == 0:
            return None, None
        if action != ACTION_MINE_GAS_TOP_RIGHT and choice == 1:
            return None, None
        if action != ACTION_MINE_GAS_BOTTOM_LEFT and choice == 2:
            return None, None
        if action != ACTION_MINE_GAS_BOTTOM_RIGHT and choice == 3:
            return None, None

        if self.move_number == 0:
            if _SELECT_IDLE_WORKER in obs.observation['available_actions']:
                return actions.FunctionCall(_SELECT_IDLE_WORKER, [_NOT_QUEUED]), action
        elif self.move_number == 1:
            action_function = self.gather_gas(obs, geysers[choice][0], geysers[choice][1])
            if action_function is None:
                return None, None
            else:
                return action_function, action
        elif self.move_number == 2:
            return actions.FunctionCall(_NOOP, []), action

        return None, None

    def gather_gas(self, obs, target_x, target_y):
        refinery_y, refinery_x = get_refineries(self.unit_type)
        if refinery_y.any() and target_x in refinery_x and target_y in refinery_y:
            if _HARVEST_GATHER in obs.observation['available_actions']:
                return actions.FunctionCall(_HARVEST_GATHER, [_QUEUED, [target_x, target_y]])
        return None

    def gather_minerals(self, obs, fields):
        minerals_y, minerals_x = get_mineral_fields(self.unit_type)
        field_x, field_y = random.choice(fields)
        if field_y in minerals_y and field_x in minerals_x:
            if _HARVEST_GATHER in obs.observation['available_actions']:
                target = [field_x, field_y]
                return actions.FunctionCall(_HARVEST_GATHER, [_QUEUED, target])
        return None

    def select_random_scv(self):
        scv_y, scv_x = get_scvs(self.unit_type)
        if scv_y.any():
            i = random.randint(0, len(scv_y) - 1)
            target = [scv_x[i], scv_y[i]]
            return actions.FunctionCall(_SELECT_POINT, [_NOT_QUEUED, target])
        return None

    def select_random_cc(self):
        cc_y, cc_x = get_command_centers(self.unit_type)
        if cc_y.any():
            i = random.randint(0, len(cc_y) - 1)
            target = [cc_y[i], cc_x[i]]
            return actions.FunctionCall(_SELECT_POINT, [_NOT_QUEUED, target])
        return None

    def handle_initial_action(self, obs):
        action = actions.FunctionCall(_NOOP, [])
        if self.initializing == 0:
            action = set_control_group(BUILDERS_CTRL_GROUP)
        elif self.initializing == 1:
            action = self.gather_minerals(obs, left_mins)
        self.initializing += 1
        return action

    def handle_first_action(self, obs):
        cc_y, cc_x = get_command_centers(self.unit_type)
        set_command_center_pixels(len(cc_y))
        set_start_command_center_xy(cc_y, cc_x)

        scv_y, scv_x = get_scvs(self.unit_type)
        scv_amount = p.get_food_used(obs)
        set_scv_pixels(int(len(scv_y / float(scv_amount))))

        self.invocations += 1

        return actions.FunctionCall(_SELECT_IDLE_WORKER, [_SELECT_ALL])

    def handle_last_action(self, obs):
        cm = str(p.get_minerals(obs))
        cv = str(p.get_vespene(obs))
        iwc = str(p.get_idle_worker_count(obs))
        depot_count = str(self.supply_depots)
        refinery_count = str(self.refineries)
        command_centers = str(get_command_center_amount(self.unit_type))
        food_used = str(p.get_food_used(obs))
        food_cap = str(p.get_food_cap(obs))
        sp_attempts = str(self.build_supply_depot_attempts)
        sp_attempts_f = str(self.build_supply_depot_attempts_failed)
        refinery_attempts = str(self.build_refinery_attempts)
        refinery_attempts_f = str(self.build_refinery_attempts_failed)
        cc_attempts = str(self.build_cc_attempts)
        cc_attempts_f = str(self.build_cc_attempt_failed)
        score = cs.get_score(obs)
        data = [sp_attempts, sp_attempts_f, depot_count, refinery_attempts, refinery_attempts_f, refinery_count, cc_attempts, cc_attempts_f, cm, cv, iwc, command_centers, food_used + "/" + food_cap, str(score)]
        with open('/home/kenn/Development/sc2-bot/CustomAgents/scores.txt', 'a+') as f:
            f.write('{0[0]:<15}{0[1]:<15}{0[2]:<15}{0[3]:<15}{0[4]:<15}{0[5]:<15}{0[6]:<15}{0[7]:<15}{0[8]:<15}{0[9]:<15}{0[10]:<15}{0[11]:<15}{0[12]:<15}{0[13]:<15}\n'.format(data))

        # If we score less than 4000 we are doing so poorly, we want to learn that it was very bad.
        # Symbolizes the ultimate loss.
        self.qlearn.learn(str(self.previous_state), self.previous_action, int(score) - 4000, 'terminal')

        self.previous_state = None
        self.previous_action = None
        self.move_number = 0

        self.supply_depots = 0
        self.refineries = 0
        self.builder_iterator = 0

        self.build_supply_depot_attempts = 0
        self.build_supply_depot_attempts_failed = 0
        self.build_refinery_attempts = 0
        self.build_refinery_attempts_failed = 0
        self.build_cc_attempts = 0
        self.build_cc_attempt_failed = 0

        self.initializing = 0

        self.qlearn.q_table.to_pickle(DATA_FILE + '.gz', 'gzip')

        return actions.FunctionCall(_NOOP, [])

