from pysc2.lib import features

_SCORE_CUMULATIVE = 'score_cumulative'
# score_cumulative parameters
_SCORE = 0
_IDLE_PRODUCTION_TIME = 1
_IDLE_WORKER_TIME = 2
_COLLECTED_MINERALS = 7
_COLLECTED_VESPENE = 8
_COLLECTION_RATE_MINERALS = 9
_COLLECTION_RATE_VESPENE = 10

_PLAYER = 'player'
# player parameters
_MINERALS = 1
_VESPENE = 2
_FOOD_USED = 3
_FOOD_CAP = 4
_FOOD_ARMY = 5
_FOOD_WORKERS = 6
_IDLE_WORKER_COUNT = 7
_ARMY_COUNT = 8

_SCREEN = 'screen'
# screen parameters
_UNIT_TYPE = features.SCREEN_FEATURES.unit_type.index



"""
    FUNCTIONS TO ACCESS THE CUMULATIVE SCORES OF THE OBSERVATION
"""


class CumulativeScores(object):

    @staticmethod
    def get_score(obs):
        return obs.observation[_SCORE_CUMULATIVE][_SCORE]

    @staticmethod
    def get_collected_minerals(obs):
        return obs.observation[_SCORE_CUMULATIVE][_COLLECTED_MINERALS]

    @staticmethod
    def get_collected_vespene(obs):
        return obs.observation[_SCORE_CUMULATIVE][_COLLECTED_VESPENE]

    @staticmethod
    def get_idle_worker_time(obs):
        return obs.observation[_SCORE_CUMULATIVE][_IDLE_WORKER_TIME]

    @staticmethod
    def get_idle_production_time(obs):
        return obs.observation[_SCORE_CUMULATIVE][_IDLE_PRODUCTION_TIME]

    @staticmethod
    def get_collection_rate_minerals(obs):
        return obs.observation[_SCORE_CUMULATIVE][_COLLECTION_RATE_MINERALS]

    @staticmethod
    def get_collection_rate_vespene(obs):
        return obs.observation[_SCORE_CUMULATIVE][_COLLECTION_RATE_VESPENE]


"""
    FUNCTIONS TO ACCESS THE PLAYER DATA OF THE OBSERVATION 
"""


class Player(object):

    @staticmethod
    def get_minerals(obs):
        return obs.observation[_PLAYER][_MINERALS]

    @staticmethod
    def get_vespene(obs):
        return obs.observation[_PLAYER][_VESPENE]

    @staticmethod
    def get_idle_worker_count(obs):
        return obs.observation[_PLAYER][_IDLE_WORKER_COUNT]

    @staticmethod
    def get_food_used(obs):
        return obs.observation[_PLAYER][_FOOD_USED]

    @staticmethod
    def get_food_cap(obs):
        return obs.observation[_PLAYER][_FOOD_CAP]