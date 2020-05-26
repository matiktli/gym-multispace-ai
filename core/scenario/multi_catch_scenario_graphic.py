from gym_multispace.scenario import BaseScenario, ScenarioUtils
from gym_multispace.core.entity import Agent, SpecialObject
from gym_multispace.core.world import World
from gym_multispace.core.world_engine import Equations
from collections import deque
import numpy as np
import random


# Test scenario
class Scenario(BaseScenario):

    def __init__(self):
        super().__init__()
        self.short_term_memory = deque(maxlen=1)

    def generate_world(self):
        print('GENERATING WORLD')
        world = World()
        world.is_reward_shared = False
        world.is_discrete = True
        world.agents = []
        world.special_objects = []

        # Agent that gonna be a hunter and need to catch the pray
        hunter = Agent()
        hunter.can_grab = False
        hunter.uuid = 'a_0_hunter'
        hunter.view_range = np.inf
        hunter.state.mass = 1
        hunter.state.size = 1
        hunter.color = 'blue'
        world.agents.append(hunter)

        # Agent that gonna be a pray and need to escape the hunter
        pray = Agent()
        pray.can_grab = False
        pray.uuid = 'a_1_pray'
        pray.view_range = np.inf
        pray.state.mass = 3
        pray.state.size = 1.5
        pray.color = 'red'
        world.agents.append(pray)

        # Obstacle object
        obstacle = SpecialObject()
        obstacle.uuid = f'o_0_obstacle'
        obstacle.state.mass = 1
        obstacle.state.size = 2
        obstacle.color = 'green'
        world.special_objects.append(obstacle)

        return world

    # reset callback function
    def reset_world(self, world):
        print('RESETING WORLD')
        buffer = int(world.state.size[0] * 0.15)
        center_p = tuple([x / 2 for x in world.state.size])
        picked_p_for_hunter = (random.randrange(buffer, world.state.size[0] - buffer),
                               random.randrange(buffer, world.state.size[1] - buffer))
        picked_p_for_pray = (random.randrange(buffer, world.state.size[0] - buffer),
                             random.randrange(buffer, world.state.size[1] - buffer))

        world.agents[0].state.pos = picked_p_for_hunter
        world.agents[1].state.pos = picked_p_for_pray
        world.special_objects[0].state.pos = center_p
        world.achieved_goal = False

    # reward callback function
    # reward callback function
    def get_reward(self, agent, world):
        reward = 0
        cur_distance_between = np.sqrt(
            np.sum(np.square(world.agents[0].state.pos - world.agents[1].state.pos)))
        if len(self.short_term_memory) == 0:
            reward = 0
        else:
            prev_memory = self.short_term_memory.pop()
            prev_distance = prev_memory['distance']
            if distance_between < prev_distance:
                reward = +1
            else:
                reward = -1
        if distance_between < agent.state.size:
            world.achieved_goal = True
        self.short_term_memory.append(
            {'distance': distance_between})

        if agent.uuid == 'a_1_pray':
            # If you are a prey your reward is negated
            reward *= -1
        return reward

    # observation callback function
    def get_observation(self, agent, world):
        image = ScenarioUtils.get_graphical_observation(
            agent, world, self.obs_world_shape)
        return image

    # done callback function
    def is_done(self, agent, world):
        # We are restricting number of steps in learner itself
        return world.achieved_goal
