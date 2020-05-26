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

        # Agent that is gonna attemt to move object from a to b
        agent = Agent()
        agent.can_grab = False
        agent.uuid = 'a_0_agent'
        agent.view_range = np.inf
        agent.state.mass = 3
        agent.state.size = 1
        agent.color = 'blue'
        world.agents.append(agent)

        # Object to be delivered by agent to certain point
        obj = SpecialObject()
        obj.uuid = f'o_0_object'
        obj.state.mass = 1
        obj.state.size = 1
        obj.can_be_moved = True
        obj.can_collide = True
        obj.color = 'green'
        world.special_objects.append(obj)

        return world

    # reset callback function
    def reset_world(self, world):
        print('RESETING WORLD')
        center_p = tuple([x / 2 for x in world.state.size])
        object_place = center_p
        agent_place = object_place + (0, 5 * world.agents[0].state.size)
        world.agents[0].state.pos = agent_place
        world.special_objects[0].state.pos = object_place
        world.achieved_goal = False

    # reward callback function
    def get_reward(self, agent, world):
        # First stage of reward is
        reward = 0
        assert self.short_term_memory is not None
        distance_to_goal = np.sqrt(
            np.sum(np.square(agent.state.pos - world.special_objects[0].state.pos)))
        if len(self.short_term_memory) == 0:
            reward = 0
        else:
            prev_memory = self.short_term_memory.pop()
            prev_distance = prev_memory['distance']
            if distance_to_goal < prev_distance:
                reward = +1
            else:
                reward = -1
        if distance_to_goal < agent.state.size:
            world.achieved_goal = True
        self.short_term_memory.append(
            {'agent_pos': agent.state.pos, 'target_pos': world.special_objects[0].state.pos, 'distance': distance_to_goal})
        return reward

    # observation callback function
    def get_observation(self, agent, world):
        # Simple observation of all agents position
        image = ScenarioUtils.get_graphical_observation(
            agent, world, self.obs_world_shape)
        return image

    # done callback function
    def is_done(self, agent, world):
        # We are restricting number of steps in learner itself
        return world.achieved_goal
