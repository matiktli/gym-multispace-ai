from gym_multispace.scenario import BaseScenario, ScenarioUtils
from gym_multispace.core.entity import Agent, SpecialObject
from gym_multispace.core.world import World
from gym_multispace.renderer import Scaler, CircleVisualObject
from collections import deque
import numpy as np
import random
import cv2


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

        # Create attacker agent
        agent = Agent()
        agent.can_grab = False
        agent.uuid = 'a_0_agent'
        agent.view_range = np.inf
        agent.state.mass = 1
        agent.state.size = 1
        agent.color = 'red'
        world.agents.append(agent)

        # Setup objective in world
        goal = SpecialObject()
        goal.uuid = f'o_0_goal'
        goal.state.mass = 1
        goal.state.size = 1
        world.special_objects.append(goal)

        return world

    # reset callback function
    def reset_world(self, world):
        print('RESETING WORLD')
        # Place attacker in the center
        center_p = tuple([x / 2 for x in world.state.size])
        world.agents[0].state.pos = (
            center_p[0], center_p[1])

        # Place goal in the right area
        world.special_objects[0].state.pos = (random.randrange(
            0, world.state.size[0]), random.randrange(0, world.state.size[1]))

        world.achieved_goal = False

    # reward callback function
    def get_reward(self, agent, world):
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
        image = ScenarioUtils.get_graphical_observation(
            agent, world, self.obs_world_shape)
        return image

    # done callback function
    def is_done(self, agent, world):
        # We are restricting number of steps in learner itself
        return world.achieved_goal
