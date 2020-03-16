from gym_multispace.scenario import BaseScenario
from gym_multispace.core.entity import Agent, SpecialObject
from gym_multispace.core.world import World
from gym_multispace.core.world_engine import Equations
import numpy as np
import random


# Test scenario
class Scenario(BaseScenario):

    def generate_world(self):
        print('GENERATING WORLD')
        world = World()
        world.state.size = (20, 20)
        world.is_reward_shared = False
        world.is_discrete = True
        world.agents = []
        world.special_objects = []

        # Agent that gonna be a hunter and need to catch the pray
        hunter = Agent()
        hunter.can_grab = False
        hunter.uuid = 'a_0_hunter'
        hunter.view_range = np.inf
        hunter.state.mass = 3
        hunter.state.size = 1
        hunter.color = 'blue'
        world.agents.append(hunter)

        # Agent that gonna be a pray and need to escape the hunter
        pray = Agent()
        pray.can_grab = False
        pray.uuid = 'a_1_pray'
        pray.view_range = np.inf
        pray.state.mass = 5
        pray.state.size = 1
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
        world.agents[0].state.pos = picked_p_for_pray
        world.special_objects[0].state.pos = center_p

    # reward callback function
    def get_reward(self, agent, world):
        hunter = world.agents[0]
        pray = world.agents[1]
        distance = Equations.distance(hunter.state.pos, pray.state.pos)

        if agent.uuid == 'a_0_hunter':
            if distance < (hunter.state.size + pray.state.size + 0.2):
                world.achieved_goal = True
                return 100
            world.achieved_goal = False
            # Hunter receives reward base on negative distance ot the pray
            return -distance
        if agent.uuid == 'a_1_pray':
            if distance < (hunter.state.size + pray.state.size + 0.2):
                world.achieved_goal = True
                return -100
            world.achieved_goal = False
            # Pray receives reward base on distance to hunter
            return distance
        else:
            raise Exception('Wrong agent definition')

    # observation callback function
    def get_observation(self, agent, world):
        print(f'GETTING OBS FOR AGENT: {agent.uuid}.')
        # Simple observation of all agents position
        obs = []
        for obj in world.objects_all:
            obs.append(obj.state.pos)
            obs.append(obj.state.vel)
        return np.concatenate(obs)

    # done callback function
    def is_done(self, agent, world):
        # We are restricting number of steps in learner itself
        return world.achieved_goal
