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
        obj.state.mass = 3
        obj.state.size = 1
        target.can_be_moved = True
        target.can_collide = True
        obj.color = 'green'
        world.special_objects.append(obj)

        # Area where object need to be delivered
        target = SpecialObject()
        target.uuid = f'o_1_target'
        target.state.mass = 1
        target.state.size = 2
        target.can_move = False
        target.can_be_moved = False
        target.can_collide = False
        target.color = 'red'
        world.special_objects.append(target)

        return world

    # reset callback function
    def reset_world(self, world):
        print('RESETING WORLD')
        buffer = int(world.state.size[0] * 0.15)
        center_p = tuple([x / 2 for x in world.state.size])
        picked_p_for_object = (random.randrange(buffer, world.state.size[0] - buffer),
                               random.randrange(buffer, world.state.size[1] - buffer))
        picked_p_for_agent = (random.randrange(picked_p_for_object[0] - buffer, picked_p_for_object[0] + buffer),
                              random.randrange(picked_p_for_object[1] - buffer, picked_p_for_object[1] + buffer))

        world.agents[0].state.pos = picked_p_for_agent
        world.special_objects[0].state.pos = picked_p_for_object
        world.special_objects[1].state.pos = center_p

    # reward callback function
    def get_reward(self, agent, world):
        if agent.uuid == 'a_0_agent':
            pushable_object = world.special_objects[0]
            target_area = world.special_objects[1]
            distance = Equations.distance(
                pushable_object.state.pos, target_area.state.pos)
            # Agents get reward base on negative distance of object to target area
            # Also if distance if very close it gets imidiete boost
            # Using negative reward in order to populate q table with data faster
            if distance < 0.2:
                world.achieved_goal = True
                return 100
            world.achieved_goal = False
            return -distance
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
