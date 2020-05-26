from gym_multispace.scenario import BaseScenario
from gym_multispace.core.entity import Agent, SpecialObject
from gym_multispace.core.world import World
import numpy as np
import random


# Test scenario
class Scenario(BaseScenario):

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
        agent.state.mass = 3
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

    # reward callback function
    def get_reward(self, agent, world):
        if agent.uuid == 'a_0_agent':
            # Attacker reward is based on distance to the target entity (negative reward)
            distance_to_goal = np.sqrt(
                np.sum(np.square(agent.state.pos - world.special_objects[0].state.pos)))
            return -distance_to_goal
        else:
            raise Exception('Wrong agent definition')

    # observation callback function
    def get_observation(self, agent, world):
        # Simple observation of all agents position
        obs_pos = [ag.state.pos for ag in world.objects_all]
        obs_vel = [ag.state.vel for ag in world.objects_all]
        obs_mass = [ (ag.state.mass,0) for ag in world.objects_all]
        return np.concatenate((obs_pos, obs_vel, obs_mass))

    # done callback function
    def is_done(self, agent, world):
        # We are restricting number of steps in learner itself
        return False
