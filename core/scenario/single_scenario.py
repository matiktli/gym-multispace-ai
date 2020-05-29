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
        agent.color = 'blue'
        world.agents.append(agent)

        # Setup objective in world
        goal = SpecialObject()
        goal.uuid = f'o_0_goal'
        goal.state.mass = 1
        goal.state.size = 1
        goal.color = 'green'
        world.special_objects.append(goal)

        # create 4obstacles that builds a wall
        for i in range(0, 7):
            wall_part_obj = SpecialObject()
            wall_part_obj.uuid = f'o_{i}_wall'
            wall_part_obj.state.mass = 1
            wall_part_obj.state.size = 1
            wall_part_obj.color = 'red'
            world.special_objects.append(wall_part_obj)

        return world

    # reset callback function
    def reset_world(self, world):
        print('RESETING WORLD')
        # Place attacker in the center
        center_p = tuple([x / 2 for x in world.state.size])
        world.agents[0].state.pos = (np.random.randint(
            4, world.state.size[0]/2-4), np.random.randint(4, world.state.size[1]/2-4))

        wall_positions = [(1.5*x + 6, world.state.size[1]/2)
                          for x in range(0, len(world.special_objects)-1)]
        counter = 0
        for sp_obj in world.special_objects:
            if 'wall' in sp_obj.uuid:
                sp_obj.state.pos = wall_positions[counter]
                counter += 1
            else:
                sp_obj.state.pos = (np.random.randint(
                    world.state.size[0]/2+4, world.state.size[0]-4), np.random.randint(world.state.size[1]/2+4, world.state.size[1]-4))

    # reward callback function
    def get_reward(self, agent, world):
        if agent.uuid == 'a_0_agent':
            # Attacker reward is based on distance to the target entity (negative reward)
            target = world.get_object_by_name('o_0_goal')
            distance_to_goal = np.sqrt(
                np.sum(np.square(agent.state.pos - target.state.pos)))
            if distance_to_goal < 2:
                # Just special state to place agent on top of object if close enough
                if target.state.pos[0] + 1.5 < agent.state.pos[0]:
                    return +100
                return + 50

            for sp_obj in world.special_objects:
                if 'wall' in sp_obj.uuid:
                    distance_to_wall_part = np.sqrt(
                        np.sum(np.square(agent.state.pos - sp_obj.state.pos)))
                    if distance_to_wall_part < 2:
                        return -100
            return -distance_to_goal
        else:
            raise Exception('Wrong agent definition')

    # observation callback function
    def get_observation(self, agent, world):
        # Simple observation of all agents position
        observation_per_agent = []
        for obj in world.objects_all:
            entity_obs = []
            entity_obs.append(obj.state.pos[0])
            entity_obs.append(obj.state.pos[1])
            entity_obs.append(obj.state.vel[0])
            entity_obs.append(obj.state.vel[1])
            # entity_obs.append(obj.state.mass)
            # entity_obs.append(obj.state.size)
            observation_per_agent.append(entity_obs)
        return np.concatenate(observation_per_agent)

    # done callback function
    def is_done(self, agent, world):
        # We are restricting number of steps in learner itself
        return False
