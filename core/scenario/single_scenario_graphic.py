from gym_multispace.scenario import BaseScenario
from gym_multispace.core.entity import Agent, SpecialObject
from gym_multispace.core.world import World
from gym_multispace.renderer import Scaler, CircleVisualObject
import numpy as np
import random
import cv2


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
        print(f'GETTING OBS FOR AGENT: {agent.uuid}.')
        # Simple observation of all agents position
        image = 255 * np.ones((250, 250, 3), np.uint8)
        for agent in world.objects_all:
            v_obj = CircleVisualObject(
                agent.state.pos, agent.color, agent.state.size)
            # Trick to add oppacity images in cv2
            overlay = image.copy()
            overlay = v_obj.render(overlay)
            oppacity = 0.6
            image = cv2.addWeighted(overlay,
                                    oppacity,
                                    image,
                                    1 - oppacity,
                                    0)
        
        return image

    # done callback function
    def is_done(self, agent, world):
        # We are restricting number of steps in learner itself
        return False
