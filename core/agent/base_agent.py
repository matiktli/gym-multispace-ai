from abc import abstractmethod


class BaseAgent():

    def __init__(self):
        self.is_learning = True
        pass

    @abstractmethod
    def make_decission(self, state):
        pass


# Exploration rate linear function
class LinearSchedule:
    def __init__(self, schedule_timesteps, final_p, initial_p=1.0):
        self.schedule_timesteps = schedule_timesteps
        self.final_p = final_p
        self.initial_p = initial_p

    def value(self, t):
        fraction = min(float(t)/self.schedule_timesteps, 1.0)
        return self.initial_p + fraction * (self.final_p - self.initial_p)
