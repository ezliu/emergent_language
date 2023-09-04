import collections
import math
import textwrap
import gym
import gym_miniworld
from gym_miniworld import entity
from gym_miniworld import miniworld
from gym_miniworld import random
import torch
import numpy as np
from PIL import Image

import meta_exploration
import render


class InstructionWrapper(meta_exploration.InstructionWrapper):
  """InstructionWrapper for OfficeEnv."""

  def __init__(self, env, exploration_trajectory, **kwargs):
    super().__init__(env, exploration_trajectory, **kwargs)
    self._step = 0

  def _instruction_observation_space(self):
    return gym.spaces.Box(np.array([0]), np.array([2]), dtype=np.int)

  def _reward(self, instruction_state, action, original_reward):
    del original_reward

    # TODO(evzliu): Probably should more gracefully grab the base_env
    reward = 0
    done = False

    for obj in self.env._base_env._office_insides:
      if self.env._base_env.near(obj) and obj.color == "blue":
        done = True
        reward = 1
        break
    return reward, done

  def _generate_instructions(self, test=False):
    return self.random.randint(2, size=(1,))

  def render(self):
    image = self.env.render()
    image.write_text("Instructions: {}".format(self._current_instructions))
    return image


class BigKey(entity.Key):
  """A key with a bigger size."""

  def __init__(self, color, size=0.6):
    assert color in entity.COLOR_NAMES
    entity.MeshEnt.__init__(
        self,
        mesh_name='key_{}'.format(color),
        height=size,
        static=False
    )


class Door(entity.MeshEnt):
  def __init__(self, height):
    super().__init__("other_door", height, False)


class Desk(entity.MeshEnt):
  def __init__(self, height):
    super().__init__("office_desk", height, False)


class Chair(entity.MeshEnt):
  def __init__(self, height):
    super().__init__("office_chair", height, False)


class Couch(entity.MeshEnt):
  def __init__(self, height):
    super().__init__("couch", height, False)


class OfficeEnv(miniworld.MiniWorldEnv):
  """The sign environment from IMPORT.

  Touching either the red or blue box ends the episode.
  If the box corresponding to the color_index is touched, reward = +1
  If the wrong box is touched, reward = -1.

  The sign behind the wall either says "blue" or "red"
  """

  def __init__(self, size=12, max_episode_steps=30, color_index=0):
    params = gym_miniworld.params.DEFAULT_PARAMS.no_random()
    params.set('forward_step', 0.7)
    params.set('forward_drift', 0)
    params.set('turn_step', 30)  # 30 degree rotation

    self._size = size
    self._color_index = color_index
    self._num_offices = 4

    super().__init__(
        params=params, max_episode_steps=max_episode_steps, domain_rand=False)

    # Allow for left / right / forward + custom open door action
    self.action_space = gym.spaces.Discrete(self.actions.move_forward + 2)

  def set_color_index(self, color_index):
    self._color_index = color_index

  def _gen_world(self):
    num_offices = self._num_offices
    office_width = self._size / (num_offices + 1)

    mid_room = self.add_rect_room(
        min_x=self._size / 5 + self._size / 10, max_x=4 * self._size / 5, min_z=0,
        max_z=self._size,
        wall_tex="green_stucco", ceil_tex="ceiling_tile_noborder",
        floor_tex="wood")
    sign_room = self.add_rect_room(
        min_x=self._size * 0.9, max_x=self._size * 1.55, min_z=0,
        max_z=self._size,
        wall_tex="green_stucco", ceil_tex="ceiling_tile_noborder",
        floor_tex="wood")
    self.connect_rooms(
        mid_room, sign_room, min_z=self._size / 3, max_z=self._size / 3 * 2)

    for office_index in range(num_offices):
      start = office_index * self._size / num_offices
      office = self.add_rect_room(
        min_x=0, max_x=self._size / 5, min_z=start, max_z=start + office_width,
        wall_tex="brick_wall", ceil_tex="ceiling_tile_noborder",
        floor_tex="wood")
      door_width = office_width * 3 / 7
      self.connect_rooms(office, mid_room, min_z=start + office_width / 2 -
          door_width, max_z=start + office_width / 2 + door_width)

    colors = ["red"] * num_offices
    colors[num_offices - 1 - self._color_index] = "blue"
    self._office_insides = []
    self._doors = []
    for office_index, color in enumerate(colors):
      door = Door(2.9)
      pos = (self._size / 5 + 1, 0,
             office_index * self._size / num_offices + office_width / 2)
      self.place_entity(door, pos=pos, dir=math.pi / 2)
      self._doors.append(door)
      self._office_insides.append(self.place_entity(
          gym_miniworld.entity.Box(color=color), pos=(self._size / 10, 0,
              office_index * self._size / num_offices + office_width / 2)))

    self.place_entity(Chair(1.7), pos=(3 * self._size / 5 + 1, 0,
        self._size * 8 / 10), dir=math.pi)
    self.place_entity(Desk(2), pos=(3 * self._size / 5 + 1, 0,
        self._size * 9 / 10), dir=math.pi)
    self.place_entity(Couch(1.3), pos=(3 * self._size / 5 + 1.5, 0,
        self._size * 0.15), dir=math.pi / 2)
    self._sign_door = Door(5)
    self.place_entity(
        self._sign_door, pos=(self._size * 0.9, 0, self._size / 2),
        dir=math.pi / 2)

    def ordinal(num):
      suffix = collections.defaultdict(
          lambda: "th", {1: "first", 2: "second", 3: "third", 4: "fourth"})
      return f"{suffix[num]}"

    num_steps = np.random.randint(2)
    if num_steps == 0:
      text = f"{ordinal(self._color_index + 1)} room"
    else:
      permitted_deltas = []
      for delta in [-1, 1]:
        result = self._color_index + delta
        if 0 <= result < num_offices:
          permitted_deltas.append(delta)

      chosen_delta = np.random.choice(permitted_deltas)
      relative_desc = {-1: "right of", 1: "left of"}[chosen_delta]
      text = f"{relative_desc} {ordinal(self._color_index + 1 + chosen_delta)} room"

    lines = textwrap.wrap(text, 12, break_long_words=False)
    for i, line in enumerate(lines):
      sign = gym_miniworld.entity.TextFrame(
          pos=[self._size * 1.5, 1.8 - i, self._size * 0.5],
          dir=math.pi,
          str=line,
          height=1,
      )
      self.entities.append(sign)
    self.place_agent(min_x=6, max_x=9, min_z=4, max_z=6)

  def step(self, action):
    if action == self.actions.move_forward + 1:  # custom open door action
      for door in self._doors + [self._sign_door]:
        if self.near(door) and door in self.entities:
          self.entities.remove(door)

    obs, reward, done, info = super().step(action)
    return obs, reward, done, info


# From:
# https://github.com/maximecb/gym-miniworld/blob/master/pytorch-a2c-ppo-acktr/envs.py
class TransposeImage(gym.ObservationWrapper):
    def __init__(self, env=None):
        super(TransposeImage, self).__init__(env)
        obs_shape = self.observation_space.shape
        self.observation_space = gym.spaces.Box(
            self.observation_space.low[0, 0, 0],
            self.observation_space.high[0, 0, 0],
            [obs_shape[2], obs_shape[1], obs_shape[0]],
            dtype=self.observation_space.dtype)

    def observation(self, observation):
        return observation.transpose(2, 1, 0)


class MiniWorldOffice(meta_exploration.MetaExplorationEnv):
  """Wrapper around the gym-miniworld Maze conforming to the MetaExplorationEnv
  interface.
  """
  def __init__(self, env_id, wrapper):
    super().__init__(env_id, wrapper)
    self._base_env = OfficeEnv()
    self._env = TransposeImage(self._base_env)
    self.observation_space = gym.spaces.Dict({
      "observation": self._env.observation_space,
      "env_id": gym.spaces.Box(np.array([0]), np.array([3]), dtype=np.int)
    })
    self.action_space = self._env.action_space

  @classmethod
  def instruction_wrapper(cls):
    return InstructionWrapper

  @classmethod
  def configure(self, config):
    pass

  # Grab instance of env and modify it, to prevent creating many envs, which
  # causes memory issues.
  @classmethod
  def create_env(cls, seed, test=False, wrapper=None):
    if wrapper is None:
      wrapper = lambda state: torch.tensor(state)

    random = np.random.RandomState(seed)
    train_ids, test_ids = cls.env_ids()
    to_sample = test_ids if test else train_ids
    env_id = to_sample[random.randint(len(to_sample))]
    office_instance._env_id = env_id
    office_instance._wrapper = wrapper
    return office_instance

  def _step(self, action):
    return self._env.step(action)

  def _reset(self):
    # Don't set the seed, otherwise can cheat from initial camera angle position!
    self._env.set_color_index(self.env_id)
    return self._env.reset()

  @classmethod
  def env_ids(cls):
    return list(range(3)), list(range(3))

  def render(self):
    first_person_render = self._base_env.render(mode="rgb_array")
    top_render = self._base_env.render(mode="rgb_array", view="top")
    image = render.concatenate(
        [Image.fromarray(first_person_render), Image.fromarray(top_render)],
        "horizontal")
    image.thumbnail((320, 240))
    image = render.Render(image)
    image.write_text("Env ID: {}".format(self.env_id))
    return image


# Prevents from opening too many windows.
office_instance = MiniWorldOffice(0, None)
