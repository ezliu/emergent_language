import collections
import pathlib
import textwrap

import cv2
cv2.ocl.setUseOpenCL(False)
from gym import spaces
from gym_minigrid import minigrid
import numpy as np
from PIL import Image, ImageDraw
import torch

import meta_exploration
import render


class Map(minigrid.Ball):
  """Marker object that should trigger the agent seeing top-down view if on
  top of it.
  """

  def can_overlap(self):
    return True

  def can_pickup(self):
    return False


class Office(minigrid.Floor):
  """Marker object for an office."""


class MapMazeEnv(minigrid.MiniGridEnv):
  TILE_SIZE = 8  # Number of pixels per tile image for rendering

  # Maps (seed, class) to structured ID for better embeddings
  IDX_TO_STRUCTURED_ID = {}

  def __init__(self, seed=0, width=13, height=11, max_steps=30, test=False):
    self._seed = seed
    self._test = test
    super().__init__(
        # Dummy mission space
        mission_space=minigrid.MissionSpace(mission_func=lambda: "reach goal"),
        width=width,
        height=height,
        max_steps=max_steps,
    )
    self.observation_space["map"] = spaces.Box(
        low=0,
        high=255,
        shape=(84, 84, 1),
        dtype="uint8",
    )
    self.observation_space["map_on"] = spaces.Discrete(1)

  def _create_grid(self, width, height, rng):
    """Creates the grid layout and returns it, as well as the locations of the
    offices and doors.
    """
    # TODO(evzliu): Could do all the randomness in the init
    self.colors_to_loc = collections.defaultdict(list)

    self.grid = minigrid.Grid(width, height)
    self.grid.horz_wall(0, 0)
    self.grid.horz_wall(0, height - 1)
    self.grid.vert_wall(0, 0)
    self.grid.vert_wall(width - 1, 0)

    # Left of first column
    self.grid.wall_rect(1, 2, 2, 1)
    self.grid.wall_rect(1, 4, 2, 1)
    self.grid.wall_rect(1, 6, 2, 1)

    # In between first and second columns
    self.grid.wall_rect(4, 2, 5, 1)
    self.grid.wall_rect(4, 4, 5, 1)
    self.grid.wall_rect(4, 6, 5, 1)
    self.grid.wall_rect(6, 1, 1, 6)

    # Right of second column
    self.grid.wall_rect(10, 2, 2, 1)
    self.grid.wall_rect(10, 4, 2, 1)
    self.grid.wall_rect(10, 6, 2, 1)

    # Map portion
    self.grid.wall_rect(1, 8, 5, 2)
    self.grid.wall_rect(7, 8, 5, 2)

    self.place_agent(top=(1, 7), size=(1, 1))
    self.place_obj(Map(), top=(6, 9), size=(1, 1), max_tries=1)
    self.agent_dir = 0

    locations = [
        (1, 1), (1, 3), (1, 5),
        (5, 1), (5, 3), (5, 5),
        (7, 1), (7, 3), (7, 5),
        (11, 1), (11, 3), (11, 5)]

    door_locations = [
        (2, 1), (2, 3), (2, 5),
        (4, 1), (4, 3), (4, 5),
        (8, 1), (8, 3), (8, 5),
        (10, 1), (10, 3), (10, 5)]

    return locations, door_locations

  def _gen_grid(self, width, height):
    rng = np.random.RandomState(self._seed)
    # TODO(evzliu): Could do all the randomness in the init

    locations, door_locations = self._create_grid(
        self.width, self.height, rng)

    # Place an office at each location, at least coloring the full span of
    # colors
    self.colors_to_loc = collections.defaultdict(list)
    offices = [Office(color) for color in minigrid.COLOR_NAMES]
    offices += [
        Office(rng.choice(minigrid.COLOR_NAMES))
        for _ in range(len(locations) - len(offices))]
    rng.shuffle(offices)
    for office, location in zip(offices, locations):
      self.place_obj(office, top=location, size=(1, 1), max_tries=1)
      self.colors_to_loc[office.color].append(location)

    for loc in door_locations:
      self.place_obj(
          minigrid.Door(color="red"), top=loc, size=(1, 1), max_tries=1)

    #render_type = "pil" if self._seed % 2 else "default"
    render_type = "pil"
    #self._map_view = MapRender(
    #    self, self.colors_to_loc, self._seed % 10, render_type)
    # TODO(evzliu): Make an enum for this
    #if self._test:
    #  self._map_view = CachedMapRender(
    #      self._seed, self.colors_to_loc, np.random.randint(90, 100))
    #else:
    #  self._map_view = CachedMapRender(
    #      self._seed, self.colors_to_loc, np.random.randint(90))
    # TODO(evzliu): This is called every reset call, so new
    # episodes will have different map types.
    #self._map_view = CachedMapRender(self._seed, self.colors_to_loc, "language")

    # Language map
    #self._map_view = CachedMapRender(
    #    self._seed, False, self.shortest_path, self.colors_to_loc, "language")

    # Picture map
    if self._test:
      self._map_view = CachedMapRender(
          self._seed, type(self), self.shortest_path, self.colors_to_loc,
          np.random.randint(90, 100))
    else:
      self._map_view = CachedMapRender(
          self._seed, type(self), self.shortest_path, self.colors_to_loc,
          np.random.randint(90))

    #self._map_view = CachedMapRender(
    #    self._seed, locations, self._large, self.colors_to_loc, "language")

    # Update structured IDs
    structured_id = tuple(
        minigrid.COLOR_TO_IDX[office.color] for office in offices)
    self.IDX_TO_STRUCTURED_ID[(self._seed, type(self))] = structured_id

  def gen_obs(self):
    obs = super().gen_obs()
    obs["map"] = self.map_view()[0]
    obs["map_on"] = self.map_view()[1]
    return obs

  def pad_id(self, structured_id):
    """Pads a structured ID to be the length of large layout IDs."""
    delta = self.num_offices(True) - len(structured_id)
    return structured_id + (len(minigrid.COLORS),) * delta

  def map_view(self):
    if isinstance(self.grid.get(*self.agent_pos), Map):
      obs = self._map_view.render()
      # From gym baselines conversion to grayscale
      obs = cv2.resize(obs, (84, 84), interpolation=cv2.INTER_AREA)
      return obs, True
    return np.zeros((84, 84, 3), dtype=np.uint8), False

  def render(self, mode="rgb_array"):
    agent_view = Image.fromarray(
        super().render(mode=mode, tile_size=15))
    map_view = Image.fromarray(np.squeeze(self.map_view()[0]))
    true_map_view = self._map_view.render()
    true_map_view = cv2.resize(true_map_view, (84, 84), interpolation=cv2.INTER_AREA)
    true_map_view = Image.fromarray(np.squeeze(true_map_view))
    map_views = render.concatenate([map_view, true_map_view], "vertical")
    image = render.concatenate([agent_view, map_views], "horizontal")
    return render.Render(image, banner_size=100)

  @classmethod
  def num_offices(cls):
    # TODO(evzliu): Could avoid hard-coding this by making the locations a
    # property
    return 12

  def path_to_map(self):
    forward = 5
    return ([MapMazeEnv.Actions.forward] * forward +
            [MapMazeEnv.Actions.right] +
            [MapMazeEnv.Actions.forward] * (self.max_steps - 6))

  @staticmethod
  def shortest_path(dst, hallway_length=7):
    path = []
    if dst[0] <= 5:
      path += [MapMazeEnv.Actions.forward] * 2
    elif dst[0] <= 11:
      path += [MapMazeEnv.Actions.forward] * 8
    else:
      path += [MapMazeEnv.Actions.forward] * 14

    path += [MapMazeEnv.Actions.left]
    path += [MapMazeEnv.Actions.forward] * (hallway_length - dst[1])

    if dst[0] == 1 or dst[0] == 7 or dst[0] == 13:
      path += [MapMazeEnv.Actions.left]
    else:
      path += [MapMazeEnv.Actions.right]

    path += [MapMazeEnv.Actions.toggle] + [MapMazeEnv.Actions.forward] * 2
    return path


class TallSkinnyMapMazeEnv(MapMazeEnv):
  def __init__(self, seed=0, max_steps=30, test=False):
    super().__init__(
        seed=seed, width=13, height=14, max_steps=max_steps, test=test)

  def _create_grid(self, width, height, rng):
    """Creates the grid layout and returns it, as well as the locations of the
    offices and doors.
    """
    # TODO(evzliu): Could do all the randomness in the init
    self.colors_to_loc = collections.defaultdict(list)

    self.grid = minigrid.Grid(width, height)
    self.grid.horz_wall(0, 0)
    self.grid.horz_wall(0, height - 1)
    self.grid.vert_wall(0, 0)
    self.grid.vert_wall(width - 1, 0)

    # Left of first column
    self.grid.wall_rect(1, 2, 2, 2)
    self.grid.wall_rect(1, 5, 2, 2)
    self.grid.wall_rect(1, 8, 2, 2)

    # In between first and second columns
    self.grid.wall_rect(4, 2, 5, 2)
    self.grid.wall_rect(4, 5, 5, 2)
    self.grid.wall_rect(4, 8, 5, 2)
    self.grid.wall_rect(6, 1, 1, 9)

    # Right of second column
    self.grid.wall_rect(10, 2, 2, 2)
    self.grid.wall_rect(10, 5, 2, 2)
    self.grid.wall_rect(10, 8, 2, 2)

    # Map portion
    self.grid.wall_rect(1, 11, 5, 2)
    self.grid.wall_rect(7, 11, 5, 2)

    self.place_agent(top=(1, 10), size=(1, 1))
    self.place_obj(Map(), top=(6, 12), size=(1, 1), max_tries=1)
    self.agent_dir = 0

    locations = [
        (1, 1), (1, 4), (1, 7),
        (5, 1), (5, 4), (5, 7),
        (7, 1), (7, 4), (7, 7),
        (11, 1), (11, 4), (11, 7) ]

    door_locations = [
        (2, 1), (2, 4), (2, 7),
        (4, 1), (4, 4), (4, 7),
        (8, 1), (8, 4), (8, 7),
        (10, 1), (10, 4), (10, 7)]
    return locations, door_locations

  @classmethod
  def num_offices(cls):
    return 12

  @staticmethod
  def shortest_path(dst):
    return super(TallSkinnyMapMazeEnv, TallSkinnyMapMazeEnv).shortest_path(
        dst, hallway_length=10)


class TallMapMazeEnv(MapMazeEnv):
  def __init__(self, seed=0, max_steps=30, test=False):
    super().__init__(
        seed=seed, width=19, height=14, max_steps=max_steps, test=test)

  def _create_grid(self, width, height, rng):
    """Creates the grid layout and returns it, as well as the locations of the
    offices and doors.
    """
    # TODO(evzliu): Could do all the randomness in the init
    self.colors_to_loc = collections.defaultdict(list)

    self.grid = minigrid.Grid(width, height)
    self.grid.horz_wall(0, 0)
    self.grid.horz_wall(0, height - 1)
    self.grid.vert_wall(0, 0)
    self.grid.vert_wall(width - 1, 0)

    # Left of first column
    self.grid.wall_rect(1, 2, 2, 2)
    self.grid.wall_rect(1, 5, 2, 2)
    self.grid.wall_rect(1, 8, 2, 2)

    # In between first and second columns
    self.grid.wall_rect(4, 2, 5, 2)
    self.grid.wall_rect(4, 5, 5, 2)
    self.grid.wall_rect(4, 8, 5, 2)
    self.grid.wall_rect(6, 1, 1, 9)

    # In between second and third columns
    self.grid.wall_rect(10, 2, 5, 2)
    self.grid.wall_rect(10, 5, 5, 2)
    self.grid.wall_rect(10, 8, 5, 2)
    self.grid.wall_rect(12, 1, 1, 9)

    # Right of third column
    self.grid.wall_rect(16, 2, 2, 2)
    self.grid.wall_rect(16, 5, 2, 2)
    self.grid.wall_rect(16, 8, 2, 2)

    # Map portion
    self.grid.wall_rect(1, 11, 5, 2)
    self.grid.wall_rect(7, 11, 11, 2)

    self.place_agent(top=(1, 10), size=(1, 1))
    self.place_obj(Map(), top=(6, 12), size=(1, 1), max_tries=1)
    self.agent_dir = 0

    locations = [
        (1, 1), (1, 4), (1, 7),
        (5, 1), (5, 4), (5, 7),
        (7, 1), (7, 4), (7, 7),
        (11, 1), (11, 4), (11, 7),
        (13, 1), (13, 4), (13, 7),
        (17, 1), (17, 4), (17, 7)]

    door_locations = [
        (2, 1), (2, 4), (2, 7),
        (4, 1), (4, 4), (4, 7),
        (8, 1), (8, 4), (8, 7),
        (10, 1), (10, 4), (10, 7),
        (14, 1), (14, 4), (14, 7),
        (16, 1), (16, 4), (16, 7)]
    return locations, door_locations

  @classmethod
  def num_offices(cls):
    return 18

  @staticmethod
  def shortest_path(dst):
    return super(TallMapMazeEnv, TallMapMazeEnv).shortest_path(
        dst, hallway_length=10)


class CachedMapRender:
  """Loads from, rather than generating style transfer on the fly."""

  # TODO(evzliu): Make subclasses rather than multiple transfer types
  def __init__(
      self, env_id, env_type, shortest_path, color_to_loc, transfer_type):
    self._env_id = env_id
    self._env_type = env_type
    self._shortest_path = shortest_path
    self._cached_img = None
    self._transfer_type = transfer_type
    self._colors_to_loc = color_to_loc

  def render(self):

    if self._cached_img is not None:
      return self._cached_img

    if self._transfer_type == "language":
      goal_color = minigrid.COLOR_NAMES[0]
      locations = self._colors_to_loc[goal_color]
      goal_loc = min(
          locations, key=lambda loc: len(self._shortest_path(loc)))

      # TODO(evzliu): Hard-coded
      col = {1: 0, 5: 1, 7: 2, 11: 3, 13: 4, 17: 5}[goal_loc[0]]
      row = {1: 0, 3: 1, 4: 1, 5: 2, 7: 2}[goal_loc[1]]
      locations = sum(self._colors_to_loc.values(), [])
      max_row = max(loc[0] for loc in locations)
      max_col = max(loc[1] for loc in locations)

      # TODO(evzliu): Extraneous np.random.randint() here, but makes it
      # pseudo-deterministic if you seed the np.random.
      desc = LocationDescription.generate(
          col, max_col, row, max_row,
          np.random.RandomState(np.random.randint(np.iinfo(np.int32).max)),
          np.random.randint(2))
      self._desc = desc
      rendering = Image.new("RGB", (84, 84), (255, 255, 255))
      draw = ImageDraw.Draw(rendering)
      for i, line in enumerate(textwrap.wrap(desc.description, 13)):
        draw.text((2, 10 * i + 2), line, (0, 0, 0))
      self._cached_img = np.array(rendering)
      return self._cached_img

    if self._transfer_type == 0:
      rendering = Image.new("RGB", (84, 84), (255, 255, 255))
      draw = ImageDraw.Draw(rendering)

      # TODO(evzliu): Need to adjust this for the large version
      for color, locs in self._colors_to_loc.items():
        for loc in locs:
          x = loc[0] * 7
          y = loc[1] * 15
          draw.text((x - 5, y - 5), color[:2], (0, 0, 0))
      self._cached_img = np.array(rendering)
      return self._cached_img

    path = f"envs/final_renders/{self._env_type.__name__}/{self._env_id}/{self._transfer_type}.png"
    self._cached_img = np.array(Image.open(path))
    return self._cached_img


class LocationDescription:
  """Compositional language description of an office location in a grid."""

  def __init__(self, description):
    self._description = description

  @property
  def description(self):
    return self._description

  @staticmethod
  def generate(col, max_col, row, max_row, rng, num_steps=2):
    excluded_utterances = [
        "3rd office in the 2nd row",
    ]

    required_utterances = []

    MAX_RETRIES = 100
    for _ in range(MAX_RETRIES):
      description = LocationDescription._generate(
          col, max_col, row, max_row, rng, num_steps=num_steps)
      keep = not any(
          utterance in description.description
          for utterance in excluded_utterances)
      keep = keep and (len(required_utterances) == 0 or any(
          utterance in description.description
          for utterance in required_utterances))
      if keep:
        return description
    return description

  @staticmethod
  def _generate(col, max_col, row, max_row, rng, num_steps=2):
    def ordinal(num):
      suffix = collections.defaultdict(
          lambda: "th", {1: "st", 2: "nd", 3: "rd"})
      return f"{num}{suffix[num]}"

    if num_steps == 0:
      rand = rng.rand()

      # Corner reference
      if ((row == 0 or row == max_row) and
          (col == 0 or col == max_col) and rand < 0.3):
        row_ref = "upper" if row == 0 else "lower"
        col_ref = "left" if col == 0 else "right"
        return LocationDescription(f"{row_ref} {col_ref} corner")

      # Perimeter reference
      if (row == 0 or row == max_row) and rand < 0.6:
        row_ref = "upper" if row == 0 else "lower"
        return LocationDescription(
            f"{ordinal(col + 1)} office in the {row_ref} row")

      #if (row == 0 or row == max_row or
      #    col == 0 or col == max_col) and rand < 0.6:
      #  if row == 0 or row == max_row:
      #    row_ref = "upper" if row == 0 else "lower"
      #    return LocationDescription(
      #        f"{ordinal(col + 1)} office in the {row_ref} row")

      #  col_ref = "left" if col == 0 else "right"
      #  return LocationDescription(
      #      f"{ordinal(row + 1)} office in the {col_ref} col")

      ## Row / col reference
      #if rng.rand() < 0.5:
        #return LocationDescription(
        #    f"{ordinal(row + 1)} office in the {ordinal(col + 1)} col")
      return LocationDescription(
          f"{ordinal(col + 1)} office in the {ordinal(row + 1)} row")

    def in_bounds(col, row):
      return 0 <= row <= max_row and 0 <= col <= max_col

    neighbors = []
    for dx in [-1, 0, 1]:
      for dy in [-1, 0, 1]:
        if dx == 0 and dy == 0:
          continue

        if not in_bounds(col + dx, row + dy):
          continue

        dx_desc = {-1: "right of", 1: "left of", 0: ""}[dx]
        dy_desc = {-1: "below", 1: "above", 0: ""}[dy]
        office_ref = " office" if num_steps > 1 else ""
        in_between = " and " if (dx != 0 and dy != 0) else ""
        relative_desc = f"{dx_desc}{in_between}{dy_desc} the{office_ref}"
        neighbors.append(((col + dx, row + dy), relative_desc))
    neighbor, relative_desc = neighbors[rng.randint(len(neighbors))]
    neighbor_desc = LocationDescription._generate(
        neighbor[0], max_col, neighbor[1], max_row, rng, num_steps - 1)
    return LocationDescription(f"{relative_desc} {neighbor_desc.description}")

  def __str__(self):
    return f"{str(type(self))}({self.description})"


class InstructionWrapper(meta_exploration.InstructionWrapper):
  """Instruction wrapper for MapMazeEnv.

  Instructions are to visit a particular-colored office.

  The reward function for a given color c is +1 for visiting an office of that
  color and 0 otherwise.
  """

  def _instruction_observation_space(self):
    return spaces.Box(
        np.array([0]), np.array([len(minigrid.COLOR_NAMES)]), dtype=np.int32)

  def _reward(self, instruction_state, action, original_reward):
    del original_reward

    done = False
    reward = -0.05
    goal_color = minigrid.COLOR_NAMES[instruction_state.instructions[0]]
    # TODO(evzliu): Probably shouldn't touch this private var
    obj = self.env._env.grid.get(*self.env._env.agent_pos)
    if obj and isinstance(obj, Office) and obj.color == goal_color:
      reward = 1
      done = True
    # TODO(evzliu): Penalize per timestep + going to wrong office?
    return reward, done

  def _generate_instructions(self, test=False):
    #return self.random.randint(len(minigrid.COLOR_NAMES), size=(1,))
    return np.zeros((1,), dtype=np.int32)

  def render(self):
    image = self.env.render()
    goal_color = minigrid.COLOR_NAMES[self._current_instructions[0]]
    image.write_text(
        f"Instructions: ({self._current_instructions}, {goal_color})")
    return image


class MapMetaEnv(meta_exploration.MetaExplorationEnv):
  """Wraps MapMazeEnv inside a MetaExplorationEnv interface."""
  ENV_ID_SPACE = spaces.Discrete(200)
  ALLOW_LARGE = False

  def __init__(self, env_id, wrapper, test=False):
    super().__init__(env_id, wrapper)
    self._env = self._id_to_layout(env_id)(seed=env_id, test=test)
    self.observation_space = self._env.observation_space
    self.observation_space["observation"] = self.observation_space["image"]
    self.observation_space["env_id"] = spaces.Box(
        np.array([0]), np.array([self.ENV_ID_SPACE.n]), dtype=np.int32)
    self.action_space = self._env.action_space

  def _allowed_layouts(self):
    allowed_layouts = [MapMazeEnv]
    if self.ALLOW_LARGE:
      allowed_layouts.append(TallMapMazeEnv)
    return allowed_layouts

  def _id_to_layout(self, env_id):
    if not self.ALLOW_LARGE:
      return MapMazeEnv
    allowed_layouts = self._allowed_layouts()
    return allowed_layouts[env_id % len(allowed_layouts)]

  @classmethod
  def instruction_wrapper(cls):
    # TODO(evzliu): This could probably be a staticmethod?
    return InstructionWrapper

  @property
  def steps_remaining(self):
      return self._env.steps_remaining

  @classmethod
  def create_env(cls, seed, test=False, wrapper=None):
    # Change default wrapper to handle dict states
    if wrapper is None:
      wrapper = (
          lambda state: {k: torch.tensor(state[k])
                         for k in ["image", "map", "map_on"]})

    random = np.random.RandomState(seed)
    train_ids, test_ids = cls.env_ids()
    split = test_ids if test else train_ids
    env_id = split[random.randint(len(split))]
    return cls(env_id, wrapper, test=test)

  def _step(self, action):
    return self._env.step(action)

  def _reset(self):
    return self._env.reset()

  @classmethod
  def env_ids(cls):
    ids = list(range(cls.ENV_ID_SPACE.n))
    return ids, ids

  def render(self):
    image = self._env.render()
    image.write_text("Env ID: {}".format(self.env_id))
    return image

  @classmethod
  def configure(cls, config):
    cls.ALLOW_LARGE = config.get("allow_large")


class StructuredMapMetaEnv(MapMetaEnv):
  """MapMetaEnv with structured IDs."""

  def __init__(self, env_id, wrapper, test=False):
    super().__init__(env_id, wrapper, test=test)
    num_offices = max(
        layout.num_offices() for layout in self._allowed_layouts())
    self.observation_space["env_id"] = spaces.Box(
        np.array([0] * num_offices),
        np.array([len(minigrid.COLORS) + 1] * num_offices), dtype=np.int32)

  @property
  def env_id(self):
    unstructured_id = super().env_id
    layout = self._id_to_layout(unstructured_id)
    structured_id = self._env.IDX_TO_STRUCTURED_ID[(unstructured_id, layout)]

    # pad id if multiple layouts give different sizes
    max_size = self.observation_space["env_id"].low.shape[0]
    delta = max_size - len(structured_id)
    return structured_id + (len(minigrid.COLORS),) * delta
