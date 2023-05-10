# IGLU Datasets

Interactive Grounded Language Understanding in a Collaborative Environment
--

The primary goal of the IGLU project is to approach the problem of how
to develop interactive agents that learn to solve a task while provided with grounded natural language
instructions in a collaborative environment.

Following this objective, the project has collected several datasets with different types of interactions during a block building task. The data and scripts used to collect it will be progressively released in this repository.

Due to the complexity of a general interactive task, the problem is simplified to interactions inside of a finite, Minecraft-like world of blocks. The goal of the interaction is to build a structure using a limited number of block types, which can vary in complexity. Examples of possible target structures to build are:

![Shots of three possible target structures to build](./resources/imgs/voxelworld_combined_shots.png)

 Two roles arise: the Architect is provided with a target structure that needs to be built by the Builder. The Architect provides instructions to the Builder on how to create the target structure and the Builder can ask clarifying questions to the Architect if an instruction is unclear.

The progression of each game is recorded, corresponding to the construction of a target structure
by an Architect and Builder pair, as a discrete sequence of game observations. Each observation
contains the following information: 1) a time stamp, 2) the chat history up until that point in time,
3) the Builder’s position (a tuple of real-valued x, y, z coordinates as well as pitch and yaw angles,
representing the orientation of their camera), 4) the Builder’s block inventory, 5) the locations of
the blocks in the build region.

<img src="./resources/imgs/voxelwrold_building_dialog.gif" width="420" height="280" alt="Gif with interactions between Architect and Builder"/>

## Installation

To install the stable version of the IGLU-datasets API, please use the following command:

```sh
pip install git+https://github.com/microsoft/iglu-datasets.git@master
```

Alternatively, you can download the repo and install it manually:

```sh
git clone https://github.com/microsoft/iglu-datasets.git
cd iglu-datasets && python setup.py install
```

The requirements to install the API are the followings:

```
tqdm
PILLOW
opencv-python==4.5.5.64
pandas
numpy
filelock
requests
```

## Working with IGLU datasets 


The IGLU-datasets library prives an easy and flexible way for for with Single and Multi turn datasets.
Here is an example of how to use it:

```python
from iglu_datasets import MultiturnDataset, SingleturnDataset

# leave dataset_version empty to access the most recent version of the dataset.
# create a multiturn dataset instance
dataset = MultiturnDataset(dataset_version='v1.0') 

# create a singleturn dataset instance
dataset = SingleturnDataset(dataset_version='v1.0') 
```

On creation, this class will automatically download the dataset from this repository and parse (might take a few minutes) the raw data to store in a format that is fast and convienient to load. 
There are two ways for accessing the underlying data.
First, the `.sample()` method can be used. This method simply retrieves one random example from the dataset. The example is 
wrapped into `Task` format which has the following fields:

```python
sample = dataset.sample() # here a `Task` instance will be returned
print('Previous dialog\n', sample.dialog) # the whole prior conversation EXCLUDING the current instruction.
print('Instruction\n', sample.instruction) # the current instruction to execute by the builder. 
# If builder asks for clarifying questions, this will be a tuple (instruction, question, answer) 
# concatenated to a string.
print('Target grid\n', sample.target_grid) # 3D numpy array of shape (9, 11, 11). 
# Represents the volume snapshot of the target blocks world that correspond to the builder's 
# result in responce to the instruction.
print('Starting grid\n', sample.starting_grid) # 3D numpy array of shape (9, 11, 11). 
# Represents the volume snapshot of the starting blocks world with which the builder 
# starts executing the instruction.
```

The multiturn dataset consists of structures that represent overall collaboration goals. For each structure, we have several collaboration sessions that pair architects with builders to build each particular structure. Each session consists of a sequence of "turns". Each turn represents an *atomic* instruction and corresponding changes of the blocks in the world. The structure of a `Task` object is following:

  * `target_grid` - target blocks configuration that needs to be built
  * `starting_grid` - optional, blocks for the environment to begin the episode with.
  * `dialog` - full conversation between the architect and builder, including the most recent instruction
  * `instruction` - last utterance of the architect

Sometimes, the instructions can be ambiguous and the builder asks a clarifying question which the architect answers. In the latter case, `instruction` will contain three utterances: an instruction, a clarifying question, and an answer to that question. Otherwise, `instruction` is just one utterance of the architect.

Here is an example of task (the target structure is shown on the left and blocks to add are on the right):

<img src="./resources/vids/output.gif" width="640" height="392" alt="Gif with task vis"/>


To represent collaboration sessions, the `Subtasks` class is used. This class represents a sequence of dialog utterances and their corresponding goals (each of which is a partially completed structure). On `.sample()` call, it picks a random turn and returns a `Task` object, where starting and target grids are consecutive partial structures and the dialog contains all utterances up until the one corresponding to the target grid.

In the example above, the dataset object is structured as follows:

```python
# .tasks is a dict mapping from structure to a list of sessions of interaction
dataset.tasks 
# each value contains a list corresponding to collaboration sessions.
dataset.tasks['c73']
# Each element of this list is an instance of `Subtasks` class
dataset.tasks['c73'][0]
```

The `.sample()` method of `MultiturnDataset` does effectively the following:

```python
def sample(dataset):
  task_id = random.choice(dataset.tasks.keys())
  session = random.choice(dataset.tasks[task_id])
  subtask = session.sample() # Task object is returned
  return subtask
```

This behavior can be customized simply by overriding the reset method in a subclass:

```python
from iglu_datasets import MultiturnDataset

class MyDataset(MultiturnDataset):
  def __init__(self, *args, **kwargs):
    super().__init__(*args, **kwargs)
    self.my_task_id = 'c73'
    self.my_session = 0
  
  def sample(self):
    return self.tasks[self.my_task_id][self.my_session].sample()

my_dataset = MyDataset(dataset_version='v1.0')
# do training/sampling
```

On the first creation, the dataset is downloaded and parsed automatically. Below you will find the structure of the dataset:

```
dialogs.csv
builder-data/
  ...
  1-c118/ # session id - structure_id
    step-2
  ...
  9-c118/
    step-2
    step-4
    step-6
  1-c120/
    step-2
  ...
  23-c126/
    step-2
    step-4
    step-6
    step-8
```

Here, `dialog.csv` contains the utterances of architects and builders solving different tasks in 
different sessions. The `builder-data/` directory contains builder behavior recorded by the voxel.js engine. Right now we extract only the resulting grids and use them as targets.

### Singleturn dataset

The `SingleturnDataset` has the same structure of each sample. The main difference compared to the `MultiturnDataset` is 
that here tasks are not structured into a chain but rather branch out from random chain steps. 


Below you will find the structure of the single turn dataset:

```
single_turn_instructions.csv
multi_turn_dialogs.csv
initial_world_states/
  builder-data/
    <same as in multiturn>
target_world_states/
    actionHit/
      <tree structure with game sessions>
    <tree structure with game sessions>
```

Here, `multi_turn_dialogs.csv` and `initial_world_states/` is just the copy of the multiturn dataset under a different name. In `single_turn_instructions.csv` you can find the single turn instructions, and references to game sessions where the block states can be restored. 

### Grid prediction score calculation


Given a predicted grid and a target one, the intersection score is calculated based on their similarity. The score is determined regardless of global spatial position of currently placed blocks, it only takes into account how much the built blocks are similar to the target structure. To make it possible, at each step we calculate the intersection between the built and the target structures for each spatial translation within the horizontal plane and rotation around the vertical axis. Then we take the maximal intersection value among all translation and rotations. To calculate the score, we compare the maximal intersection size from the current step with the one from the previous step. The resulting intersection size can serve as a true positive rate for the `f1`/`precision`/`recall` score calculations, also it can be used as a reward function for a reinforcement learning agent. A visual example is shown below.

<img src="./resources/imgs/intersections.png" width="256">

Specifically, we run the code that is equivalent to the following one. Note that our version is much more optimized, while the version is given for reference:

```python
def maximal_intersection(grid, target_grid):
  """
  Args:
    grid (np.ndarray[Y, X, Z]): numpy array snapshot of a built structure
    target_grid (np.ndarray[Y, X, Z]): numpy array snapshot of the target structure
  """
  maximum = 0
  # iterate over orthogonal rotations
  for i in range(4):
    # iterate over translations
    for dx in range(-X, X + 1):
      for dz in range(-Z, Z + 1):
        shifted_grid = translate(grid, dx, dz)
        intersection = np.sum( (shifted_grid == target) & (target != 0) )
        maximum = max(maximum, intersection)
    grid = rotate_y_axis(grid)
  return maximum
```

In practice, a more optimized version is used. There is a way to convert this score into a reward function for a reinforcement learning agent. To do that, we can calculate the reward based on the temporal difference between maximal intersection of the two consecutive grids. Formally, suppose `grids[t]` is a built structure at timestep `t`. The reward is then calculated as:

```python
def calc_reward(prev_grid, grid, target_grid, , right_scale=2, wrong_scale=1):
  prev_max_int = maximal_intersection(prev_grid, target_grid)
  max_int = maximal_intersection(grid, target_grid)
  diff = max_int - prev_max_int
  prev_grid_size = num_blocks(prev_grid)
  grid_size = num_blocks(grid)
  if diff == 0:
    return wrong_scale * np.sign(grid_size - prev_grid_size)
  else:
    return right_scale * np.sign(diff)
```

In other words, if a recently placed block strictly increases or decreases the maximal intersection, the reward is positive or negative and is equal to `+/-right_scale`. Otherwise, its absolute value is equal to `wrong_scale` and the sign is positive if a block was removed or negative if added. This reward function is implemented in the [embodied IGLU environment](https://github.com/iglu-contest/gridworld).

## References

The described datasets are collected as a part of [IGLU:Interactive Grounded Language Understanding in a Collaborative Environment](https://www.aicrowd.com/challenges/neurips-2022-iglu-challenge), which is described in the following papers:

```
@article{mohanty2022collecting,
  title={Collecting Interactive Multi-modal Datasets for Grounded Language Understanding},
  author={Mohanty, Shrestha and Arabzadeh, Negar and Teruel, Milagro and Sun, Yuxuan and Zholus, Artem and Skrynnik, Alexey and Burtsev, Mikhail and Srinet, Kavya and Panov, Aleksandr and Szlam, Arthur and others},
  journal={arXiv preprint arXiv:2211.06552},
  year={2022}
}
```

```
@inproceedings{kiseleva2022interactive,
  title={Interactive grounded language understanding in a collaborative environment: Iglu 2021},
  author={Kiseleva, Julia and Li, Ziming and Aliannejadi, Mohammad and Mohanty, Shrestha and ter Hoeve, Maartje and Burtsev, Mikhail and Skrynnik, Alexey and Zholus, Artem and Panov, Aleksandr and Srinet, Kavya and others},
  booktitle={NeurIPS 2021 Competitions and Demonstrations Track},
  pages={146--161},
  year={2022},
  organization={PMLR}
}
```

```
@article{kiseleva2022iglu,
  title={Iglu 2022: Interactive grounded language understanding in a collaborative environment at neurips 2022},
  author={Kiseleva, Julia and Skrynnik, Alexey and Zholus, Artem and Mohanty, Shrestha and Arabzadeh, Negar and C{\^o}t{\'e}, Marc-Alexandre and Aliannejadi, Mohammad and Teruel, Milagro and Li, Ziming and Burtsev, Mikhail and others},
  journal={arXiv preprint arXiv:2205.13771},
  year={2022}
}
```

Consider citing the papers above if you use the assets for your research.

## Contributing

This project welcomes contributions and suggestions.  Most contributions require you to agree to a
Contributor License Agreement (CLA) declaring that you have the right to, and actually do, grant us
the rights to use your contribution. For details, visit https://cla.opensource.microsoft.com.

When you submit a pull request, a CLA bot will automatically determine whether you need to provide
a CLA and decorate the PR appropriately (e.g., status check, comment). Simply follow the instructions
provided by the bot. You will only need to do this once across all repos using our CLA.

This project has adopted the [Microsoft Open Source Code of Conduct](https://opensource.microsoft.com/codeofconduct/).
For more information see the [Code of Conduct FAQ](https://opensource.microsoft.com/codeofconduct/faq/) or
contact [opencode@microsoft.com](mailto:opencode@microsoft.com) with any additional questions or comments.

## Trademarks

This project may contain trademarks or logos for projects, products, or services. Authorized use of Microsoft 
trademarks or logos is subject to and must follow 
[Microsoft's Trademark & Brand Guidelines](https://www.microsoft.com/en-us/legal/intellectualproperty/trademarks/usage/general).
Use of Microsoft trademarks or logos in modified versions of this project must not cause confusion or imply Microsoft sponsorship.
Any use of third-party trademarks or logos are subject to those third-party's policies.
