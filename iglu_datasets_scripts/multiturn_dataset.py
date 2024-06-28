# Originally taken from: https://github.com/iglu-contest/gridworld
# Author: Milagro Teruel, Artem Zholus
from ipaddress import ip_address
import os
import json
import re
import pandas as pd
import numpy as np
import pickle
import bz2
from collections import defaultdict

from .task import Subtasks, Task, Tasks
from .download import download
from .common import VOXELWORLD_GROUND_LEVEL, fix_log, fix_xyz

from zipfile import ZipFile
from tqdm import tqdm



class MultiturnDataset(Tasks):
    DATASET_URL = {
        # temporal link for testing
        "v1.0": 'https://github.com/microsoft/iglu-datasets/raw/main/datasets/multiturn_dataset.zip',
    }  # Dictionary holding dataset version to dataset URI mapping
    DIALOGS_FILENAME = 'dialogs.csv'
    BLOCK_MAP = {  # voxelworld's colour id : iglu colour id
        00: 0,  # air
        57: 1,  # blue
        50: 6,  # yellow
        59: 2,  # green
        47: 4,  # orange
        56: 5,  # purple
        60: 3,  # red
    }

    def __init__(self, dataset_version="v1.0", task_kwargs=None, force_download=False) -> None:
        """
        IGLU Multiturn dataset.

        Current version of the dataset covers 31 structures in 127 staged game sessions
        resulting in 584 tasks.

        Args:
            dataset_version: Which dataset version to use.
            task_kwargs: Task-class specific kwargs. For reference see gridworld.task.Task class
            force_download: Whether to force dataset downloading
        """
        self.dataset_version = dataset_version
        if dataset_version not in self.DATASET_URL.keys():
            raise Exception(
                "Unknown dataset_version:{} provided.".format(dataset_version))
        if task_kwargs is None:
            task_kwargs = {}
        self.task_kwargs = task_kwargs
        data_path, custom = self.get_data_path()
        if isinstance(self.DATASET_URL[self.dataset_version], tuple):
            filename = self.DATASET_URL[self.dataset_version][1].split('/')[-1]
        else:
            filename = self.DATASET_URL[self.dataset_version].split('/')[-1]
        if custom:
            filename = f'cached_{filename}'
        parse = False
        if not custom:
            try:
                # first, try downloading the lightweight parsed dataset
                self.download_parsed(data_path=data_path, file_name=filename, force_download=force_download)
                self.load_tasks_dataset(os.path.join(data_path, filename))
            except Exception as e:
                print(e)
                parse = True
        if custom or parse:
            print('Loading parsed dataset failed. Downloading full dataset.')
            # if it fails, download it manually and cache it
            self.download_dataset(data_path, force_download)
            dialogs = self.get_instructions(data_path)
            self.tasks = defaultdict(list)
            self.parse_tasks(dialogs, data_path)
            self.dump_tasks_dataset(os.path.join(data_path, filename))

    def get_instructions(self, data_path):
        return pd.read_csv(os.path.join(data_path, self.DIALOGS_FILENAME))

    @classmethod
    def get_data_path(cls):
        """Returns the path where iglu dataset will be downloaded and cached.

        It can be set using the environment variable IGLU_DATA_PATH. Otherwise,
        it will be `~/.iglu/data/iglu`.

        Returns
        -------
        str
            The absolute path to data folder.
        """
        if 'IGLU_DATA_PATH' in os.environ:
            data_path = os.path.join(
                os.environ['IGLU_DATA_PATH'], 'data', 'iglu')
            custom = True
        elif 'HOME' in os.environ:
            data_path = os.path.join(
                os.environ['HOME'], '.iglu', 'data', 'iglu')
            custom = False
        else:
            data_path = os.path.join(
                os.path.expanduser('~'), '.iglu', 'data', 'iglu')
            custom = False
        return data_path, custom

    def download_dataset(self, data_path, force_download):
        path = os.path.join(data_path, 'iglu_dataset.zip')
        if (not os.path.exists(os.path.join(data_path, self.DIALOGS_FILENAME))
                or force_download):
            url = self.DATASET_URL[self.dataset_version]
            if not isinstance(url, str):
                url = url[0]
            download(
                url=url,
                destination=path,
                data_prefix=data_path,
                description='Downloading multiturn dataset'
            )
            print('Dataset downloaded!')
            with ZipFile(path) as zfile:
                zfile.extractall(data_path, members=tqdm(zfile.namelist(), desc='Extracting zip file'))

    def download_parsed(self, data_path, file_name='parsed_tasks_multiturn_dataset.tar.bz2',
                        force_download=False):
        path = os.path.join(data_path, file_name)
        if (not os.path.exists(path) or force_download):
            url = self.DATASET_URL[self.dataset_version]
            if isinstance(url, str):
                raise ValueError('this dataset version does not support parsed data!')
            url = url[1]
            download(
                url=url,
                destination=path,
                data_prefix=data_path,
                description='downloading task dataset'
            )

    def dump_tasks_dataset(self, path):
        print('caching tasks dataset... ', end='')
        pickled = pickle.dumps(self.tasks)
        compressed = bz2.compress(pickled)
        with open(path, 'wb') as f:
            f.write(compressed)
        print('done')

    def load_tasks_dataset(self, path):
        with open(path, 'rb') as f:
            data = f.read()
        data = bz2.decompress(data)
        self.tasks = pickle.loads(data)

    def process(self, s):
        return re.sub(r'\$+', '\n', s)

    @classmethod
    def transform_block(cls, block):
        """Adjust block coordinates and replace id."""
        x, y, z, bid = block
        y = y - VOXELWORLD_GROUND_LEVEL - 1
        bid = cls.BLOCK_MAP[bid]
        return x, y, z, bid

    def parse_tasks(self, dialogs, path):
        """Fills attribute `self.tasks` with utterances from `dialogs` and
        VoxelWorld states for each step.

        Parameters
        ----------
        dialogs : pandas.DataFrame
            Contains information of each turn in the session, originally stored
            in database tables. The information includes:
                - PartitionKey: corresponds to Game attempt or session. It is
                  constructed following the pattern `{attemptId}-{taskId}`
                - structureId: task id of the session.
                - StepId: number of step in the session. For multi-turn IGLU
                  data, all odd steps have type architect and even steps
                  have type builder. Depending on the task type, different
                  columns will be used to fill the task.
                - IsHITQualified: boolean indicating if the step is valid.

        path : str
            Path with the state of the VoxelWorld grid after each session.
            Each session should have an associated directory named with the
            session id, with json files that describe the world state after
            each step.

        """
        # Partition key
        groups = dialogs.groupby('PartitionKey')
        for sess_id, gr in tqdm(groups, total=len(groups), desc='parsing dataset'):
            # This corresponds to the entire dialog between steps with
            # changes to the blocks
            utt_seq = []
            blocks = []
            kwargs = {}
            kwargs['session_id'] = sess_id
            if not os.path.exists(f'{path}/builder-data/{sess_id}'):
                continue
            # Each session should have a single taskId associated.
            assert len(gr.structureId.unique()) == 1
            structure_id = gr.structureId.values[0]
            # Read the utterances and block end positions for each step.
            for i, row in gr.sort_values('StepId').reset_index(drop=True).iterrows():
                if not row.IsHITQualified:
                    continue
                if row.StepId % 2 == 1:
                    # Architect step
                    if isinstance(row.instruction, str):
                        utt_seq.append([])
                        utt_seq[-1].append(
                            f'<Architect> {self.process(row.instruction)}')
                    elif isinstance(row.Answer4ClarifyingQuestion, str):
                        utt_seq[-1].append(
                            f'<Architect> {self.process(row.Answer4ClarifyingQuestion)}')
                else:
                    # Builder step
                    if isinstance(row.ClarifyingQuestion, str):
                        utt_seq[-1].append(
                            f'<Builder> {self.process(row.ClarifyingQuestion)}')
                        continue
                    blocks.append([])
                    curr_step = f'{path}/builder-data/{sess_id}/step-{row.StepId}'
                    if not os.path.exists(curr_step):
                        break
                        # TODO: in this case the multiturn collection was likely
                        # "reset" so we need to stop parsing this session. Need to check that.
                    with open(curr_step) as f:
                        step_data = json.load(f)
                    for block in step_data['worldEndingState']['blocks']:
                        x, y, z, bid = self.transform_block(block)
                        blocks[-1].append((x, y, z, bid))
            # Aggregate all previous blocks into each step
            if len(blocks) < len(utt_seq):
                # handle the case of missing of the last blocks record
                utt_seq = utt_seq[:len(blocks)]
            i = 0
            while i < len(blocks):
                # Collapse steps where there are no block changes.
                if len(blocks[i]) == 0:
                    if i == len(blocks) - 1:
                        blocks = blocks[:i]
                        utt_seq = utt_seq[:i]
                    else:
                        blocks = blocks[:i] + blocks[i + 1:]
                        utt_seq[i] = utt_seq[i] + utt_seq[i + 1]
                        utt_seq = utt_seq[:i + 1] + utt_seq[i + 2:]
                else:
                    i += 1
            if len(blocks) > 0:
                # Create random subtasks from the sequence of dialogs and blocks
                task = Subtasks(utt_seq, blocks, **self.task_kwargs, **kwargs)
                assert len(utt_seq) == len(blocks)
                self.tasks[structure_id].append(task)

    def sample(self):
        sample = np.random.choice(list(self.tasks.keys()))
        sess_id = np.random.choice(len(self.tasks[sample]))
        self.current = self.tasks[sample][sess_id]
        return self.current.sample()

    def __len__(self):
        return sum(len(sess.structure_seq) for sess in sum(self.tasks.values(), []))

    def __iter__(self):
        for task_id, tasks in self.tasks.items():
            for j, task in enumerate(tasks):
                for k, subtask in enumerate(task):
                    yield task_id, j, k, subtask
