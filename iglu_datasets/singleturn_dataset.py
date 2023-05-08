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
from .multiturn_dataset import MultiturnDataset

from zipfile import ZipFile
from tqdm import tqdm




class SingleTurnIGLUDataset(MultiturnDataset):
    SINGLE_TURN_INSTRUCTION_FILENAME = 'single_turn_instructions.csv'
    MULTI_TURN_INSTRUCTION_FILENAME = 'multi_turn_dialogs.csv'
    DATASET_URL = {
        "v1.0": 'https://github.com/microsoft/iglu-datasets/raw/main/datasets/single_turn_dataset.zip',
    }
    BLOCK_MAP = {  
        # voxelworld's colour id : iglu colour id
        00: 0,  # air
        57: 1,  # blue
        50: 6,  # yellow
        59: 2,  # green
        47: 4,  # orange
        56: 5,  # purple
        60: 3,  # red
        # voxelworld (freeze version)'s colour id : iglu colour id
        86: 1,  # blue
        87: 6,  # yellow
        88: 2,  # green
        89: 4,  # orange
        90: 5,  # purple
        91: 3,  # red
    }
    
    def __init__(self, dataset_version='v1.0', task_kwargs=None,
            force_download=False, limit=None) -> None:
        self.limit = limit
        super().__init__(dataset_version=dataset_version,
            task_kwargs=task_kwargs, force_download=force_download)

    def get_instructions(self, data_path):
        single_turn_df = pd.read_csv(os.path.join(
            data_path, self.SINGLE_TURN_INSTRUCTION_FILENAME))
        if self.limit is not None:
            return single_turn_df[:self.limit]
        return single_turn_df

    def get_multiturn_dialogs(self, data_path):
        return pd.read_csv(os.path.join(
            data_path, self.MULTI_TURN_INSTRUCTION_FILENAME))

    @classmethod
    def get_data_path(cls):
        """Returns the path where iglu dataset will be downloaded and cached.

        It can be set using the environment variable IGLU_DATA_PATH. Otherwise,
        it will be `~/.iglu/data/single_turn_dataset`.

        Returns
        -------
        str
            The absolute path to data folder.
        """
        if 'IGLU_DATA_PATH' in os.environ:
            data_path = os.environ['IGLU_DATA_PATH']
            custom = True
        elif 'HOME' in os.environ:
            data_path = os.path.join(
                os.environ['HOME'], '.iglu', 'data', 'single_turn_dataset')
            custom = False
        else:
            data_path = os.path.join(
                os.path.expanduser('~'), '.iglu', 'data', 'single_turn_dataset')
            custom = False
        return data_path, custom

    def download_dataset(self, data_path, force_download):
        instruction_filepath = os.path.join(
            data_path, self.SINGLE_TURN_INSTRUCTION_FILENAME)
        path = os.path.join(data_path, 'single_turn_dataset.zip')
        if os.path.exists(instruction_filepath) and not force_download:
            print("Using cached dataset")
            return
        url = self.DATASET_URL[self.dataset_version]
        if not isinstance(url, str):
            url = url[0]
        print(f"Downloading dataset from {url}")
        download(
            url=url,
            destination=path,
            data_prefix=data_path
        )
        with ZipFile(path) as zfile:
            zfile.extractall(data_path)

    def create_task(self, previous_chat, initial_grid, target_grid,
                    last_instruction):
        task = Task(
            dialog=previous_chat,
            instruction=last_instruction,
            target_grid=Tasks.to_dense(target_grid),
            starting_grid=Tasks.to_sparse(initial_grid),
            full_grid=Tasks.to_dense(target_grid),
        )
        # To properly init max_int and prev_grid_size fields
        task.sample()
        return task

    def get_previous_dialogs(self, single_turn_row, multiturn_dialogs):
        # Filter multiturn rows with this game id and previous to step
        utterances = []
        mturn_data_path = single_turn_row.InitializedWorldPath.split('/')[-2:]
        if len(mturn_data_path) != 2 or '-' not in mturn_data_path[1]:
            print(f"Error with initial data path {single_turn_row.InitializedWorldPath}."
                  "Could not parse data path to get previous dialogs.")
            return utterances
        mturn_game_id = mturn_data_path[0]
        try:
            mturn_last_step = int(mturn_data_path[1].replace("step-", ""))
        except Exception as e:
            print(f"Error with initial data path {single_turn_row.InitializedWorldPath}."
                  "Could not parse step id to get previous dialogs.")
            return utterances
        dialog_rows = multiturn_dialogs[
            (multiturn_dialogs.PartitionKey == mturn_game_id) &
            (multiturn_dialogs.StepId < mturn_last_step) &
            (multiturn_dialogs.IsHITQualified == True)]

        for _, row in dialog_rows.sort_values('StepId')\
                .reset_index(drop=True).iterrows():
            if row.StepId % 2 == 1:
                # Architect step
                if isinstance(row.instruction, str):
                    utterance = row.instruction
                    utterances.append(
                        f'<Architect> {self.process(utterance)}')
                elif isinstance(row.Answer4ClarifyingQuestion, str):
                    utterance = row.Answer4ClarifyingQuestion
                    utterances.append(
                        f'<Architect> {self.process(utterance)}')
            elif isinstance(row.ClarifyingQuestion, str):
                utterances.append(
                    f'<Builder> {self.process(row.ClarifyingQuestion)}')

        return utterances

    def parse_tasks(self, dialogs, path):
        """Fills attribute `self.tasks` with instances of Task.

        A Task contains an initial world state, a target world state and a
        single instruction.

        Parameters
        ----------
        dialogs : pandas.DataFrame
            Contains information of each session, originally stored
            in database tables. The information includes:
                - InitializedWorldStructureId or InitializedWorldGameId:
                  Original target structure id of the initial world.
                - InitializedWorldPath: Path to a json file that contains the
                  initial blocks of the world.
                - ActionDataPath: Path relative to dataset location with the
                  target world.
                - InputInstruction: Session instruction
                - IsHITQualified: boolean indicating if the step is valid.

        path : _type_
            Path with the state of the VoxelWorld grid after each session.
            Each session should have an associated directory named with the
            session id, with json files that describe the world state after
            each step.

        """
        dialogs = dialogs[dialogs.InitializedWorldPath.notna()]
        dialogs['InitializedWorldPath'] = dialogs['InitializedWorldPath'] \
            .apply(lambda x: x.replace('\\', os.path.sep))
        dialogs['InitializedWorldPath'] = dialogs['InitializedWorldPath'] \
            .apply(lambda x: x.replace('/', os.path.sep))

        # Get the list of games for which the instructions were clear.
        turns = dialogs[dialogs.GameId.str.match("CQ-*")]

        # Util function to read structure from disk.
        def _load_structure(structure_path):
            filepath = os.path.join(path, structure_path)
            if not os.path.exists(filepath):
                return None

            with open(filepath) as structure_file:
                structure_data = json.load(structure_file)
                blocks = structure_data['worldEndingState']['blocks']
                structure = [self.transform_block(block) for block in blocks]

            return structure

        multiturn_dialogs = self.get_multiturn_dialogs(path)

        tasks_count = 0
        pbar = tqdm(turns.iterrows(), total=len(turns), desc='parsing dataset')
        for _, row in pbar:
            pbar.set_postfix_str(f"{tasks_count} tasks") 
            assert row.InitializedWorldStructureId is not None

            # Read initial structure
            initial_world_blocks = _load_structure(row.InitializedWorldPath)
            if initial_world_blocks is None:
                pbar.write(f"Skipping '{row.GameId}'. Can't load starting structure from '{row.InitializedWorldPath}'.")
                continue

            target_world_blocks = _load_structure(row.TargetWorldPath)
            if target_world_blocks is None:
                pbar.write(f"Skipping '{row.GameId}'. Can't load target structure from '{row.TargetWorldPath}'.")
                continue
            
            # Check if target structure matches the initial structure.
            if sorted(initial_world_blocks) == sorted(target_world_blocks):
                pbar.write(f"Skipping '{row.GameId}'. Target structure is the same as the initial one.")
                continue

            # Get the original game.
            orig = dialogs[dialogs.GameId == row.GameId[len("CQ-"):]]
            if len(orig) == 0:
                pbar.write(f"Skipping '{row.GameId}'. Can't find its original game '{row.GameId[len('CQ-'):]}'.")
                continue

            assert len(orig) == 1
            orig = orig.iloc[0]
            
            # Load original structure.
            orig_target_world_blocks = _load_structure(orig.TargetWorldPath)
            if orig_target_world_blocks is None:
                pbar.write(f"Skipping '{row.GameId}'. Can't load original target structure from '{orig.TargetWorldPath}'.")
                continue
           
            # Check if original structure matches the rebuilt one.
            if sorted(orig_target_world_blocks) != sorted(target_world_blocks):
                pbar.write(f"Skipping '{row.GameId}'. Target structure doesn't match the one in '{orig.GameId}'.")
                continue

            last_instruction = '<Architect> ' + self.process(row.InputInstruction)
            # Read utterances
            utterances = self.get_previous_dialogs(row, multiturn_dialogs)
            utterances.append(last_instruction)
            utterances = '\n'.join(utterances)
            # Construct task
            task = self.create_task(
                utterances, initial_world_blocks, target_world_blocks,
                last_instruction=last_instruction)

            # e.g. initial_world_states\builder-data/8-c92/step-4 -> 8-c92/step-4 
            task_id, step_id = row.InitializedWorldPath.split("/")[-2:]
            #self.tasks[row.InitializedWorldStructureId].append(task)
            self.tasks[f"{task_id}/{step_id}"].append(task)
            tasks_count += 1

    def __iter__(self):
        for task_id, tasks in self.tasks.items():
            for j, task in enumerate(tasks):
                yield task_id, j, 1, task

    def __len__(self):
        return len(sum(self.tasks.values(), []))
