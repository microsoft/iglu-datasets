# IGLU Dataset

This dataset consists of 

  - `clarifying_questions_train.csv` 
  - `question_bank.csv`
  - `initial_world_states` folder: Contains states of starting world for participants to build on.
  - `target_world_states` folder: actionHit folder contains final state of the world after performing the free-form building task.

`clarifying_questions_train.csv` has the following columns:

  * `GameId` - Id of the game session.
  * `InitializedWorldPath` - Path to the file under `initial_world_states` that contains state of the world intialized to the architect. The architect provides an instruction to build based on this world state. More information to follow on how the world state can be parsed/ visualized. 
  * `InputInstruction` - Instruction provided by the architect.
  * `IsInstructionClear` - Specifies whether the instruction provided by architect is clear or ambiguous. This has been marked by another annotator who is not the architect.
  * `ClarifyingQuestion` - Question asked by annotator upon marking instruction as being ambiguous.
  * `qrel` - Question id (qid) of the relevant clarifying question for the current instruction.
  * `qbank` - List of clarifying question ids that need to be ranked for each unclear instruction. The mapping between clarifying questions and ids is present in the `question_bank.csv`.

*Merged list of ids in the `qrel` and `qbank` columns will give you the list of all qids to be ranked for each ambiguous instruction.*

`question_bank.csv`: This file contains mapping between `qids` mentioned in `qrel` and `qbank` columns of the `clarifying_questions_train.csv` to the bank of clarifying questions issued by annotators.