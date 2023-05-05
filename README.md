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

TODO add past competition links

TODO Add citation


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
