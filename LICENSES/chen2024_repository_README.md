Welcome to the  repository for the "Light Adaptation Clamp" tool, as described in our recent publication [Chen 2023](https://www.biorxiv.org/content/10.1101/2023.10.20.563304v1). This repository hosts both the source code and the detailed mathematical formulations underlying the tool's functionality.

**About the Light Adaptation Clamp Tool**

In this repository, we present a  tool called `Light Adaptation Clamp (LaC)` capable of designing light stimuli that specifically and predictably modulate phototransduction currents in both rod and cone photoreceptors. This tool  accounts for and compensates for nonlinear characteristics, such as light adaptation, inherent in phototransduction processes. Developed on the foundation of well-established models of the rod and cone phototransduction cascade, it enables separation between the nonlinearities associated with phototransduction and those present in subsequent neural circuits. Consequently, this tool opens up new possibilities for directly investigating the influence of photoreceptor adaptation on downstream visual signals and perception.

**Repository Contents**

Source Code: The `src` folder contains the MATLAB scripts and functions used to implement the light adaptation clamp tool. It includes routines for simulating photoreceptor responses, estimating stimuli, and more.
Mathematical Formulations: Detailed LaTeX equations provided in `GitHub Wiki` elucidate the mathematical concepts and equations that form the basis of our simulations and analyses.
Usage Examples: Practical examples and usage scenarios are showncased in `demoMain.m` to help users understand how to apply the tool in their research.

**Contributing**

If you have suggestions, bug reports, or improvements, please feel free to open an issue or submit a pull request.

**Citation**

If you use this tool in your research, please cite: [Chen 2023](https://www.biorxiv.org/content/10.1101/2023.10.20.563304v1)
