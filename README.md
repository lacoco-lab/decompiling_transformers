### Introduction

This is the repo for paper [Discovering Interpretable Algorithms by Decompiling Transformers to RASP](http://arxiv.org/abs/2602.08857).

### Scripts

All of our scripts are in `crasp/scripts/patching/`
Here we briefly describe what they are used for.

- Model Training
  - `train_new_models.py` defines classes for collator, models with BCE loss for training on formal languages etc.
  - `patching_data.py` defines all datasets for algorithmic and formal language tasks.
  - `train_new_models_search.py` search over hyperparameter combinations and train models for a given task.
  - `train_new_models_checkpoints.py` - save checkpoints during model training
  - `train_new_models_archs.py` train models with more hyperparameter combinations,  because `train_new_models_search.py` may stop early once it finds a generalizing model.

- Causal Pruning
  - `pruning_model.py` defines classes for models running with different kinds of computational graphs in different pruning stages. To understand the code, check Algorithm 1 in [Li and Janson 2024](https://neurips.cc/virtual/2024/poster/93600)
    - `PruningModelWithHooks` defines model where each components is prunable (stage 1)
    - `PruningModelWithHooksFullPaths` defines model where each path is prunable (stage 2)
    - `PruningModelWithHooksForQK` defines model where each QK product (selector) is prunable (stage 3)
    - `MaskSampler`, `MaskSamplerFullPaths`, and `MaskSamplerForQK` define classes holding the parameters that control the distribution of masks
    - `OptimalAblationVectors` and `OptimalQueryBiasVectors` contains various learned constants, e.g., optimal ablations
    - there are also various function for converting computational graph (config) between stages
  - `patching.py` runs the causal pruning
  - `delineate_curve_for_model.py` implements the automatic pruning coefficient searching. It automatically submits jobs to [HTCondor](https://htcondor.readthedocs.io/en/latest/index.html) (a job queuing system on clusters), and determines the new coefficients to use based on previous results. Each submitted job is a run for `patching.py` with certain hyperparameters.

- Primitive Matching for MLPs
  - `convert_mlp.py` implements tracing activation variables, iterating over primitives to match with the best one.
  - `patching_utils.py` defines MLP primitives
  - `show_heatmap.py` implements (1) the backup approach for explaining MLPs, that is, it saves input activation variables and effects on output logits or attention logits. (2) tracing activation variables and the op matrix ($A$)
  - `run_convert_mlp.py` runs the above process on the saved pruned models, and save new results for this step

- Primitive Matching for Unembedding and Attention
  - `attention_primitives_hook.py` contains a modified attention forward pass, which implements replacement with primitives.
  - `logits_primitives_hook.py` contains a forward hook for unembedding projection, which implements replacement with primitives.
  - `find_primitives_for_all_runs.py` is an endpoint used to run replacement of attention and unembedding with primitives.
  - `find_primitives.py` contains a function implementing general logic around replacement of attention and unembedding with primitives.
  - `try_primitives.py` contains a funcation implementing forward pass of a model with primitives and calculating metrics.
  - `primitives_classes.py` defines dataclasses used for primitives replacement.
  - `primitives_for_coefficients.py` defines a set of primitives.
  - `primitives_helpers.py` contains various helper functions.
  - `primitives_search.py` implements algorithms for primitive replacement. 'greedy' stands for replacement with predefined primitives, and 'round' for rounding the original heatmaps. In the paper we use 'greedy_then_round', which stands for replacing with predefined primitives where possible, and rounding the rest.
  - `round_primitive.py` implements a class and training loop for rounding heatmaps.

- Generating D-RASP code
  - `convert_to_code.py` implements getting D-RASP code from a model.
  - `get_programs_and_heatmaps.py` is an endpoint used to copy all the relevant heatmaps to one place, generate D-RASP code and produce a latex file with program visulization.
  - `plot_for_mlp_interpretation.py` contains a function used for plotting interpretations of the MLPs, which were not replaced with primitives. Is used in `get_programs_and_heatmaps.py` file.

- Results Visualization
  - `streamlit_app/app.py` implements an streamlit APP that visualize decompilation results, where you can check the pareto frontiers, decompiled programs, and various heatmaps.
  - `streamlit_app/draw_fig_xxx.py` and `streamlit_app/get_xxx.py` draw figures for various pareto frontiers and generalization results shown in the paper.
  - `plot_example_for_main_paper.py` is an endpoint used for plotting attention and unembedding primitives for the main paper.
  - `plot_func_for_main_paper.py` implements the plotting function for `plot_example_for_main_paper.py`.
  - `plot_for_mlp_main_paper_d4.py` and `plot_for_mlp_main_paper_sort.py` are used to plot MLP interpretation figures in the main paper.
  - `print_all_primitives.py` generates and plots examples of all the predefined primitives.
  - `run_pretty_example.py` is an endpoint used to run a model on a specific short example to collect pretty heatmaps for activation variables.


### Saved Models

The folder `share/saved_models` contains all the models on which we run our decompilation pipeline. We share them so that they can serve as a small benchmark. Future research may compare new methods with ours by applying on the same set of models.

### Running the Pipeline

Our implementation makes use of job queuing system, so if the environment is different, directly running it without modification or adaptation will not work. Nonetheless, it would be helpful to understand basic running order of our scripts.

1. (optional) train models to be decompiled later. Or you can directly use models in `share/saved_models`
2. run `delineate_curve_for_model.py` to perform pruning, which produces many pruned models with different degree of sparsity. Check frontiers with `streamlit_app/app.py`, to see if LLNA holds. If the highest match accuracy becomes low in stage 2, set `--split_mlp=False` and run again.
3. run `run_convert_mlp.py` to replace MLPs with primitives
4. look at the frontiers again, create a `good_models.json` file that contains a dictionary in the same format as shown in this repo. For each series, this file selects a specific pruning run, for which we will then generate D-RASP code. Make sure that this pruning run is successful in stage 3, otherwise it cannot be visualized with the current code. If none of the pruning runs are successful in stage 3, keep the entry for the series in `good_models.json` empty. For the paper, we selected pruning runs with the smallest number of edges which achieved match accuracy of at least 0.9 in stage 3.
5. run `find_primitives_for_all_runs.py` to replace attention and unembedding in the pruning runs from `good_models.json` with primitives. Specify correct paths in this file before running.
6. run `run_pretty_example.py` to collect pretty plots for heatmaps of activation variables used for program visualization later. Update paths and dictionaries in this file before running.
7. run `get_programs_and_heatmaps.py` to create latex files with visualizations of programs. This script copies relevant heatmaps to a specified location, generates D-RASP code for each model, and creates latex files with heatmaps and programs. The output of this script is used in the Appendix section of the paper.

