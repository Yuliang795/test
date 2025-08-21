# Do LLMs Understand Constraint Programming? Zero-Shot Constraint Programming Model Generation Using LLMs

This repository contains materials for the paper:
 **"Do LLMs Understand Constraint Programming? Zero-Shot Constraint Programming Model Generation Using LLMs"**
 [Yuliang Song, Eldan Cohen, LION 19, 2025]

📄 [Paper on OpenReview](https://openreview.net/forum?id=6zlpzSKzqj)


------

## Contents

- [CPEVAL Benchmark](#cpeval-benchmark)
- [Constraint Modeling Workflows](#constraint-modeling-workflows)
- [Getting Started](#getting-started)
- [Citation](#citation)

------

## CPEVAL Benchmark
- The benchmark is stored in `base_prob.json`. 
- The evaluation suite is provided in `csplib_verify_scripts.py`. 

### 📂 Dataset
Each problem, keyed by a **problem ID** aligned with CSPLib, is represented as a dictionary with the following fields:
* **`source`**: Origin of the problem (e.g., `"csplib"`).
* **`name`**: Name of the problem.
* **`content`**: Problem description.
* **`value_info`**: Specification of the input parameters, including their meaning, type, and structure.
* **`ref_sol_format`**: Specification of the expected output solution format, including variable names, types, sizes, and optionally examples.
* **`prob_type`**: `"cop"` (constraint optimization problem) or `"csp"` (constraint satisfaction problem).

The corresponding input parameters files are provided in `instance/`.

### 🔍 Evaluation Suite
The evaluation suite is composed of a set of checker functions to verify the correctness of the generated models in terms of constraint satisfaction and solution optimality. Each function, named `prob<problem_id>_verify_func`, takes two arguments:
 (1) `data_dict` - contains all the input parameter values provided in the benchmark.
 (2) `hypothesis_solution` - contains solution variables in the predefined formats.

The checker function returns results in two strings:
 - `"pass"` if all constraints checked are satisfied.
 - `"optimal"` if the solution is optimal (for COP problems).

------

## Constraint Modeling Workflows

*Coming soon.*

------

## Getting Started

*Coming soon.*

------

## Citation
```bibtex
@inproceedings{song2025llmcp,
  title={Do LLMs Understand Constraint Programming? Zero-Shot Constraint Programming Model Generation Using LLMs},
  author={Song, Yuliang and Cohen, Eldan},
  booktitle={Proceedings of the 19th Learning and Intelligent Optimization Conference (LION-25)},
  pages={In press},
  year={2025},
  url={https://openreview.net/forum?id=6zlpzSKzqj}
}
```
