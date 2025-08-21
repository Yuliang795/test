# Do LLMs Understand Constraint Programming? Zero-Shot Constraint Programming Model Generation Using LLMs

This repository contains materials for the paper:
 **"Do LLMs Understand Constraint Programming? Zero-Shot Constraint Programming Model Generation Using LLMs"**
 [Yuliang Song, Eldan Cohen, LION 19, 2025]

------

## Contents

- [CPEVAL Benchmark](#cpeval-benchmark)
- [Constraint Modeling Workflows](#constraint-modeling-workflows)
- [Getting Started](#getting-started)
- [Citation](#citation)

------

## CPEVAL Benchmark
- The benchmark is stored in a single JSON file. 
- The evaluation suite is provided in `cpeval.py`. 
- The solution format checker is provided in `solution_checker.py`.

### 📂Dataset
Each problem, keyed by a **problem ID** aligned with CSPLib, is represented as a dictionary with the following fields:
* **`source`**: Origin of the problem (e.g., `"csplib"`).
* **`name`**: Name of the problem.
* **`content`**: Problem description.
* **`value_info`**: Specification of the input parameters, including their meaning, type, and structure.
* **`ref_sol_format`**: Specification of the expected output solution format, including variable names, types, sizes, and optionaly examples.
* **`prob_type`**: `"cop"` (constraint optimization problem) or `"csp"` (constraint satisfaction problem).

The corresponding input parameters files are provided in `instance/`.

### 🔍Evaluation Suite
The evluation suite is composed of a set of checker functions to verify the correctness of the generated models in terms of constraint satisfaction and solution optimality. Each function, named `prob<problem_id>_verify_func`, takes two arguments:
 (1) `data_dict` - contains all the input parameter values provided in the benchmark.
 (2) `hypothesis_solution` - contains solution variables in the predefined formats.

The checker function returns results in two strings:
 - `"pass"` if all constraints checked are satisfied.
 - `"optimal"` if the solution is optimal (for COP problems).


### 📏Output Format Checker
Each problem has a dedicated output format checker function, named `prob<problem_id>_solformat_checker`, which verifies whether the generated solutions conform to the predefined formats. The core of this checking mechanism is relatively universal, but each problem has its own specific function to support customized requirements. 
*Coming soon.*

------

## Constraint Modeling Workflows

*Coming soon.*

------

## Getting Started

*Coming soon.*

------

## Citation

*Coming soon.*
