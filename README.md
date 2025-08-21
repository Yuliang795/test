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

### 📂Dataset
Each entry in the dataset JSON file corresponds to a problem instance, keyed by a **problem ID** as in CSPLib.

Each problem is represented as a dictionary with the following fields:
* **`source`**: Origin of the problem (e.g., `"csplib"`).
* **`name`**: Name of the problem.
* **`content`**: Problem description.
* **`value_info`**: Specification of the input parameters, including their meaning, type, and structure.
* **`ref_sol_format`**: Specification of the expected output solution format, including variable names, types, sizes, and optionaly examples.
* **`prob_type`**: `"cop"` (constraint optimization problem) or `"csp"` (constraint satisfaction problem).

------

## Constraint Modeling Workflows

*Coming soon.*

------

## Getting Started

*Coming soon.*

------

## Citation

*Coming soon.*
