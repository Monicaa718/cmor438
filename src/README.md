# Source Code (src)

This directory contains the core implementation code for the CMOR 438 (Rice ML) project.

All machine learning models, algorithms, and supporting utilities are implemented here. The goal of this folder is to keep model logic clean, modular, and reusable, separate from examples and testing code.

## Contents

Typical files in this directory include:
- Model implementations
- Training and optimization logic
- Helper functions for data processing and evaluation
- Core mathematical or algorithmic components used across the project

Each module is designed to focus on a single responsibility to make the code easier to understand, debug, and extend.

## Design Philosophy

- **Clarity over cleverness**: Code is written to be readable and instructional.
- **Modularity**: Each algorithm or component is isolated into its own file when possible.
- **Reusability**: Functions and classes in `src` can be imported by both `examples/` and `tests/`.

## How This Folder Is Used

- Code in `src/` is imported and demonstrated in the `examples/` directory.
- Unit tests in `tests/unit/` directly test functions and classes defined here.
- This structure mirrors real-world machine learning and software engineering workflows.

If you are adding a new model or algorithm, its main implementation should live in this directory.
