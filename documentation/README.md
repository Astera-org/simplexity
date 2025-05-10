# Simplexity Experimentation Framework Documentation

Welcome to the documentation for the Simplexity experimentation framework.
This guide is designed to help both experimentalists and LLM coding agents understand and utilize the codebase for running machine learning experiments.

## Table of Contents

*   **1. Introduction**
    *   [1.1. Overview of the Framework](./1.0-Introduction/1.1-Overview.md)
    *   [1.2. Target Users](./1.0-Introduction/1.2-Target-Users.md)
*   **2. Getting Started**
    *   [2.1. Installation](./2.0-Getting-Started/2.1-Installation.md)
    *   [2.2. Core Concepts](./2.0-Getting-Started/2.2-Core-Concepts.md)
*   **3. Running Experiments**
    *   [3.1. Using `run_experiment.py`](./3.0-Running-Experiments/3.1-Run-Experiment-Script.md)
    *   [3.2. Configuration System (Hydra)](./3.0-Running-Experiments/3.2-Configuration-Hydra.md)
    *   [3.3. Understanding `experiment.yaml`](./3.0-Running-Experiments/3.3-Experiment-YAML-Structure.md)
    *   [3.4. Hyperparameter Sweeping (Optuna)](./3.0-Running-Experiments/3.4-Hyperparameter-Sweeping.md)
*   **4. Key Components**
    *   [4.1. Generative Processes](./4.0-Key-Components/4.1-Generative-Processes.md)
    *   [4.2. Predictive Models](./4.0-Key-Components/4.2-Predictive-Models.md)
    *   [4.3. Training](./4.0-Key-Components/4.3-Training.md)
    *   [4.4. Evaluation](./4.0-Key-Components/4.4-Evaluation.md)
    *   [4.5. Logging](./4.0-Key-Components/4.5-Logging.md)
    *   [4.6. Persistence (Saving/Loading Models)](./4.0-Key-Components/4.6-Persistence.md)
*   **5. Extending the Framework**
    *   [5.1. Adding New Generative Processes](./5.0-Extending-the-Framework/5.1-New-Generative-Process.md)
    *   [5.2. Adding New Predictive Models](./5.0-Extending-the-Framework/5.2-New-Predictive-Model.md)
    *   [5.3. Customizing Training Loops](./5.0-Extending-the-Framework/5.3-Custom-Training-Loops.md)
*   **6. For LLM Agents**
    *   [6.1. Interacting with Configuration Files](./6.0-For-LLM-Agents/6.1-Interacting-with-Configs.md)
    *   [6.2. Programmatic Experiment Execution](./6.0-For-LLM-Agents/6.2-Programmatic-Execution.md)
    *   [6.3. Interpreting Outputs and Logs](./6.0-For-LLM-Agents/6.3-Interpreting-Outputs.md)
    *   [6.4. Example Prompts and Rules](./6.0-For-LLM-Agents/6.4-Example-Prompts-and-Rules.md)
*   **7. Troubleshooting**
    *   [7.1. Common Issues](./7.0-Troubleshooting/7.1-Common-Issues.md)

## Using this Documentation with Cursor

This documentation suite is designed to be highly effective when used within an AI-assisted coding environment like Cursor. Here are some tips for leveraging Cursor's capabilities:

*   **Interactive Q&A:**
    *   Open any documentation file (e.g., `./4.0-Key-Components/4.1-Generative-Processes.md`) directly in Cursor.
    *   Use the chat panel to ask specific questions about the content of that file. Cursor will use the opened file as direct context for its answers.
    *   Example: With `./3.0-Running-Experiments/3.2-Configuration-Hydra.md` open, ask "Can you explain the `_target_` key in more detail?"

*   **Contextual Codebase Questions:**
    *   When working on framework code (e.g., `simplexity/training/train_equinox_model.py`), you can ask Cursor questions that bridge the code and the documentation.
    *   Reference documentation files using the `@` symbol in Cursor's chat to provide precise context from the documentation.
    *   Example: "Based on the principles in `@documentation/5.0-Extending-the-Framework/5.3-Custom-Training-Loops.md`, how would I modify this training step to include gradient accumulation?"

*   **Leveraging LLM Agent Guides:**
    *   The `./6.0-For-LLM-Agents/` section, particularly `./6.0-For-LLM-Agents/6.5-Comprehensive-Agent-Guide.md` and `./6.0-For-LLM-Agents/6.4-Example-Prompts-and-Rules.md`, can be used as a basis for configuring Cursor's AI or for structuring your own prompts to get the most out of its assistance.
    *   You can copy-paste relevant parts of the "Agent Rules" from these files into your chat with Cursor to set the context for a series of questions.

*   **Navigating and Searching:**
    *   Use Cursor's file search (often `Cmd+P` or `Ctrl+P`) to quickly jump to specific documentation files by their new numbered names (e.g., `2.1-Installation.md`).
    *   For broader queries, ensure Cursor has access to the `documentation/` folder (e.g., by having it open in your workspace) so it can draw from this comprehensive knowledge base.

By combining the structured information in these documents with Cursor's AI capabilities, you can accelerate your understanding and development within the Simplexity framework.

This documentation is a work in progress. Contributions and feedback are welcome! 