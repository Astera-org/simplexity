# LLM Context Loading Guide for Simplexity Extensions

This guide helps determine which documentation to load into an LLM agent's context based on the task.

## Context Loading Strategy

### 1. **Initial Project Setup** 
**Load:** `research_extension_summary.md` + `research_extension_template.py`
```
When the agent needs to:
- Start a new research project
- Create initial project structure
- Understand the overall approach
```

### 2. **Understanding Patterns & Philosophy**
**Load:** `research_extension_patterns.md`
```
When the agent needs to:
- Understand WHY certain design decisions were made
- Debug integration issues
- Make architectural decisions
- Understand the "spirit" of the framework
```

### 3. **Detailed Implementation**
**Load:** `research_extension_guide.md` + relevant source code
```
When the agent needs to:
- Implement specific components
- Understand configuration details
- Follow step-by-step procedures
- Access comprehensive examples
```

### 4. **Configuration Tasks**
**Load:** `research_extension_config_template.yaml` + specific config examples
```
When the agent needs to:
- Set up Hydra configurations
- Configure hyperparameter sweeps
- Wire components together
- Override default settings
```

## Task-Based Loading Patterns

### Creating a Custom Component
```
1. Start with: research_extension_summary.md (overview)
2. Add: research_extension_template.py (code structure)
3. If stuck: research_extension_patterns.md (design patterns)
4. For config: research_extension_config_template.yaml
```

### Debugging Integration Issues
```
1. Start with: research_extension_patterns.md (understand patterns)
2. Add: Specific simplexity source code
3. If needed: research_extension_guide.md (integration details)
```

### Setting Up Experiments
```
1. Start with: research_extension_config_template.yaml
2. Add: research_extension_guide.md (configuration section)
3. Include: Existing config examples from simplexity
```

### Writing Tests
```
1. Start with: research_extension_template.py (test section)
2. Add: Example tests from simplex-research
3. Include: Simplexity test patterns
```

## Context Optimization Tips

### Minimal Context Loading
For simple tasks, start with just the summary:
```python
# Task: "Create a custom data sampler"
# Load only: research_extension_summary.md + relevant base class
```

### Progressive Context Loading
Add documents as needed:
```python
# Initial: summary.md
# If confused about structure: + patterns.md
# If need implementation details: + guide.md
# If config issues: + config_template.yaml
```

### Task-Specific Combinations

**New Component Development:**
- `research_extension_template.py` (primary)
- `research_extension_summary.md` (reference)
- Simplexity base class (if extending)

**Hyperparameter Tuning Setup:**
- `research_extension_config_template.yaml` (primary)
- Hydra/Optuna examples (reference)
- `research_extension_guide.md` (section 7 only)

**Integration with Existing Code:**
- `research_extension_patterns.md` (primary)
- Relevant simplexity source files
- `research_extension_guide.md` (section 5 only)

**Testing Implementation:**
- Test examples from template
- `research_extension_patterns.md` (section 5)
- Simplexity test utilities

## Decision Tree

```
Is the agent...
├── Starting fresh?
│   └── Load: summary.md + template.py
├── Implementing a specific component?
│   └── Load: template.py + relevant base classes
├── Configuring an experiment?
│   └── Load: config_template.yaml + guide.md (config sections)
├── Debugging/troubleshooting?
│   └── Load: patterns.md + relevant source code
├── Optimizing/tuning?
│   └── Load: config_template.yaml + guide.md (section 7)
└── Understanding design decisions?
    └── Load: patterns.md + guide.md (principles section)
```

## Anti-Patterns to Avoid

1. **Don't load everything at once** - Wastes context window
2. **Don't skip the summary** - It provides essential overview
3. **Don't load guide.md for simple tasks** - It's comprehensive but verbose
4. **Don't forget source code** - Sometimes better than documentation

## Quick Reference Table

| Task Type | Primary Doc | Secondary Doc | Additional Context |
|-----------|------------|---------------|-------------------|
| Project Setup | summary.md | template.py | - |
| Custom Component | template.py | patterns.md | Base classes |
| Configuration | config_template.yaml | guide.md §2 | Existing configs |
| Integration | patterns.md | guide.md §5 | Source code |
| Testing | template.py | patterns.md §5 | Test examples |
| Debugging | patterns.md | Source code | Error messages |
| Optimization | config_template.yaml | guide.md §7 | Optuna docs |

## Example Prompts with Context

### Efficient Context Usage
```
"Create a custom state sampler for my experiment"
Context: research_extension_summary.md + NonergodicStateSampler example
```

### Comprehensive Context Usage
```
"Design a complete experiment with custom components and hyperparameter tuning"
Context: All docs except patterns.md (unless architectural decisions needed)
```

Remember: Start minimal, add context as needed. The agent should request additional context if it needs more information. 