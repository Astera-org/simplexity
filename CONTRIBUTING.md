# Contributing Guidelines

This document outlines requirements and best practices for contributing code to this repository. We use a two-tier review process with different standards for `dev` and `main` branches.

## Branch Strategy

- **`dev`**: Integration branch for ongoing work. Code here should be structurally sound and tested, but may still be evolving.
- **`main`**: Production-ready code. Higher bar for test coverage, implementation quality, and static analysis compliance.

## Requirements by Target Branch

### Merging into `dev`

PRs targeting `dev` must meet the following criteria:

**Testing**
- All existing tests must pass
- No regressions in functionality

**Design**
- Interfaces and data structures should be well-considered and expected to remain stable
- Public APIs should be designed with future extensibility in mind
- Avoid patterns that will require breaking changes later

**Static Analysis**
- Strive to pass formatting (`ruff format`)
- Strive to pass linting (`ruff`)
- Strive to pass type checking (`pyright`)
- Minor violations may be accepted with justification

### Merging into `main`

PRs targeting `main` must meet all `dev` requirements plus:

**Testing**
- New tests required for new functionality
- Comprehensive coverage of edge cases and failure modes
- Coverage metrics should not decrease

**Implementation**
- Code will be scrutinized for correctness, efficiency, and maintainability
- Algorithms and logic should be well-documented
- Error handling must be robust

**Static Analysis**
- All checks must pass
- Any `# type: ignore`, `# noqa`, or equivalent suppressions require explicit justification in the PR description
- No new warnings or errors permitted

## PR Process

### Before Opening a PR

1. Run the test suite locally: `uv run --extra dev --extra pytorch pytest`
2. Run static checks:
   ```bash
   uv run --extra dev ruff format --check .
   uv run --extra dev ruff check .
   uv run --extra dev --extra pytorch pyright
   ```
3. Ensure your branch is up to date with the target branch

### PR Description

Include the following in your PR description:

- **Summary**: What does this change do?
- **Motivation**: Why is this change needed?
- **Testing**: How was this tested?
- **Breaking Changes**: Any breaking changes to public APIs?
- **Suppressions** (if any): Justification for any ignored static analysis rules

### Review Criteria

Reviewers will evaluate:

| Criterion | `dev` | `main` |
|-----------|-------|--------|
| Existing tests pass | Required | Required |
| New test coverage | Encouraged | Required |
| Interface stability | Required | Required |
| Implementation quality | Reviewed | Scrutinized |
| Formatting | Should pass | Must pass |
| Linting | Should pass | Must pass |
| Type checking | Should pass | Must pass |

## Style Guidelines

- Follow existing code conventions in the repository
- Prefer explicit over implicit
- Write docstrings for public functions and classes
- Keep functions focused and composable

## Questions?

If you're unsure whether your contribution meets these standards, open a draft PR early to get feedback before investing too much time.