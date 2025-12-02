# Selective Test Execution with pytest-testmon

This directory contains GitHub Actions workflows for the Simplexity project, including selective test execution using pytest-testmon.

## Workflows

### `simplexity.yaml`
Main CI workflow that runs on all PRs and pushes to main:
- **Static Analysis**: Runs ruff, pylint, and pyright on all code
- **Unit Tests with Selective Execution**:
  - **On PRs**: Runs only tests affected by code changes using `pytest --testmon`
  - **On main branch**: Runs full test suite using `pytest --testmon-noselect`
  - Coverage reporting via diff-cover (PRs only) and Codecov

### `merge-tests.yaml`
Merge queue workflow that runs before allowing merges:
- Triggers on GitHub merge queue events
- Runs the complete test suite without test selection
- Must pass before PR can be merged to main
- Provides final validation before code reaches main branch

### `claude.yml` and `claude-code-review.yml`
AI-assisted code review workflows (see individual files for details)

## pytest-testmon

### What is it?
pytest-testmon is a pytest plugin that intelligently selects which tests to run based on code changes. It uses Coverage.py to track which code each test executes, building a database of test-code dependencies.

### How it works
1. **Database Building**: On first run, testmon tracks which lines of code each test executes
2. **Change Detection**: On subsequent runs, it detects which code has changed
3. **Test Selection**: It runs only tests that previously executed the changed code
4. **Database Update**: After each run, it updates its knowledge of test-code relationships

### Cache Strategy
- The `.testmondata` database is cached in GitHub Actions using `actions/cache`
- Cache key includes OS, branch name, and hash of all Python files
- Fallback keys ensure cache hits across branches and temporal changes
- Database is preserved across CI runs for intelligent test selection

## Running Tests Locally

### Run affected tests only
```bash
pytest --testmon
```
Only runs tests affected by your code changes since the last run.

### Run all tests and update database
```bash
pytest --testmon-noselect
```
Runs all tests but still updates the testmon database with coverage information.

### Clear testmon database
```bash
pytest --testmon-nocache
```
Clears the testmon database and rebuilds it from scratch.

### Disable testmon temporarily
```bash
pytest
```
Regular pytest without testmon (database not updated).

## Benefits

### For CI/CD
- **Faster PR builds**: Run only affected tests, reducing CI time
- **Comprehensive merge validation**: Full test suite runs before merge
- **Smart caching**: testmon database cached between runs for consistency
- **Accurate selection**: Based on actual code coverage, not file structure assumptions

### For Local Development  
- **Quick iteration**: Run only relevant tests while developing
- **Confidence**: Full suite still runs on main and in merge queue
- **No manual management**: testmon automatically tracks dependencies

## Configuration

testmon requires no special configuration for Simplexity. Test selection happens automatically based on:
- Code changes detected via Coverage.py
- Historical test-code dependency mapping in `.testmondata`
- Standard pytest configuration in `pyproject.toml`

## Troubleshooting

### Tests not being selected when they should
- Clear the database: `pytest --testmon-nocache`
- Ensure your changes actually affect code that tests execute
- Check that `.testmondata` exists and isn't corrupted

### All tests running on every PR
- Check GitHub Actions cache is working
- Verify testmon was installed: `uv run pip show pytest-testmon`
- Look for testmon output in test logs

### Want to force all tests in a PR
Add a comment in your PR or push an empty commit:
```bash
git commit --allow-empty -m "Run full test suite"
git push
```
Then temporarily modify the workflow to use `--testmon-noselect`.
