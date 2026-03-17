# Refactor: Generative Process Returns from Tuples to TypedDicts

## Context

Generative processes return tuples from `generate()` and `generate_data_batch()`, making call sites fragile and unclear. This refactor:
1. Moves all public return types to **TypedDicts** for type safety, extensibility, and consistency
2. Consolidates `DataBatch` / `DataBatchWithHistory` into a **single type** with all fields always present
3. Uses **PEP 695 generic TypedDict** for `GenerateResult` to preserve the `State` type parameter
4. Keeps JAX and Torch TypedDicts **separate** for clean typing

---

## TypedDict Definitions

### `generative_process.py` — `GenerateResult[State]` (generic)
```python
class GenerateResult[State](TypedDict):
    states: State            # final post-transition state (always populated)
    observations: chex.Array # emitted tokens (always populated)
    all_states: State        # pre-transition history; pytree-preserving empty (batch, 0) per leaf when not requested
```

### `generator.py` — `DataBatch` (non-generic, State erased at this level)
```python
class DataBatch(TypedDict):
    gen_states: jax.Array | tuple[jax.Array, ...]    # final post-transition state
    inputs: jax.Array                                 # input tokens
    labels: jax.Array                                 # label tokens
    belief_states: jax.Array | tuple[jax.Array, ...]  # pre-transition history; pytree-preserving empty (batch, 0) per leaf when N/A
    prefix_probabilities: jax.Array                    # prefix probs; (batch, 0) when N/A
```

### `torch_generator.py` — `TorchDataBatch`
```python
class TorchDataBatch(TypedDict):
    gen_states: jax.Array | tuple[jax.Array, ...]    # final state (stays JAX)
    inputs: torch.Tensor                              # input tokens (torch)
    labels: torch.Tensor                              # label tokens (torch)
    belief_states: jax.Array | tuple[jax.Array, ...]  # state history (stays JAX); pytree-preserving empty (batch, 0) per leaf when N/A
    prefix_probabilities: jax.Array                    # prefix probs (stays JAX); (batch, 0) when N/A
```

### Empty Field Contract

All empty sentinels preserve the **batch dimension** and State pytree structure. Inside `generate()` (which is vmap'd), we use `jnp.empty(0, dtype=leaf.dtype)` — vmap adds the batch dim, yielding `(batch_size, 0)` per leaf.

| Field | When empty | Internal value (pre-vmap) | Caller sees (post-vmap) |
|-------|-----------|--------------------------|------------------------|
| `all_states` in GenerateResult | `return_all_states=False` | `jax.tree.map(lambda leaf: jnp.empty(0, dtype=leaf.dtype), state)` | Same pytree, each leaf `(batch_size, 0)` preserving original dtype |
| `belief_states` in DataBatch | basic `generate_data_batch()` | Reuses `result["all_states"]` (already batched) | Same pytree, each leaf `(batch_size, 0)` preserving original dtype |
| `prefix_probabilities` in DataBatch | basic `generate_data_batch()` | `jnp.empty((batch_size, 0), dtype=jnp.float32)` (explicit, not vmap'd) | `(batch_size, 0)` float32 |

---

## File-by-file Changes

### 1. `simplexity/generative_processes/generative_process.py`
- Add `GenerateResult[State]` TypedDict using PEP 695 syntax
- Change `generate()` return type to `GenerateResult[State]`
- `return_all_states=True` — capture the carry (currently discarded as `_`):
  ```python
  final_state, (all_states, obs) = jax.lax.scan(gen_states_and_obs, state, keys)
  return GenerateResult(states=final_state, observations=obs, all_states=all_states)
  ```
- `return_all_states=False` — create structure-preserving empty sentinel with original dtype:
  ```python
  final_state, obs = jax.lax.scan(gen_obs, state, keys)
  empty_states = jax.tree.map(lambda leaf: jnp.empty(0, dtype=leaf.dtype), state)
  return GenerateResult(states=final_state, observations=obs, all_states=empty_states)
  ```
- Internal scan helpers keep tuple returns (scan requires it)

### 2. `simplexity/generative_processes/independent_factored_generative_process.py`
- Import `GenerateResult`
- Same pattern as base class — `jax.tree.map` naturally handles tuple state:
  - `False`: `jax.tree.map(lambda leaf: jnp.empty(0, dtype=leaf.dtype), state)` → `tuple(jnp.empty(0, dtype=leaf.dtype), ...)`
  - `True`: capture carry, populate all fields

### 3. `simplexity/generative_processes/generator.py`
- Add `DataBatch` TypedDict
- `generate_data_batch()` → returns `DataBatch`:
  - Consume `generate()` via `result["states"]`, `result["observations"]`
  - Reuse `result["all_states"]` as `belief_states` (already batched empty from vmap)
  - Set `prefix_probabilities=jnp.empty((batch_size, 0), dtype=jnp.float32)` (explicit batch dim)
- `generate_data_batch_with_full_history()` → returns `DataBatch`:
  - Gets `gen_states` from `result["states"]` (final carry)
  - Gets `belief_states` from `result["all_states"]`
  - Populates all 5 fields

### 4. `simplexity/generative_processes/torch_generator.py`
- Add `TorchDataBatch` TypedDict
- `generate_data_batch()` → returns `TorchDataBatch`:
  - Consume JAX `DataBatch` via dict access
  - Convert inputs/labels to torch, pass through JAX state/probability fields
- `generate_data_batch_with_full_history()` → returns `TorchDataBatch`:
  - Consume JAX `DataBatch`, convert inputs/labels to torch

### 5. Test updates — call site migration

**`tests/generative_processes/test_hidden_markov_model.py`**
- `test_single_transition` (~lines 135-162): `result = z1r.generate(...)` then `result["states"]`, `result["observations"]`
- `test_generate` (~lines 165-179): same dict access

**`tests/generative_processes/test_generalized_hidden_markov_model.py`**
- `test_hmm_single_transition` (~lines 148-175): dict access
- `test_generate` (~lines 179-194): dict access
- `test_generate_with_intermediate_states` (~lines 198-213): use `result["all_states"]`

**`tests/generative_processes/test_generator.py`**
- Lines 31, 52, 81: `result = generate_data_batch(...)` then `result["gen_states"]`, `result["inputs"]`, `result["labels"]`
- Full history tests: update key names if changed

**`tests/generative_processes/test_torch_generator.py`**
- Same pattern as test_generator

**`tests/generative_processes/test_independent_factored_generative_process.py`**
- `process.generate()` → `result["states"]`, `result["observations"]`
- `return_all_states=True` → `result["all_states"]`

### 6. `tests/end_to_end/training.py`
- Inner `generate()` function (~line 131): consume `generate_data_batch()` via dict access, still returns `(inputs, labels)` tuple internally
  ```python
  def generate(step: int) -> tuple[torch.Tensor, torch.Tensor]:
      result = generate_data_batch(...)
      return result["inputs"], result["labels"]
  ```
- Line 238: `generate(0)[0]` still works (inner function returns tuple)
- `activation_tracker_step` (~line 189): `generate_data_batch_with_full_history()` now returns `TorchDataBatch` — access via same keys (no change needed)

### 7. New semantic tests (add to existing test files)

**In test_generalized_hidden_markov_model.py (or test_hidden_markov_model.py):**
- Verify `result["states"]` is the final post-transition carry when `return_all_states=True`:
  ```python
  result = model.generate(initial_states, keys, seq_len, True)
  expected_final = eqx.filter_vmap(model.transition_states)(
      result["all_states"][:, -1, :], result["observations"][:, -1]
  )
  chex.assert_trees_all_close(result["states"], expected_final)
  ```
- Verify empty sentinel preserves batch dim (array-state case; tuple-state covered in factored tests below):
  ```python
  result = model.generate(initial_states, keys, seq_len, False)
  assert result["all_states"].shape == (batch_size, 0)
  ```

**In test_generator.py:**
- Verify `gen_states` exists and has correct shape from both `generate_data_batch()` and `generate_data_batch_with_full_history()`
- Verify empty fields from basic function preserve batch dim:
  ```python
  result = generate_data_batch(states, hmm, batch_size, seq_len, key)
  assert result["belief_states"].shape == (batch_size, 0)
  assert result["prefix_probabilities"].shape == (batch_size, 0)
  ```
- Verify `gen_states` from full history has correct shape

**In test_independent_factored_generative_process.py:**
- Verify empty `all_states` is a tuple of empties (preserves FactoredState structure):
  ```python
  result = process.generate(batch_states, keys, seq_len, False)
  assert isinstance(result["all_states"], tuple)
  assert all(s.shape == (batch_size, 0) for s in result["all_states"])
  ```

---

## Migration Guide (Breaking Change)

This is a breaking change for external callers. Summary of changes:

### `GenerativeProcess.generate()` — tuple → `GenerateResult` dict
```python
# Before:
states, observations = process.generate(state, key, seq_len, False)
all_states, observations = process.generate(state, key, seq_len, True)

# After:
result = process.generate(state, key, seq_len, False)
states = result["states"]
observations = result["observations"]

result = process.generate(state, key, seq_len, True)
states = result["states"]          # final post-transition state
observations = result["observations"]
all_states = result["all_states"]  # pre-transition state history (NEW)
```

### `generate_data_batch()` — tuple → `DataBatch` / `TorchDataBatch` dict
```python
# Before:
gen_states, inputs, labels = generate_data_batch(...)

# After:
result = generate_data_batch(...)
gen_states = result["gen_states"]
inputs = result["inputs"]
labels = result["labels"]
# Also available (empty when not from full_history):
# result["belief_states"], result["prefix_probabilities"]
```

### `generate_data_batch_with_full_history()` — now returns same `DataBatch` type
```python
# Before:
result = generate_data_batch_with_full_history(...)
belief_states = result["belief_states"]  # same key
inputs = result["inputs"]               # same key

# After: same keys, plus gen_states is now also available
gen_states = result["gen_states"]  # NEW: final state
```

---

## Verification

1. `uv run --extra dev ruff check` — linting
2. `uv run --extra dev ruff format --check` — formatting
3. `uv run --extra dev --extra pytorch pyright` — type checking (generic TypedDict + all dict accesses verified)
4. `uv run --extra dev --extra pytorch pytest` — all tests pass (including new semantic tests)
