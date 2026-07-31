# Migrating to a vendored generative process

Generative processes are moving out of simplexity and into your own project, sourced from
[generators](https://github.com/ealt/generators). This guide shows how to run a vendored
process under `@managed_run()`.

Nothing is removed yet. `simplexity.generative_processes` remains fully supported and emits only a
`PendingDeprecationWarning`; full deprecation waits until generators reaches feature parity. If your
code works today, it keeps working, and you can migrate when it suits you.

## Why copy instead of import

Generators is not a library and should not be added to your dependencies. You copy the modules you
need into your project, delete what you do not use, and modify the rest until it fits your problem.
The process code then lives where you can read, instrument, and optimize it, and your experiment
artifacts carry the exact code that produced them. The invariant that survives your edits is not
the code but the specification: generators' `SPEC.md` defines the required results, and its
conformance vectors are how you check that your copy still implements the same mathematics.

Simplexity supports this by identifying a generative process by the operations it provides rather
than by where its code lives.

## The contract

Your process must provide `simplexity.run_management.protocols.GenerativeProcess`:

| Member | Meaning |
| --- | --- |
| `vocab_size` | Number of emittable observations. Used to resolve `vocab_size` in your config and `d_vocab` in your model's. |
| `initial_state` | The state generation starts from. |
| `emit_observation(state, key)` | Sample one observation. |
| `transition_states(state, obs)` | Update the state after an observation. |
| `observation_probability_distribution(state)` | Distribution over observations from a state. |
| `probability(observations)` | Probability of a sequence. |
| `generate(state, key, sequence_len, return_all_states)` | Generate a batch of sequences, optionally returning every belief state. |

It is a `Protocol`, so you do not inherit from anything — an object with these members conforms.

Belief-state analysis additionally needs `LogSpaceGenerativeProcess`
(`log_observation_probability_distribution`, `log_probability`). Generators does not provide
log-space operations, so add them only if your run does belief analysis; training and generation
never call them.

## Steps

### 1. Copy the modules

Copy what you need from generators into your project, fix the module-level imports to your own
paths, and delete the rest. `ghmm/process.py` plus `utils.py` is the minimal unit; add
`factored/*.py` for composite processes. Copy the transition-matrix constructors you want from
`transition_matrices/`, or write your own — those are data, not behaviour.

Record the upstream commit you copied from, so you can diff against it later to pick up
improvements.

### 2. Write an adapter

Generators exposes module-level functions over a `NamedTuple` of process data, so bind that data
once and expose the operations as methods. `tests/vendored_process/adapter.py` in this repo is a
complete worked example, around fifty lines for a GHMM.

Two things usually need attention:

- **Generation.** Generators' `generate` returns only the final state. Simplexity's training path
  also wants every intermediate belief state, so re-scan rather than calling it, and batch with
  `eqx.filter_vmap` (or an explicit `jax.vmap`) as the example does.
- **Composite processes.** `factored/*.py` takes keyword-only `encode` and `decode` callables
  because generators deliberately does not own the composite-token encoding. Bind your choice in
  the adapter. Note also that `factored`'s `seq_prob` takes a state, unlike the GHMM's, so your
  `probability` should bind the initial state.

### 3. Write a factory

Assembling a process is code — constructing matrices, calling `init`, binding `encode`/`decode` —
and none of that fits in YAML. Give your adapter module a factory function for the config to
target:

```python
def build_my_process(x: float, a: float) -> MyVendoredProcess:
    return MyVendoredProcess(data=ghmm_process.init(mess(x, a, 3)))
```

This mirrors how simplexity's own `builder.build_hidden_markov_model` is configured.

### 4. Declare it in your config

Point `_target_` at your factory, and set `component: generative_process` so run management knows
what the section configures. The declaration is required for processes outside
`simplexity.generative_processes`: your process lives in your own namespace, so it cannot be
recognized from its import path.

```yaml
name: my_vendored_process
component: generative_process
instance:
  _target_: my_project.processes.adapter.build_my_process
  x: 0.15
  a: 0.6

base_vocab_size: ???
bos_token: ???
eos_token: null
vocab_size: ???
```

The `???` fields resolve from your process's `vocab_size`, exactly as for a simplexity process, and
feed `d_vocab` in your model's config.

### 5. Wire up conformance vectors

Copy the relevant conformance vectors from generators into your test suite and run them against
your copy: green means your fork still implements the specification.

Be aware that generators specifies the vector encoding in `SPEC.md` section 7 but does not yet ship
the vectors themselves. Until it does, this safety net is unavailable, and the protocol check will
not cover it — `isinstance` against a protocol verifies that members exist, not that they compute
the right thing. Until vectors exist, test your copy against whatever ground truth you have.

### 6. Run

Your entrypoint is unchanged. `components.get_generative_process()` returns your process, and the
generation helpers accept it:

```python
from simplexity.generative_processes.torch_generator import generate_data_batch
```

## Troubleshooting

**"no generative process configs found", and `get_generative_process()` returns `None`.** Your
section was not claimed. Check that `component: generative_process` is set on the section itself,
not on the nested `instance`. The log message lists the instance keys that were considered.

**"config declares generative processes at ..., but their configs are invalid".** The section
declared itself but failed validation. The preceding validation warnings say why. This is
deliberately an error rather than a skip: a declared process that quietly does not load used to
leave the run with no process at all.

**"... is not a generative process: missing ...".** Your factory returned something that does not
provide the whole contract; the message names the absent members. A common cause is a typo in a
method name, since conformance is structural.

## Reference

- `docs/design/generators_instantiation.md` — the design, the tradeoffs considered, and the
  outstanding gaps between generators and this package.
- `tests/vendored_process/` — a checked-in vendored process and its adapter.
- `tests/end_to_end/test_vendored_process_training.py` — that process training under
  `@managed_run()`.
