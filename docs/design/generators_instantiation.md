# Instantiating vendored generative processes under `managed_run`

**Status:** design proposal, awaiting direction
**Scope:** how `simplexity`'s run management instantiates generative processes that live in the
*consumer's* repository, as a precondition for deprecating `simplexity/generative_processes`
in favour of [`generators`](https://github.com/Astera-org/generators)
**Branch:** `feat/generators-instantiation`

---

## 1 Summary

`generators` distributes generative-process code by **copying**: consumers vendor the modules into
their own project and modify them freely. `simplexity`'s runner instantiates components from Hydra
config. Those two facts are compatible. What is *not* compatible is one specific mechanism:
`simplexity` decides whether a config section is a generative process by testing whether its
`_target_` string starts with `"simplexity.generative_processes."`. Vendored code can never satisfy
a namespace test, because refusing to be a shared namespace is the entire point of the vendoring
model.

So the tension is narrower than "config-driven instantiation vs. a library that refuses to be a
dependency." Hydra instantiating consumer-local code is not a problem — it already happens for
`torch.optim.Adam` and `HookedTransformer` on every run. **The problem is that
`generative_process` is the one component whose discovery predicate is a namespace prefix and whose
instantiation asserts a `simplexity` base class.** Fixing it means replacing a *nominal* test
(where does this code live?) with a *structural* one (what can this object do?) — which is what
`generators`' own SPEC licenses:

> §2 Operations — "The spec defines what results are required, not how they are organized or
> computed."

**Recommendation, in one line:** make the process contract a `runtime_checkable` Protocol keyed on
the SPEC's operations, let the config section *declare* itself as a generative process instead of
being sniffed by namespace, and make a declared-but-nonconforming section a hard error instead of a
silent skip. Deprecate `simplexity/generative_processes` by warning and documentation, not
deletion, and only after the four capability gaps in §4.3 have homes.

Measured up front, so the rest of this document argues from evidence rather than expectation
(§3 has the reproduction):

| Question | Measured answer |
| --- | --- |
| Does a vendored `generators` process work with `simplexity`'s generation path today? | **Yes** — `generate_data_batch` runs on it unmodified, correct shapes, normalized `obs_dist`, correct `seq_prob`. |
| How much of the `GenerativeProcess` interface does it miss? | **2 of 8 members**: `log_observation_probability_distribution`, `log_probability`. Nothing on the training path. |
| What actually blocks it? | `is_generative_process_target` → `False`, so the config key is filtered out before instantiation. |
| What does the user see when that happens? | One `INFO` line, `[generative process] no generative process configs found`. **No warning, no error.** |
| Does a structural Protocol also accept the existing in-repo processes? | **Yes** — same predicate accepts `build_hidden_markov_model`'s output, so this is not a breaking change. |

---

## 2 The tension, precisely

### 2.1 What the runner does today

Four places in `simplexity/run_management/run_management.py` participate:

| Step | Code | Mechanism |
| --- | --- | --- |
| **Discover** which config sections are processes | `_setup_generative_processes` → `filter_instance_keys(..., is_generative_process_target, ...)` (`run_management.py:348`) | `target.startswith("simplexity.generative_processes.")` (`structured_configs/generative_process.py:388`) |
| **Validate** the config | `validate_generative_process_config` (`structured_configs/generative_process.py:401`) | dispatches on the exact `_target_` of five known builders; the `else` branch re-requires the namespace prefix (line 426) |
| **Instantiate** | `_instantiate_generative_process` (`run_management.py:330`) | `typed_instantiate(instance_config, GenerativeProcess)` → `assert isinstance(obj, GenerativeProcess)` (`utils/config_utils.py:145`) |
| **Resolve vocab metadata** | `_get_attribute_value` (`run_management.py:409`) | re-runs the *same* namespace filter to find `vocab_size` / `bos_token` for the predictive model |

Note the fourth row: the namespace filter is not only how the process is built, it is also how
`vocab_size` reaches the predictive-model config (`_setup_predictive_models`, `run_management.py:470`).
A process the filter rejects therefore fails **twice** — no process, and no vocab-size resolution.

### 2.2 Why `generators` cannot satisfy it

`generators/USAGE.md` opens:

> This repo is not a library. Do not add it to your dependencies. Copy the modules you need into
> your codebase.

and states the mechanism explicitly: "The unit of reuse is the module, and the mechanism is
copying… The invariant that survives your modifications is not the code — it is the spec." The
repo has empty `__init__.py` files and no registry "coupling modules together" on purpose.

A vendored process therefore has:

- a `_target_` under the *consumer's* package name, never `simplexity.*`;
- a shape the consumer has deliberately changed (deletion is the stated specialization mechanism);
- no class at all — `generators` is **module-level functions over a `NamedTuple`**, e.g.
  `generators/ghmm/process.py` exposes `init(Ts) -> Data`, `obs_dist(data, eta)`,
  `sample(data, eta, key)`, `update(data, eta, x)`, `generate(data, eta, keys)`,
  `seq_prob(data, xs)`. There is nothing for `isinstance(obj, GenerativeProcess)` to be true of.

### 2.3 The reframe

`simplexity` uses a namespace prefix as a proxy for a capability. That proxy is *sound* while all
processes live in `simplexity`, and *unfixable* once they deliberately do not. Every option below
is a different answer to "what replaces the proxy?"

Worth noting: **`simplexity` has already made this move for two other components.**

- `is_predictive_model_target` (`structured_configs/predictive_model.py:192`) accepts any
  `*.nn.*` or `*.models*` target — `torch.nn`, `equinox.nn`, `penzai.models`.
- `is_optimizer_target` (`structured_configs/optimizer.py:77`) accepts `torch.optim.*` and `optax.*`.
- `_instantiate_predictive_model` (`run_management.py:437`) calls bare `hydra.utils.instantiate`
  with **no** type assertion at all — the predictive-model contract is already fully structural.

So relaxing the generative-process contract is not a new pattern in this repo. It is bringing the
most rigid component filter into line with how the repo already treats the two components whose
implementations live upstream. (Both existing relaxations are still *namespace sniffing*, just with
a wider net — `parts[1] == "nn"` would not accept a consumer-local module either. §5.2 argues that
sniffing is the wrong axis regardless.)

---

## 3 The measured interface delta

The brief asked for the actual delta rather than an assumed one. Reproduction: vendor
`generators/ghmm/process.py` + `generators/utils.py` + `transition_matrices/classical.py` into a
throwaway package, write a ~50-line adapter binding the module functions to method names, and run
it through `simplexity`'s real code paths.

### 3.1 Operation-by-operation mapping

| `GenerativeProcess` member | `generators` (ghmm) | Delta |
| --- | --- | --- |
| `vocab_size` | `data.Ts.shape[0]` | derived, trivial |
| `initial_state` | `data.eta_0` | field on `Data` |
| `emit_observation(state, key)` | `sample(data, eta, key)` | same math, `data` threaded as an argument |
| `transition_states(state, obs)` | `update(data, eta, x)` | same |
| `observation_probability_distribution(state)` | `obs_dist(data, eta)` | same |
| `probability(observations)` | `seq_prob(data, xs)` | same for ghmm; **signature diverges** for factored (`seq_prob(data, eta, xs, *, decode)`) |
| `generate(state, key, len, return_all)` | `generate(data, eta, keys)` | `generators` returns final state only; `return_all_states=True` needs a 3-line scan change. Batching is the consumer's (`eqx.filter_vmap` vs. explicit `vmap`) |
| `log_observation_probability_distribution` | — | **absent** |
| `log_probability` | — | **absent** |

Two genuine gaps out of eight members, and both are log-space. Their call sites are worth
knowing before deciding whether the contract must require them:

- `log_observation_probability_distribution` is consumed by `mixed_state_presentation.py:512`
  (the mixed-state-tree / belief-enumeration machinery) and by the composite processes that
  forward to sub-processes (`nonergodic_generative_process.py:254`,
  `inflated_vocabulary_process.py:88`, `factored_generative_process.py:148`).
- Neither log-space method is called anywhere on the **training or generation path** —
  not by `generator.py`, `torch_generator.py`, `run_management.py`, or `tests/end_to_end/training.py`.

That asymmetry is the evidence for a **tiered contract** (§5.3): the operations training needs and
the operations belief-analysis needs are different sets, and today's ABC forces every process to
implement both.

### 3.2 The blocking mechanism, measured

Running the vendored adapter through the real predicates:

```
1. isinstance(vendored, GenerativeProcess) = False
2. is_generative_process_target('vendored.adapter.build_vendored_mess3') = False
3. instance keys discovered = ['generative_process.instance']
   keys surviving the generative-process filter = []
   -> components.generative_processes would be None (SILENT skip)
   validate_generative_process_config raised ConfigValidationError:
       GenerativeProcessConfig.instance must be a generative process target
5. GenerativeProcess abstract members = ['emit_observation', 'initial_state',
   'log_observation_probability_distribution', 'log_probability',
   'observation_probability_distribution', 'probability', 'transition_states', 'vocab_size']
   missing on the vendored adapter = ['log_observation_probability_distribution', 'log_probability']
4. generate_data_batch on the vendored process: inputs (4, 16), labels (4, 16)
   vocab_size=3  token range=[0,3]
   obs_dist(initial) = [0.33333325 0.3333333  0.3333333 ]  sums to 1: True
   probability(xs) = 0.0028251
```

Read line 4 first: **the vendored process already works.** `generate_data_batch` — the function
`tests/end_to_end/training.py` and three of the four external consumers call — runs on it
unmodified and produces correct results. No change to `simplexity`'s *runtime* code is required
for a vendored process to train. The work is entirely in discovery and validation.

### 3.3 The failure mode is silent

`filter_instance_keys` (`utils/config_utils.py:67`) short-circuits:

```python
if isinstance(target, str) and filter_fn(target) and _validate(cfg, instance_key, validate_fn, ...):
```

Because `filter_fn` returns `False` first, `_validate` is never called, so the
`ConfigValidationError` above is **never raised in the real path** and its warning is never logged.
Confirmed by driving `_setup_generative_processes` directly at `DEBUG`; the complete output is:

```
INFO simplexity: [generative process] no generative process configs found
RESULT: None
```

A config that names a vendored process is indistinguishable from a config that has no process at
all. `managed_run` proceeds, `components.get_generative_process()` returns `None`, and the run
fails somewhere downstream — or, if the entrypoint is tolerant, does something wrong quietly.
**This is a defect independent of which option below is chosen, and worth fixing either way:** a
config section typed as `GenerativeProcessConfig` that yields no process should be loud.

### 3.4 Assembly is not expressible in YAML

`generators`' composite modules take **keyword-only callables** the consumer must supply:

```python
def obs_dist(data, eta, *, decode: Callable[[jax.Array], jax.Array]) -> jax.Array
def generate(data, eta, keys, *, encode: Callable[[jax.Array], jax.Array]) -> tuple[...]
```

(`generators/factored/chain.py:106,164`; same in `complete.py`, `independent.py`,
`nonergodic/factored.py`.) The composite-token encoding is deliberately not owned by the module —
SPEC §4.3.2–4.3.3 specify radix encoding but the module takes it as an argument. This is what
`ZEN.md`'s "Some assembly required" means concretely.

**Consequence for config design:** a vendored process cannot generally be described by a `_target_`
plus scalar kwargs, because part of its definition is *code the consumer writes*. Whatever contract
we choose, the config for a vendored process should point at a **project-local factory** that does
the assembly and returns the finished object. That is not a new mechanism — `simplexity`'s own
`_target_: simplexity.generative_processes.builder.build_hidden_markov_model` is exactly this
pattern, and it is what the spike used.

---

## 4 The strategic fork (please resolve first)

"Deprecate `generative_processes` in favour of `generators`" has two readings, and they imply
different work. Everything downstream depends on which one is intended.

### 4.1 Reading A — `simplexity` becomes a runner, not a process home

`simplexity` stops being where processes live. It keeps the runner, the training/eval/analysis
machinery, and a *contract*; processes live in each consumer's repo as vendored `generators`
modules. `simplexity/generative_processes` gains deprecation warnings, stops growing, and
eventually thins to the protocol plus the consumer-side utilities that operate on it
(`generator.py`, `torch_generator.py`, `mixed_state_presentation.py`).

This is what the brief describes and what `generators`' USAGE.md argues for — the coordination
point disappears; research directions decouple.

### 4.2 Reading B — `simplexity` vendors `generators` internally

`simplexity` copies `generators`' modules into `simplexity/generative_processes` and reimplements
its process classes on top of them, so in-repo processes become spec-conformant.

This is worth naming because it is the tempting middle path, and it **defeats the purpose for
downstream consumers**: `simplexity` would still be the shared library that everyone imports
processes from, still be the merge queue, still be frozen by the reproducibility obligation. It
buys spec conformance for `simplexity`'s own copies and nothing else. It is not obviously wrong —
if the goal is only "adopt `generators`' math," it is cheaper — but it does not deliver the
decoupling.

**Recommendation: Reading A.** The rest of this document assumes it.

### 4.3 What Reading A owes: `generators` is not a superset

Deprecation implies a replacement exists. Measured against `simplexity/generative_processes`
(3,754 lines across 21 modules), `generators` does not yet cover four things. These are not
objections — they are work that needs an owner before any removal:

1. **Mixed-state presentation / belief enumeration.** `mixed_state_presentation.py` (576 lines) —
   `MixedStateTree`, `MixedStateTreeGenerator`, myopic-entropy computation. `generators` has no
   counterpart (SPEC covers belief *updates*, not tree enumeration). This is load-bearing for
   the interpretability work: it is what produces the ground-truth belief geometry that
   `activation_tracker` compares activations against, and `eden-experiments` imports it directly
   (`fraxl-run/salvage-candidate/scratch/sweep.py:17`, `nonergodic-demo/nonergodic_entropy.py:106`).
2. **Log-space operations.** The two missing members of §3.1. Required by the MSP path for
   numerical stability over long sequences.
3. **Framework bridges and augmentation.** `torch_generator.py` (torch tensors / device
   placement), `data_prefetcher.py`, and BOS/EOS handling in `generator.py`. SPEC §6.2 specifies
   augmentation behaviorally and Appendix A marks device/DLPack interop explicitly *optional* and
   out of conformance scope, so this is consumer-side by design — but it currently lives in the
   package being deprecated.
4. **Process zoo breadth.** `transition_matrices.py` has 16 process families (`rrxor`, `fanizza`,
   `tom_quantum`, `post_quantum`, `days_of_week`, `even_ones`, `matching_parens`,
   `no_consecutive_ones`, `mr_name`, `sns`, `coin`, `leaky_rrxor`, …). `generators`'
   `transition_matrices/` has `zero_one_random`, `cycle`, `mess`, `bloch_walk`, `moon`, plus
   `expand_vocab`/`compress_vocab`. Note these are *matrices* — data, not behavior — and
   `generators`' USAGE.md says "Process definitions are your data, not our code… the repo does not
   own a process zoo," yet it ships `transition_matrices/` anyway. The status of this directory is
   genuinely ambiguous and it is the most reusable, least controversial part of what would be
   deprecated (see open question Q5).

There is also a **missing safety net**: SPEC §7 defines the JSON encoding for conformance test
vectors, and USAGE.md's consumption recipe step 2 is "copy the relevant conformance vectors and
wire them into your test suite." **No vector files exist in the `generators` repo yet** — no JSON
fixtures, no test referencing them. The invariant that is supposed to survive a consumer's
modifications is currently unenforceable. That matters directly here: a structural contract is
only as safe as the consumer's ability to prove their fork still implements the same math.

---

## 5 Options

Three independent axes. The brief's three options map onto them; presenting them as axes rather
than as three rival packages avoids a false choice, because the brief's option 3
(factory/callable config) is not an alternative to options 1 and 2 — it is the config *shape*
either of them needs (§3.4).

### 5.1 Axis A — what is the contract?

| | **A1: keep the nominal ABC** | **A2: `runtime_checkable` Protocol** | **A3: no check (predictive-model style)** |
| --- | --- | --- | --- |
| Mechanism | consumer subclasses `simplexity.generative_processes.generative_process.GenerativeProcess` | consumer's object structurally satisfies a Protocol; verified at instantiation | bare `hydra.utils.instantiate`, duck-typed at use site |
| Hydra ergonomics | unchanged | unchanged | unchanged |
| Type checking | strongest (nominal) | good — pyright checks conformance statically *if* the consumer annotates | none |
| Preserves `generators`' point? | **No** — requires importing the deprecated package and inheriting from it; recreates the shared coordination surface | **Yes** — a Protocol is a type-only dependency; nothing behavioral is shared, consumer may still delete/rename/restructure freely | Yes, maximally |
| Failure mode | clear (`TypeError` on abstract methods) | clear if we raise (`isinstance` False → explicit error) | **poor** — `AttributeError` deep inside a `jit`-traced scan, far from the config that caused it |
| Reproducibility asserts | unaffected | unaffected | unaffected |
| Verified? | n/a | **yes** — accepts vendored adapter *and* native `HiddenMarkovModel`, rejects a bogus object (§3.2, §5.5) | n/a |

A1 is disqualified by the requirement, not by taste: inheriting from a class inside the package
being deprecated is a behavioral dependency on that package. A3 is what the repo already does for
predictive models, so it is defensible on consistency grounds, but it trades a config-time error
for a runtime one — and given that the current failure is *already* an invisible `None` (§3.3),
adding more silence is the wrong direction.

**Recommend A2.**

A Protocol's honest limits, since they should be stated rather than discovered later:
`isinstance` against a `runtime_checkable` Protocol checks **member presence only** — not
signatures, not shapes, not semantics. It will not catch a `transition_states` with the arguments
reversed, a `vocab_size` that lies, or an `obs_dist` that returns unnormalized values. It is a
smoke test that replaces a *namespace* check, not a correctness proof. What actually protects
correctness is the SPEC's conformance vectors run against the consumer's copy — which is why their
absence (§4.3) is a real risk and Q3 below matters.

### 5.2 Axis B — how does the runner discover the component?

The `_target_` string is the wrong discriminator for consumer-local code, and no widening of the
prefix test fixes that: any prefix wide enough to admit arbitrary consumer packages admits
everything, including the logger and the optimizer.

| | **B1: widen the prefix** | **B2: declared in config** | **B3: prefix fast path + explicit declaration** |
| --- | --- | --- | --- |
| Mechanism | add more accepted prefixes | the config section declares its kind; namespace ignored | in-repo targets keep the prefix path; consumer-local sections opt in with an explicit marker |
| Works for arbitrary consumer packages? | no | yes | yes |
| Existing configs change? | no | **yes** — every process config needs the new field | no |
| Risk | unbounded — a wide prefix mis-claims other components | migration churn across 11 in-repo configs + teammates' branches | small: two paths to maintain |

B2 is the clean end-state and it is *already half-true*: `_instantiate_generative_process` and
`_get_attribute_value` both derive `config_key = instance_key.rsplit(".", 1)[0]` and assume the
parent section is the process config block, and that section is already typed as
`GenerativeProcessConfig` in every entrypoint's structured config
(`tests/end_to_end/training.py:66`). The target-string filter is largely **redundant with the
structured-config schema** that already declares intent.

**Recommend B3** — B2's mechanism, added without breaking the 11 existing in-repo configs or
teammates' in-flight branches. Concretely: `is_generative_process_target` keeps returning `True`
for `simplexity.generative_processes.*`, and the section may additionally opt in via an explicit
field on `GenerativeProcessConfig`. A section that opts in and then fails the Protocol check is a
**hard error**, not a filtered key — that is the §3.3 fix.

### 5.3 Axis C — what surface does the contract require?

Today's ABC demands all eight members from every process. §3.1 measured that the training path
uses six and the belief-analysis path uses the other two.

| | **C1: all eight** | **C2: core + optional extensions** |
| --- | --- | --- |
| Required of a vendored process | 8 members incl. log-space | 6 core; log-space only if the run uses MSP/analysis |
| Cost to consumer | must hand-write log-space ops `generators` does not provide | copies what the SPEC covers; adds log-space when analysis needs it |
| Enforcement | one `isinstance` | core checked at instantiation; extension checked where used (MSP generator, composite processes) |
| Honesty | over-claims — asserts capabilities training never exercises | matches measured usage |

**Recommend C2**, with the core protocol keyed on SPEC §2 + §6.1 (obs dist, sampling, state update,
sequence probability, generation) plus the two metadata members the runner itself needs
(`vocab_size` for vocab resolution, `initial_state` for batch expansion). This also makes the
deprecation *smaller*: log-space is an analysis concern, so it belongs with the analysis code that
consumes it, not in the base contract.

### 5.4 Recommended shape, concretely

1. **New protocol module outside the deprecated package** — the contract must not live in
   `simplexity/generative_processes/`, or consumers still import from the thing being deprecated.
   Suggested home: `simplexity/run_management/` (alongside `components.py`, which is what
   type-annotates the object) or a new top-level `simplexity/processes/`. Naming to be settled;
   `Protocol` is already an established idiom in this repo
   (`generative_processes/structures/protocol.py:39`).
2. **`GenerativeProcess` (the ABC) structurally satisfies the new Protocol** — so every existing
   process passes unchanged, and `Components.generative_processes` retypes to the Protocol with no
   change to in-repo processes or their configs. Verified: §5.5.
3. **Discovery** via B3: prefix fast path, plus explicit opt-in for consumer-local sections.
4. **Validation** raises on a declared-but-nonconforming section, naming the missing members.
   Replaces the silent skip.
5. **Vendored configs point at a project-local factory** (§3.4), documented with a worked example.
6. **Acceptance test:** a vendored `generators` module under `tests/` trains end-to-end under
   `@managed_run()` — the brief's real bar. The spike (§3) is that test's skeleton; it needs
   promoting from a throwaway to a fixture, with the vendored copy checked in so the test does not
   depend on `generators` being present.

**Incidental cleanups this touches** (flagging rather than silently bundling):

- `_setup_generative_processes` (`run_management.py:359-365`) calls `_instantiate_generative_process`,
  which already calls `resolve_generative_process_config`, and then resolves again with the same
  arguments. Harmless today (the second pass hits the equality branch) but redundant, and it is
  the natural thing to unify while restructuring this function rather than adding a fourth
  variation of the same block.
- `typed_instantiate` (`utils/config_utils.py:145`) enforces its contract with a bare `assert`,
  which is stripped under `python -O`. If it becomes the Protocol check, it should raise.

### 5.5 What was verified vs. what is still a claim

Verified by execution against the real code (§3.2, and a Protocol discrimination check):

- vendored `generators` process runs through `generate_data_batch` correctly;
- exactly two ABC members are missing, both log-space, neither on the training path;
- the current failure is a silent `None` with no warning;
- a `runtime_checkable` core Protocol accepts the vendored adapter **and** the native
  `build_hidden_markov_model` output, and rejects a non-conforming object — i.e. the relaxation is
  back-compatible by construction, not by hope.

Not yet verified — the honest gaps in this proposal:

- a **factored / nonergodic** vendored process end-to-end (the spike used ghmm, the simplest case;
  the `encode`/`decode` assembly of §3.4 is where this gets interesting, and `probability`'s
  signature genuinely diverges there);
- the full `managed_run` path with MLflow + reproducibility asserts under `strict=True` (the spike
  drove `_setup_generative_processes` directly);
- `activation_tracker` against vendored-process beliefs, which is where the MSP gap (§4.3) would
  actually bite.

---

## 6 Deprecation path

`simplexity` is a shared repo with real users and a standing reproducibility obligation —
`generators`' own USAGE.md names it ("because past research must remain reproducible, even internal
APIs become effectively frozen"). The greenfield "no back-compat shims" rule does not apply.
Nothing in `generative_processes` gets deleted or broken by this work.

The repo's existing deprecation idiom is a comment plus continued function — `pyproject.toml` marks
`aws = [...] # Deprecated S3 Persister.` and `penzai = [...] # Deprecated: penzai is no longer
maintained.` while both keep working. There is **no** `DeprecationWarning` anywhere in
`simplexity/` today, so introducing runtime warnings is a new (and teammate-visible) convention —
hence Q4.

Proposed staging, each stage independently landable:

| Stage | Content | Breaks anything? |
| --- | --- | --- |
| **0. Enable** (this PR) | protocol + discovery + loud validation + vendored end-to-end test + migration guide | No. Existing configs and processes untouched; all existing tests are the back-compat proof. |
| **1. Signpost** | `docs/` migration guide, README pointer to `generators`, module docstring notes. Package still fully supported. | No |
| **2. Discourage** | new processes go in consumers, not here. Optionally a `DeprecationWarning` on import of `generative_processes.builder` — **only** once teammates have agreed (Q4), since it fires in everyone's runs. | No, but noisy |
| **3. Re-home the gaps** | §4.3 items 1–3 move out of `generative_processes` into consumer-side homes that operate on the Protocol (`analysis/`, `run_management/`); item 4 resolved per Q5 | Import paths move — needs coordination |
| **4. Thin** | remove what has no consumers, once `eden-experiments` and teammates' branches have migrated | Yes — needs a deprecation window and sign-off |

Stages 3–4 are explicitly **not** in this PR's scope; naming them is what keeps stage 0 honest
about being a beginning rather than a fait accompli.

### 6.1 Migration guide sketch (to be written as `docs/`, aimed at a consumer)

1. `uv add` nothing. Copy the modules you need from `generators` into your project
   (`ghmm/process.py` + `utils.py` is the minimal unit; add `factored/*.py` for composites), fix
   the module-level imports to your package paths, delete what you do not use.
2. Copy the transition-matrix constructors you need (`transition_matrices/classical.py`), or write
   your own — these are data.
3. Write one adapter module binding the vendored functions to the protocol's member names. ~50
   lines for a ghmm; the worked example ships in `simplexity`'s docs and its test fixture.
4. Write a factory in the same module for the config to target (this is where `encode`/`decode`
   assembly lives for composite processes).
5. Point your Hydra config at the factory; declare the section as a generative process.
6. Wire the `generators` conformance vectors into your test suite against your copy — **pending
   Q3**, since they do not exist yet.
7. Run. `vocab_size` / `bos_token` resolution and the training path work as before.

A guide that does not cover the actual consumers is not a guide — §7 is the list it must cover.

---

## 7 Consumer inventory

**In `simplexity` (all keep working; they are the regression proof):**

- 15 test modules under `tests/generative_processes/`, plus
  `tests/structured_configs/test_generative_process_config.py` (24 references — the densest
  coupling to the config machinery, and the file most affected by an added config field),
  `tests/run_management/test_components.py`, `tests/end_to_end/training.py`.
- 11 process configs in `tests/end_to_end/configs/generative_process/`.
- `simplexity` internals: `run_management/run_management.py`, `run_management/components.py`,
  `structured_configs/generative_process.py`.
- `walkthroughs/` — greps clean for `generative_processes`; no migration burden.

**Outside `simplexity` (`eden-experiments`) — four consumers, and the shape of their usage is the
good news:**

| Consumer | Imports | Migration cost |
| --- | --- | --- |
| `belief-state-recovery/ambitious/{harness,pipeline_smoke}.py` | `generative_processes.torch_generator.generate_data_batch` | low — a utility that takes a process, not a process. Works on vendored objects today (§3.2 line 4). |
| `fraxl-run/salvage-candidate/{train,analyze}.py` | same | low, same reason |
| `fraxl-run/salvage-candidate/{process.py, scratch/sweep*.py}` | `hidden_markov_model.HiddenMarkovModel`, `mixed_state_presentation.MixedStateTreeGenerator` | **medium** — the MSP gap (§4.3 item 1) |
| `nonergodic-demo/nonergodic_entropy.py` | `builder.*`, `mixed_state_presentation.*` | **medium** — same |

Three of four use `generate_data_batch`, which is process-agnostic and already vendored-compatible.
The two that would actually be blocked are blocked on MSP, which reinforces that §4.3 item 1 is the
critical-path gap rather than a footnote. Also note `r2-analysis/95_log_to_mlflow.py` uses
`managed_run` **without** a generative-process config, which is the case the current silent skip is
indistinguishable from.

**Not surveyed:** `origin` carries 60+ branches, several with process work in flight
(`adam/factor_tree`, `adam/hidden-factors`, `adam/producted-generator*`, `casper/*`). Any config
schema change collides with those on merge. This is a coordination cost, not a technical one, and
it is the main reason B3 (additive) is recommended over B2 (migrate every config).

---

## 8 Open questions

Ordered by how much they block.

**Q1 — Reading A or Reading B (§4)?** Does `simplexity` become a runner that consumes
consumer-owned processes, or does it vendor `generators` internally and stay the place processes
live? Everything else follows from this. My recommendation is A.

**Q2 — Who owns the four capability gaps (§4.3), especially the mixed-state tree?** Under Reading
A, `mixed_state_presentation.py` is *consumer-side analysis* that happens to live in the deprecated
package — it should move somewhere stable in `simplexity` (e.g. `simplexity/analysis/`) and operate
on the protocol. But it is 576 lines and two `eden-experiments` consumers import it directly. Move
it, leave it, or vendor it per-project?

**Q3 — Conformance vectors do not exist yet.** SPEC §7 specifies the encoding; no fixtures ship.
Under a structural contract, the vectors are the only thing keeping a consumer's modified copy
mathematically honest — the Protocol check cannot do it (§5.1). Is generating them in
`generators` a prerequisite for this deprecation, a parallel task, or somebody else's? If they are
not coming soon, the migration guide's safety story has a hole and should say so.

**Q4 — How teammate-visible should stage 2 be?** A `DeprecationWarning` on
`generative_processes` import fires in every teammate's run. Do you want that in this PR, in a
follow-up after you have told the team, or not at all (docs-only deprecation, matching the existing
`pyproject.toml` idiom)?

**Q5 — Does `transition_matrices.py` get deprecated too?** It is 16 process families, and it is
*data* rather than behavior — the most reusable, least contentious part of the package.
`generators` covers ~5 of them and its USAGE.md disclaims owning a process zoo while shipping
`transition_matrices/` anyway. Options: keep it in `simplexity` as data (my lean), upstream the
missing 11 into `generators`, or push each consumer to vendor what they use. This one has real
duplication consequences either way.

**Q6 — PR target: `dev` or `main`?** `CONTRIBUTING.md` defines a two-tier process and `dev` is
live (`origin/dev` at `5ca03a4`). Design-stage work with an evolving interface reads like `dev` to
me, but `main`'s bar is what the interface-stability language actually asks for. Also: should the
design doc land as its own PR ahead of the implementation, so the interface decision is reviewable
without the diff?

**Q7 — Does the design doc's location work?** This file is at `docs/design/generators_instantiation.md`.
The repo's `docs/` is currently flat with two files (`databricks_model_registry.md`,
`LOAD_SUBCONFIGS.md`) and no `design/` subdirectory, so I introduced one; say if you would rather
it be flat.

**Q8 — Naming.** The protocol's name and home are the most visible artifact of this change and
will outlive the deprecation. Reusing `GenerativeProcess` for the Protocol is clearest for
consumers but collides with the ABC of the same name in the package being deprecated. Options:
`GenerativeProcessProtocol` (explicit, ugly), `GenerativeProcess` in a new namespace (clean at the
call site, confusing during the transition), or something that names the capability rather than the
noun. Your call — this is a taste question with a long tail.
