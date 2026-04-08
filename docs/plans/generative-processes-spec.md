# Plan: Generative Processes Specification

## Context

The goal is an implementation-agnostic specification of the generative processes system, sufficient for a ground-up reimplementation in any language or framework. The spec defines **mathematical objects** and **the operations that must be possible** — not software architecture, class hierarchies, algorithms, or API shapes.

## What the spec defines

### 1. Process definitions and their inputs

Each process type has explicit authoritative inputs:

**Base GHMM:**
- Transition matrices T^(x) [V, S, S] in row-vector convention
- Optional: initial state η₀ [S] (defaults to stationary distribution if omitted)

Everything else is derived: net transition matrix T, normalizing eigenvector w, stationary distribution π. Transition matrices may be provided with principal eigenvalue ≠ 1; normalization by the principal eigenvalue is part of process construction, not a precondition on the input. An HMM is just a GHMM where w = **1** — no separate definition needed.

**Factored process:**
- Per-factor: transition matrices T_i [K_i, V_i, S_i, S_i], component type ("hmm" or "ghmm"), initial state η_i [S_i]
- Conditional dependency scheme (one of the four defined in §4) plus its parameters (e.g., control maps)
- Per-factor vocab sizes V_i (determines radix encoding)

**Nonergodic mixture:**
- Component processes (each a fully-defined generative process)
- Component weights [C] (normalized to sum to 1)
- Vocabulary maps: per-component mapping from local token indices to global token indices

**Vocabulary inflation:**
- A base generative process
- Inflation factor K ≥ 2

### 2. Operations that must be possible

Given a generative process (base GHMM or composite), it must be *possible* to:
- Compute the observation probability distribution from a belief state
- Sample an observation from a probability distribution
- Update a belief state given an observation (Bayesian filtering)
- Compute the probability of an observation sequence
- Compute the stationary distribution (when it exists and is unique)
- Generate sequences in batches from prescribed initial states

These are things you must be able to *do*, not methods a single object must *have*. An implementation could be a class with methods, a set of standalone functions, a pipeline, or anything else. The spec does not prescribe which of these operations are primitive vs. composed from others — only that each result is obtainable.

**On sequence generation:** Generated sequences must support BOS/EOS/PAD tokens with these semantics: generation always produces exactly `sequence_len` body tokens from the process. The output sequence has a separately specified `total_len` ≥ `sequence_len` + 2. The layout is: [BOS, body₁, ..., body_{sequence_len}, EOS, PAD, ..., PAD] padded to `total_len`. There is no early termination — the process always runs for `sequence_len` steps. PAD is only relevant when `total_len` > `sequence_len` + 2. This is a generation-level concern — it does not alter the process definition or its transition matrices. The spec defines the semantics but not the mechanism. Note: PAD behavior is new scope not present in the current implementation; BOS/EOS semantics are extracted from the existing generation utilities.

### 3. Composition schemes

Mathematical definitions of how to build new generative processes from existing ones. Each composite produces something for which all the operations above remain possible.

**Factored composition:**
- Multiple component GHMMs, each with its own state space and transition matrices
- Composite observations via radix token encoding (spec defines the encoding math)
- Conditional dependency between factors governs how the joint observation distribution is computed and how variant selection works for state transitions
- Stationary distribution: exists and computable when the coupling permits (e.g., independent factors → product of per-factor stationary distributions)

**Nonergodic mixture:**
- Multiple component processes with initial mixture weights
- State tracks component beliefs [C] plus per-component states
- Observation distribution is weighted sum over components
- Belief update via Bayes rule on per-component likelihoods. **Edge case:** when an observation has zero likelihood under every component, beliefs revert to the prior (no update) rather than producing NaN/division-by-zero.
- Vocabulary mapping (local ↔ global) to handle components with different alphabets
- Stationary distribution: does not exist as a unique distribution (each component has its own)
- Generation: sample one component, generate entire sequence from it

**Vocabulary inflation:**
- Wraps any process, multiplies vocab by factor K
- Token encoding: inflated = prefix × V_base + base
- P(inflated|state) = P(base|state) / K
- State dynamics depend only on base token
- Stationary distribution: same as base process (inflation doesn't affect dynamics)

### 4. Known conditional dependency schemes

Defined mathematically, not as a pluggable protocol. The spec describes the concept of "how factors can depend on each other" and gives four known schemes:

- **Independent:** joint = product of marginals, no inter-factor dependency
- **Sequential chain:** factor i depends on factor i-1's emitted token via a control map
- **Fully conditional:** each factor depends on ALL other factors' tokens via control maps over radix-encoded other-factor indices. **Important:** the joint distribution is computed as a *normalized product of conditionals* — this is an approximation, not a recovery of the true joint. The true joint under mutual conditioning is not available in closed form (each factor's distribution depends on all others' emissions, creating a circular dependency); the product-of-conditionals with normalization is a tractable approximation. When all conditional mass is zero, the distribution falls back to uniform. The spec must state both the approximation nature and the rationale explicitly.
- **Conditional transitions:** emissions independent or sequential; transitions mutually conditional (hybrid of the above)

For each: the math for computing the joint distribution, the math for selecting which transition matrix variant each factor uses given an observation, and the required parameters (control maps, vocab sizes, etc.).

**Relationship between K_i and control maps:** K_i (the number of transition matrix variants for factor i) must satisfy K_i ≥ max(control_map_i) + 1. For independent structure, K_i = 1 always. For sequential, K_i for factor i is determined by the range of the control map, which maps V_{i-1} parent-token values to variant indices. The spec should make this constraint explicit.

### 5. Conformance test vectors

Numerical input → expected output pairs for verifying correctness. Implementation-agnostic (just numbers, no framework). Derived from the existing test suite but distilled to be minimal and high-signal.

The conformance data needs a concrete encoding scheme for process definitions and heterogeneous state types:

- **Base GHMM:** transition matrices as nested arrays, optional initial state as array
- **Factored state:** ordered list of per-factor belief vectors (list of arrays)
- **Nonergodic state:** component beliefs (array) plus per-component states (list, where each element is an array or a factored state)
- **Process definitions:** JSON objects with a `type` field ("ghmm", "factored", "nonergodic", "inflated") and type-specific parameters matching the authoritative inputs defined in §1

## What the spec does NOT prescribe

- Class hierarchies, inheritance, abstract base classes, protocols, API shapes
- Whether operations are methods on an object, standalone functions, or something else
- Which operations are primitive vs. derived in the implementation
- HMM as a separate type (it's a GHMM with w = **1**)
- Internal representation choices (row/column vectors, log/linear space, log base)
- Algorithms (scan, eigendecomposition method, matrix product order)
- How BOS/EOS/PAD are implemented (matrix augmentation, wrapper logic, or otherwise)
- The transition matrix library (named processes like "mess3", "rrxor" — those are examples/test fixtures)
- Frozen factors (a batching convenience in the current implementation where certain factors in an independent factored process use a shared RNG to produce identical sequences across batch samples — not a mathematical property of the process)

## Deliverables

### 1. Specification document: `docs/generative-processes-spec.md`

**§1 Introduction and Notation**
- Purpose: enable ground-up reimplementation
- Scope: the mathematical definition of generative processes and their composition — not training, evaluation, or data pipeline concerns
- Notation conventions *used within the spec document*: row vectors for states, T^(x) shape [V, S, S]
- All indices are 0-based throughout the spec (observations, states, factors, variants)
- Input convention: transition matrices are provided in row-vector convention; implementations may transform internally as needed

**§2 Operations**
- The set of things that must be possible with a generative process
- Mathematical definition of each (what the result means, not how to compute it)
- Sequence generation semantics including BOS/EOS/PAD (generation-level concern, not process-definition concern)
- No prescribed API shape, no "primitive vs. derived" distinction

**§3 Generalized Hidden Markov Model**
- Authoritative inputs: transition matrices T^(x), optional initial state
- Validity constraints on raw input: all T^(x) entries must be finite; T = Σ_x T^(x) must have a positive real principal eigenvalue. For HMM-type matrices, entries must be non-negative and rows of T must sum to 1.
- Construction step: if principal eigenvalue λ₁ ≠ 1, normalize: T̃^(x) = T^(x) / λ₁. All subsequent formulas use the normalized matrices T̃^(x) and T̃ = Σ_x T̃^(x).
- Derived quantities from T̃: normalizing eigenvector w (right eigenvector of T̃ at eigenvalue 1), stationary distribution π (left eigenvector of T̃ at eigenvalue 1, normalized to sum to 1)
- Observation distribution: P(x|η) = (η T̃^(x) w) / (η·w)
- Belief update: η' = (η T̃^(x)) / (η T̃^(x) · w)
- Sequence probability: P(x₁:T) = (η₀ T̃^(x₁) ··· T̃^(xT) w) / (η₀·w)
- Note: when w = **1**, all formulas simplify to standard HMM (normalization by sum)
- **Zero-denominator case:** when η T̃^(x) · w = 0 (observation impossible given current belief), the belief update is undefined. This arises only from invalid use (e.g., observing a token that has zero probability) or numerical issues, not from valid generative process operation. The spec does not prescribe a specific behavior for this case.
- **Invalid input:** if T = Σ_x T^(x) has no positive real principal eigenvalue, the input is invalid and must be rejected

**§4 Composition: Factored Processes**
- Authoritative inputs: per-factor transition matrices [K_i, V_i, S_i, S_i], component types, initial states, conditional dependency scheme + parameters
- Composite token encoding (radix system: math for encode/decode)
- Joint observation distribution: depends on conditional dependency scheme
- Per-factor state update: decode composite obs → per-factor tokens, select transition matrix variant per scheme, update each factor
- Per-factor observation distribution (w=**1** case and general case)
- Stationary distribution for composites (when well-defined)

**§5 Conditional Dependency Schemes**
- General concept: how factors influence each other's observation distributions and transition variant selection
- §5.1 Independent
- §5.2 Sequential chain
- §5.3 Fully conditional — explicitly state: normalized product-of-conditionals approximation, uniform fallback when all mass is zero
- §5.4 Conditional transitions (hybrid)
- For each: joint distribution formula, variant selection formula, required parameters (control maps, etc.)
- Radix indexing math for other-factor encoding

**§6 Composition: Nonergodic Mixture**
- Authoritative inputs: component processes, weights, vocabulary maps
- State structure: component beliefs + per-component states
- Observation distribution as weighted sum
- Bayesian belief update
- Vocabulary mapping (local ↔ global)
- Sequence probability via marginalization
- Generation: sample one component

**§7 Composition: Vocabulary Inflation**
- Authoritative inputs: base process, inflation factor K
- Token encoding, distribution scaling, state dynamics

**§8 Stationary Distribution**
- For base GHMM: derived from T̃ (the spectrally normalized net transition matrix)
- For independent factored processes: product of per-factor stationary distributions
- For non-independent factored processes: may or may not exist depending on coupling; the spec does not define a general method (implementations may compute it if desired)
- For nonergodic: no unique stationary distribution (each component has its own)
- For inflated: same as base

**§9 Conformance Test Vectors**
- Encoding scheme for process definitions (JSON objects with type discriminator and type-specific parameters)
- Encoding scheme for state types (arrays for GHMM, list-of-arrays for factored, structured object for nonergodic)
- **Tolerance:** all numerical comparisons use relative tolerance (e.g., 1e-5) to accommodate floating-point differences across implementations and platforms (JAX, NumPy, Julia, Rust, etc.). Each test vector specifies its tolerance.
- **Generation tests:** conformance vectors only cover deterministic operations (observation distributions, belief updates, sequence probabilities, stationary distributions). Sequence generation is behaviorally specified (BOS/EOS/PAD layout, token positions) but not numerically tested — sampling depends on RNG implementation, which defeats implementation-agnosticism.
- Test categories:
  - GHMM belief update and observation distribution
  - GHMM sequence probability
  - GHMM with w=**1** (HMM case)
  - GHMM with non-trivial w (e.g., Fanizza process)
  - Factored process: token encoding roundtrip, joint distributions
  - Conditional dependency: variant selection, joint distribution correctness, fully-conditional approximation behavior
  - Nonergodic: mixture observation distribution, Bayesian belief update
  - Vocabulary inflation: distribution scaling, probability scaling
  - Stationary distribution computation (including default initial state = stationary distribution)
  - Nonergodic zero-likelihood fallback (beliefs revert to prior)
  - Sequence generation with BOS/EOS/PAD

**§10 Glossary**

**Note:** Matrix transformations (e.g., noisy channel blur) are not part of the normative spec. They are preprocessing steps a user may apply to transition matrices before defining a process. The spec defines what "valid" means for input matrices (see §3) but not how they were produced.

**Note on log-space:** The current GHMM implementation has log-space helper methods that are explicitly marked as incorrect (TODO). The spec defines linear-space semantics as authoritative. Implementations may use log-space internally for numerical stability, but the spec does not define log-space formulas as normative.

### 2. Conformance test data: `docs/conformance-tests.json`

Standalone data file referenced by §9. No framework dependencies. Derived from existing tests at `tests/generative_processes/`, selecting the most high-signal numerical checks. Uses the encoding scheme defined in §9 for process definitions and state representations.

## Critical source files

To extract mathematical definitions from:
- `simplexity/generative_processes/generative_process.py` — operations list
- `simplexity/generative_processes/generator.py` — BOS/EOS generation semantics
- `simplexity/generative_processes/generalized_hidden_markov_model.py` — GHMM math
- `simplexity/generative_processes/hidden_markov_model.py` — w=**1** simplifications
- `simplexity/generative_processes/factored_generative_process.py` — factored composition
- `simplexity/generative_processes/nonergodic_generative_process.py` — mixture composition
- `simplexity/generative_processes/inflated_vocabulary_process.py` — vocab inflation
- `simplexity/generative_processes/transition_matrices.py` — stationary distribution computation
- `simplexity/generative_processes/structures/*.py` — conditional dependency scheme math
- `simplexity/utils/factoring_utils.py` — per-factor kernels, TokenEncoder radix math

To derive conformance tests from:
- `tests/generative_processes/test_hidden_markov_model.py`
- `tests/generative_processes/test_generalized_hidden_markov_model.py`
- `tests/generative_processes/test_factored_generative_process.py`
- `tests/generative_processes/test_factored_structures.py`
- `tests/generative_processes/test_nonergodic_generative_process.py`
- `tests/generative_processes/test_inflated_vocabulary_process.py`
- `tests/generative_processes/test_generator.py` — BOS/EOS generation behavior
- Note: PAD and total_len conformance vectors will be newly authored (not present in current tests)

## Verification

1. **Completeness:** every operation in §2 has corresponding math somewhere in the spec; every process type has its authoritative inputs defined
2. **Accuracy:** all formulas verified against source code (done during planning phase)
3. **Implementation-agnosticism:** no JAX/Equinox/Python-specific terms; no prescribed API shape, algorithms, or internal representation
4. **Conformance tests:** run existing test suite to verify current tests pass, then verify test vectors reproduce the same expected values
5. **Non-prescriptiveness:** review spec for anything that constrains implementation approach — remove it
