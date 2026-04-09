# Generative Processes Specification

**Version:** 1.0  
**Status:** Draft

## 1 Introduction and Notation

### 1.1 Purpose

This document specifies the mathematical objects and observable behaviors that
constitute a *generative process* system. The specification is sufficient for a
ground-up reimplementation in any language or framework. It defines what results
must be obtainable from a generative process — not how to compute them.

### 1.2 Scope

The spec covers:

- The definition and construction of generalized hidden Markov models (GHMMs)
- Operations that must be possible on any generative process
- Composition schemes: factored processes, nonergodic mixtures, vocabulary
  inflation
- Conditional dependency schemes for factored processes
- Conformance test vectors for verifying correctness

The spec does **not** cover:

- Training, evaluation, or data pipeline concerns
- Software architecture, programming language, or framework choices
- Algorithms or internal representations
- Implementation-specific conveniences or optimizations

### 1.3 Notation

The following conventions are used within this document. They are not
prescriptions for implementations.

| Symbol | Meaning |
|--------|---------|
| V | Vocabulary size (number of distinct observations) |
| S | State space size (number of hidden states) |
| T^(x) | Observation-conditional transition matrix for observation x |
| T | Net transition matrix: T = sum_x T^(x) |
| T~^(x) | Spectrally normalized transition matrix (see S3.2) |
| T~ | Spectrally normalized net transition matrix |
| w | Normalizing eigenvector (right eigenvector of T~ at eigenvalue 1) |
| pi | Stationary distribution (left eigenvector of T~ at eigenvalue 1, normalized to sum to 1) |
| eta | Belief state (row vector of dimension S) |
| eta_0 | Initial belief state |
| lambda_1 | Principal (largest real) eigenvalue of T |
| K | Number of transition matrix variants (factored processes) or inflation factor |
| F | Number of factors in a factored process |
| C | Number of components in a nonergodic mixture |

**Conventions:**

- All indices are **0-based** throughout the spec: observations in {0, ...,
  V-1}, states in {0, ..., S-1}, factors in {0, ..., F-1}, variants in
  {0, ..., K-1}.
- Belief states are **row vectors**. State transitions act by right-multiplying
  the belief state by a transition matrix: eta' = eta T^(x).
- Transition matrices T^(x) have shape [V, S, S], where T^(x)[v] is the S x S
  matrix applied when observation v is emitted.
- Input transition matrices are provided in **row-vector convention**: the
  (i,j) entry of T^(x)[v] is the unnormalized weight for transitioning from
  state i to state j while emitting observation v. Implementations may
  transform internally as needed.

## 2 Operations

Given a generative process (base GHMM or composite), it must be *possible* to
obtain the following results. The spec defines what results are required, not
how they are organized or computed.

### 2.1 Observation probability distribution

Given a belief state eta, compute the probability distribution over
observations:

    P(x | eta) for all x in {0, ..., V-1}

The result is a vector of dimension V that sums to 1.

### 2.2 Observation sampling

Given a probability distribution over observations (a vector of dimension V
summing to 1), sample an observation. This is standard categorical sampling and
is not specific to generative processes.

### 2.3 Belief state update

Given a belief state eta and an observed token x, compute the posterior belief
state eta'. This is Bayesian filtering: the belief state after conditioning on
the observation.

### 2.4 Sequence probability

Given an observation sequence x_1, ..., x_T, compute the probability
P(x_1, ..., x_T) under the process's initial state.

### 2.5 Stationary distribution

When it exists and is unique, compute the stationary distribution pi of the
process. See S8 for which process types have well-defined stationary
distributions.

### 2.6 Sequence generation

Generate sequences in batches from prescribed initial states. Generated
sequences must support BOS (beginning-of-sequence), EOS (end-of-sequence), and
PAD (padding) tokens with the following semantics:

- Generation always produces exactly `sequence_len` body tokens from the
  process.
- The output sequence has a separately specified `total_len` >=
  `sequence_len` + 2.
- The layout is: [BOS, body_1, ..., body_{sequence_len}, EOS, PAD, ..., PAD]
  padded to `total_len`.
- There is no early termination. The process always runs for `sequence_len`
  steps.
- PAD is only relevant when `total_len` > `sequence_len` + 2.

This is a generation-level concern. It does not alter the process definition or
its transition matrices. The spec defines the semantics but not the mechanism.

BOS, EOS, and PAD token indices must be outside the process's vocabulary range
{0, ..., V-1}. They are framing tokens, not process observations.

## 3 Generalized Hidden Markov Model

### 3.1 Authoritative inputs

A GHMM is defined by:

- **Transition matrices** T^(x) of shape [V, S, S] in row-vector convention
- **Initial state** eta_0 of dimension S (optional; defaults to the stationary
  distribution pi when omitted)

These are the only inputs. Everything else is derived.

### 3.2 Validity and construction

**Validity constraints on raw input:**

1. All entries of T^(x) must be finite.
2. The net transition matrix T = sum_x T^(x) must have a positive real
   principal eigenvalue lambda_1.
3. If the input is intended as an HMM (standard hidden Markov model): all
   entries must be non-negative, and for each state s, sum over all
   observations and next-states must equal 1:
   sum_{x,s'} T^(x)[s, s'] = 1 for all s.

If T has no positive real principal eigenvalue, the input is invalid.

**Construction (spectral normalization):**

If lambda_1 != 1, normalize all transition matrices:

    T~^(x) = T^(x) / lambda_1

After normalization, T~ = sum_x T~^(x) has principal eigenvalue 1. All
subsequent formulas in this spec use the normalized matrices T~^(x) and T~.

If lambda_1 = 1, then T~^(x) = T^(x) (no normalization needed).

### 3.3 Derived quantities

From T~ (the spectrally normalized net transition matrix):

- **Normalizing eigenvector w**: the right eigenvector of T~ at eigenvalue 1,
  scaled so that sum(w) = S (the number of states). That is, T~ w = w and
  sum(w) = S.
- **Stationary distribution pi**: the left eigenvector of T~ at eigenvalue 1,
  normalized so that sum(pi) = 1. That is, pi T~ = pi and sum(pi) = 1.

**HMM as a special case:** When T^(x) satisfies the HMM constraints (3.2,
item 3), the normalizing eigenvector is the all-ones vector: w = **1** (a
vector of S ones). In this case, all formulas below simplify to standard HMM
formulas where normalization is by the L1 norm (sum of entries).

### 3.4 Observation probability distribution

Given a belief state eta:

    P(x | eta) = (eta T~^(x) w) / (eta . w)

where eta T~^(x) is the row vector resulting from right-multiplying eta by the
S x S matrix T~^(x), then dotted with w.

When w = **1**, this simplifies to:

    P(x | eta) = sum(eta T~^(x)) / sum(eta)

### 3.5 Belief state update

Given a belief state eta and an observed token x:

    eta' = (eta T~^(x)) / (eta T~^(x) . w)

The numerator is the row vector eta T~^(x). The denominator is the scalar dot
product of that vector with w.

When w = **1**, this simplifies to:

    eta' = (eta T~^(x)) / sum(eta T~^(x))

**Zero-denominator case:** When eta T~^(x) . w = 0 (the observation is
impossible given the current belief), the update is undefined. This arises
only from invalid use (observing a token with zero probability) or numerical
issues, not from valid generative process operation. The spec does not
prescribe a specific behavior for this case.

*Note on edge case philosophy:* The zero-denominator case in a base GHMM
represents a logical error (conditioning on an impossible event), which is
why no fallback is prescribed. By contrast, the nonergodic mixture (S6.4)
and fully conditional (S5.3) prescribe specific fallback behaviors because
zero-mass situations can arise naturally in those contexts — a mixture
component may not cover all tokens, and the product-of-conditionals
approximation may produce degenerate results.

### 3.6 Sequence probability

Given an observation sequence x_1, ..., x_T and an initial state eta_0:

    P(x_1, ..., x_T) = (eta_0 T~^(x_1) T~^(x_2) ... T~^(x_T) w) / (eta_0 . w)

The numerator is the scalar obtained by left-multiplying eta_0 through the
sequence of matrices and then dotting with w. The denominator is the scalar
eta_0 . w.

When w = **1**:

    P(x_1, ..., x_T) = sum(eta_0 T~^(x_1) ... T~^(x_T)) / sum(eta_0)

### 3.7 Normalizing constant

The value eta_0 . w appears as the denominator in S3.4, S3.5, and S3.6. This
value depends on the initial state and the normalizing eigenvector. It is not
guaranteed to equal 1 under the scaling convention sum(w) = S; in general,
pi . w != 1. Implementations must track this denominator explicitly.

## 4 Composition: Factored Processes

A factored process composes F component processes (each a GHMM or HMM) into a
single generative process over a composite observation space.

### 4.1 Authoritative inputs

- For each factor i in {0, ..., F-1}:
  - **Transition matrices** T_i of shape [K_i, V_i, S_i, S_i], where K_i is
    the number of variants, V_i is the factor's vocabulary size, and S_i is
    the factor's state space size
  - **Component type**: "hmm" or "ghmm", determining which normalization
    formulas to use
  - **Initial state** eta_i of dimension S_i

For each factor, the normalizing eigenvector w_i is **derived**, not an
authoritative input. For each variant k of factor i, w_i[k] is the right
eigenvector of the per-variant net transition matrix sum_x T_i[k, x] at
eigenvalue 1, following the same construction as S3.2-S3.3. For HMM-type
factors, w_i[k] = **1** for all k regardless of the variant — this is a
direct consequence of the HMM validity constraint (S3.2, item 3), not an
independent definition.
- **Conditional dependency scheme** (one of the four defined in S5) plus its
  parameters (control maps, etc.)
- **Per-factor vocabulary sizes** V_0, ..., V_{F-1}

The composite vocabulary size is V_composite = V_0 * V_1 * ... * V_{F-1}.

### 4.2 Composite token encoding

Composite observations are encoded as a single integer in {0, ...,
V_composite - 1} using a mixed-radix (big-endian) scheme.

**Encoding** (factor tokens to composite token):

Given per-factor tokens (t_0, t_1, ..., t_{F-1}) where t_i in {0, ...,
V_i - 1}:

    composite = sum_{i=0}^{F-1} t_i * m_i

where the radix multipliers are:

    m_i = product_{j=i+1}^{F-1} V_j

That is, m_0 = V_1 * V_2 * ... * V_{F-1}, m_1 = V_2 * ... * V_{F-1}, ...,
m_{F-1} = 1.

**Decoding** (composite token to factor tokens):

Given a composite token c:

    t_i = floor(c / m_i) mod V_i

**Invariant:** Encoding then decoding (and vice versa) is the identity:

    decode(encode(t_0, ..., t_{F-1})) = (t_0, ..., t_{F-1})

### 4.3 Per-factor observation distribution

For a single factor i with state eta_i, transition matrix T_i[k] (for variant
k), and normalizing eigenvector w_i[k]:

**GHMM-type factor:**

    P_i(x | eta_i, k) = (eta_i T_i[k, x] w_i[k]) / (eta_i . w_i[k])

**HMM-type factor** (w_i[k] = **1**):

    P_i(x | eta_i, k) = sum(eta_i T_i[k, x])

The variant index k is selected by the conditional dependency scheme (S5).

### 4.4 Joint observation distribution

The joint distribution over composite tokens depends on the conditional
dependency scheme. Each scheme (S5) defines how per-factor distributions are
combined into a joint distribution of dimension V_composite.

### 4.5 Per-factor state update

Given a composite observation c:

1. Decode to per-factor tokens: (t_0, ..., t_{F-1}) = decode(c)
2. Select per-factor variants (k_0, ..., k_{F-1}) according to the
   conditional dependency scheme
3. Update each factor independently:

**GHMM-type factor:**

    eta_i' = (eta_i T_i[k_i, t_i]) / (eta_i T_i[k_i, t_i] . w_i[k_i])

**HMM-type factor:**

    eta_i' = (eta_i T_i[k_i, t_i]) / sum(eta_i T_i[k_i, t_i])

### 4.6 Sequence probability

Computed by iterating the joint observation distribution and state updates
through the observation sequence. At each step, the joint observation
distribution (S4.4) determines the probability of the composite token, and
the per-factor state updates (S4.5) advance the state. The sequence
probability is the product of per-step joint probabilities. This must go
through the joint distribution (not per-factor probabilities independently),
because factors may be conditionally dependent.

## 5 Conditional Dependency Schemes

A conditional dependency scheme determines:

1. How the joint observation distribution is computed from per-factor
   distributions
2. How per-factor transition matrix variants are selected given an observation

The spec defines four schemes as mathematical definitions.

### 5.1 Independent

**Joint distribution:**

    P(t_0, ..., t_{F-1}) = product_{i=0}^{F-1} P_i(t_i | eta_i, 0)

All factors use variant 0. The joint is the product of marginals.

**Variant selection:**

    k_i = 0 for all i

**Required parameters:** None.

### 5.2 Sequential chain

Factor i depends on factor (i-1)'s emitted token via a control map.

**Joint distribution:**

    P(t_0, ..., t_{F-1}) = P_0(t_0 | eta_0, 0) * product_{i=1}^{F-1} P_i(t_i | eta_i, control_maps[i][t_{i-1}])

Factor 0 always uses variant 0. Each subsequent factor's variant is determined
by the preceding factor's token, mapped through a control map.

**Variant selection:**

    k_0 = 0
    k_i = control_maps[i][t_{i-1}]   for i >= 1

**Required parameters:**

- `control_maps`: a tuple of F entries, where `control_maps[0]` is unused
  (factor 0 has no parent) and `control_maps[i]` for i >= 1 is an array of
  dimension V_{i-1} mapping parent tokens to variant indices.

**Constraint:** K_i >= max(control_maps[i]) + 1 for all i >= 1.

### 5.3 Fully conditional

Each factor depends on all other factors' tokens. The joint distribution is
computed as a **normalized product of conditionals** — this is a tractable
approximation, not a recovery of the true joint.

**Rationale:** Under mutual conditioning, each factor's distribution depends on
all other factors' emissions, creating a circular dependency. The true joint is
not available in closed form. The product-of-conditionals with normalization is
a tractable approximation.

**Joint distribution:**

For each factor i, compute the conditional distribution P_i(t_i | others)
where "others" means the tokens of all factors except i, encoded as a
radix index (see S5.3.1):

    P_unnorm(t_0, ..., t_{F-1}) = product_{i=0}^{F-1} P_i(t_i | eta_i, control_maps[i][radix_other(i, t)])

    P(t_0, ..., t_{F-1}) = P_unnorm(t_0, ..., t_{F-1}) / Z

where Z = sum over all (t_0, ..., t_{F-1}) of P_unnorm.

**Zero-mass fallback:** If Z = 0 (all conditional products are zero), the
joint distribution falls back to uniform:
P(t_0, ..., t_{F-1}) = 1 / V_composite.

*Note (non-normative):* Implementations may compute the product and
normalization in log-space to avoid underflow. In that case, Z = 0
corresponds to logsumexp(log P_unnorm) being negative infinity.

**Variant selection:**

    k_i = control_maps[i][radix_other(i, (t_0, ..., t_{F-1}))]

**Required parameters:**

- `control_maps`: a tuple of F arrays, where `control_maps[i]` has dimension
  equal to the product of all other factors' vocab sizes:
  product_{j != i} V_j.

**Constraint:** K_i >= max(control_maps[i]) + 1 for all i.

#### 5.3.1 Other-factor radix indexing

For factor i, the "other-factor index" encodes the tokens of all factors
except i into a single integer. Define multipliers:

    mu_i[j] = product_{k=j+1, k!=i}^{F-1} V_k    for j != i
    mu_i[i] = 0

Then:

    radix_other(i, (t_0, ..., t_{F-1})) = sum_{j=0, j!=i}^{F-1} t_j * mu_i[j]

This maps the (F-1)-tuple of other-factor tokens to an integer in
{0, ..., (product_{j!=i} V_j) - 1}.

### 5.4 Conditional transitions

A hybrid scheme: emissions may be independent or follow a sequential chain,
while transitions are mutually conditional (as in the fully conditional
scheme). The key property of this scheme is that each factor may use
**different variant indices** for computing its emission distribution vs.
selecting its transition matrix. The emission variant determines which
T_i[k_emit] contributes to the joint observation distribution; the
transition variant determines which T_i[k_trans] is used for the state
update after the observation.

**Joint distribution:**

Depends on the emission mode:

*Independent emissions:*

    P(t_0, ..., t_{F-1}) = product_{i=0}^{F-1} P_i(t_i | eta_i, emission_variant_indices[i])

*Sequential emissions:*

    P(t_0, ..., t_{F-1}) = P_0(t_0 | eta_0, emission_variant_indices[0]) *
        product_{i=1}^{F-1} P_i(t_i | eta_i, k_emit_i)

where k_emit_i is determined by emission_control_maps[i] applied to the
prefix tokens (t_0, ..., t_{i-1}) encoded via prefix radix multipliers.

**Variant selection (for transitions):**

Transitions always use the fully-conditional scheme:

    k_trans_i = control_maps_transition[i][radix_other(i, (t_0, ..., t_{F-1}))]

The transition variant k_trans_i selects which T_i[k_trans_i] is used for the
state update. The emission variant (used for computing the observation
distribution) may differ from the transition variant.

**Required parameters:**

- `control_maps_transition`: a tuple of F arrays, where
  `control_maps_transition[i]` has dimension product_{j!=i} V_j
- `emission_variant_indices`: an array of F indices, specifying the default
  emission variant for each factor
- `emission_control_maps` (optional): a tuple of F entries for sequential
  emission mode. `emission_control_maps[0]` is unused;
  `emission_control_maps[i]` for i >= 1 maps prefix tokens to emission
  variants. When absent, emissions use fixed `emission_variant_indices`.
- `vocab_sizes`: array of F vocabulary sizes

The emission mode is determined by the presence of `emission_control_maps`:
if any entry (for i >= 1) is non-null, emissions follow a sequential chain;
otherwise, emissions are independent.

**Constraint:** K_i >= max(control_maps_transition[i]) + 1 for all i.

#### 5.4.1 Prefix radix indexing

For sequential emissions, the prefix of factor i encodes tokens
(t_0, ..., t_{i-1}) as:

    prefix_index = sum_{j=0}^{i-2} t_j * product_{l=j+1}^{i-1} V_l

    (plus t_{i-1} * 1 for the last term)

This maps the prefix tokens to an integer in
{0, ..., (product_{j=0}^{i-1} V_j) - 1}.

### 5.5 Relationship between K_i and control maps

For all schemes, the number of transition matrix variants K_i for factor i must
satisfy:

- **Independent:** K_i = 1 (only one variant needed)
- **Sequential:** K_i >= max(control_maps[i]) + 1 for i >= 1; K_0 = 1
- **Fully conditional:** K_i >= max(control_maps[i]) + 1 for all i
- **Conditional transitions:** K_i >= max(control_maps_transition[i]) + 1 for
  all i (transition variants); emission variants must also be in range

## 6 Composition: Nonergodic Mixture

A nonergodic mixture combines C component processes (each a fully-defined
generative process) into a single generative process with Bayesian component
tracking.

### 6.1 Authoritative inputs

- **Component processes**: C generative processes, each independently defined
  (may be base GHMMs, factored processes, or other composites)
- **Component weights**: an array of dimension C, normalized to sum to 1
- **Vocabulary maps**: for each component i, an array `vocab_maps[i]` of
  dimension V_i^{local} mapping local token indices to global token indices.
  The global vocabulary size is max(max(vocab_maps[i]) for all i) + 1.

No duplicate global indices are allowed within any single vocabulary map.

### 6.2 State structure

The state of a nonergodic mixture consists of:

- **Component beliefs**: an array of dimension C, representing P(component i |
  observations so far). Sums to 1.
- **Component states**: a tuple of C states, one per component process. Each
  component state has the structure appropriate to that component's type.

### 6.3 Observation probability distribution

    P(x_global | state) = sum_{i=0}^{C-1} beliefs[i] * P_i(inv_map_i(x_global))

where inv_map_i maps a global token index to the corresponding local token
index for component i. If x_global is not in component i's vocabulary (i.e.,
inv_map_i(x_global) is undefined), component i's contribution is 0.

### 6.4 Belief state update

Given a global observation x_global, update the component beliefs via Bayes
rule:

    likelihood[i] = P_i(inv_map_i(x_global))

where P_i is the observation probability under component i's current state.
If x_global is unmapped for component i, likelihood[i] = 0.

    beliefs'[i] = (beliefs[i] * likelihood[i]) / sum_j(beliefs[j] * likelihood[j])

Each component's internal state is also updated conditioned on the
observation, but only for components with positive likelihood. Specifically:

- If likelihood[i] > 0, component i's state is updated using the local token
  index: state_i' = update(state_i, inv_map_i(x_global)).
- If likelihood[i] = 0 (including when x_global is unmapped for component i),
  component i's state is unchanged: state_i' = state_i. The update is
  skipped entirely for that component — not computed and discarded, but
  never performed.

**Zero-likelihood fallback:** When sum_j(beliefs[j] * likelihood[j]) = 0
(the observation has zero likelihood under every component), both the beliefs
and all component states revert to their prior values. This prevents
division-by-zero.

### 6.5 Sequence probability

    P(x_1, ..., x_T) = sum_{i=0}^{C-1} weights[i] * P_i(mapped_seq_i)

where mapped_seq_i is the observation sequence with each global token replaced
by its local equivalent under component i. If any token in the sequence is
unmapped for component i, that component's contribution to the sum is 0.

### 6.6 Generation

Generation from a nonergodic mixture:

1. Sample one component index according to the component beliefs.
2. Generate the entire sequence from that component's process.
3. Map each emitted local token to its global equivalent via the component's
   vocabulary map.

Once a component is sampled, the entire sequence comes from that component.
There is no switching between components mid-sequence.

## 7 Composition: Vocabulary Inflation

Vocabulary inflation wraps any generative process and multiplies its
observation space by a factor K, adding K-1 "noise aliases" for each base
token.

### 7.1 Authoritative inputs

- **Base process**: any fully-defined generative process with vocabulary size
  V_base
- **Inflation factor** K >= 2

The inflated vocabulary size is V_inflated = K * V_base.

### 7.2 Token encoding

    inflated_token = prefix * V_base + base_token

where prefix in {0, ..., K-1} and base_token in {0, ..., V_base - 1}.

    base_token = inflated_token mod V_base
    prefix = floor(inflated_token / V_base)

### 7.3 Observation probability distribution

    P(inflated_token | state) = P_base(base_token | state) / K

The base distribution is tiled K times, each copy scaled by 1/K. All K
noise aliases of a given base token have equal probability.

### 7.4 Belief state update

State dynamics depend only on the base token:

    state' = update(state, inflated_token mod V_base)

The noise prefix is discarded.

### 7.5 Sequence probability

    P(inflated_seq) = P_base(base_seq) * (1/K)^T

where base_seq extracts the base token from each inflated token and T is the
sequence length. Per-token loss increases by exactly log(K) (in any log base).

### 7.6 Emission

To emit an inflated observation:

1. Emit base_token from the base process.
2. Sample prefix uniformly from {0, ..., K-1}.
3. Return prefix * V_base + base_token.

## 8 Stationary Distribution

The stationary distribution pi of a generative process, when it exists, is the
belief state that is invariant under the expected state update. It is typically
used as the default initial state.

### 8.1 Base GHMM

pi is the left eigenvector of T~ (the spectrally normalized net transition
matrix) at eigenvalue 1, normalized to sum to 1:

    pi T~ = pi,    sum(pi) = 1

This always exists for a valid GHMM (by construction, T~ has eigenvalue 1).

### 8.2 Independent factored processes

For independently composed factors, the composite stationary distribution is
the tuple of per-factor stationary distributions:

    pi_composite = (pi_0, pi_1, ..., pi_{F-1})

where each pi_i is the stationary distribution of factor i considered in
isolation (using variant 0).

### 8.3 Non-independent factored processes

For factored processes with non-trivial conditional dependencies (sequential,
fully conditional, or conditional transitions), the stationary distribution
may or may not exist depending on the specific coupling. The spec does not
define a general method for computing it. No known implementation currently
computes it. Implementations may do so for specific cases if needed.

### 8.4 Nonergodic mixtures

A nonergodic mixture does not have a unique stationary distribution. Each
component has its own stationary distribution, but there is no single
distribution that is invariant for the mixture as a whole. The initial state
of a nonergodic mixture is the component weights plus the per-component
initial states.

### 8.5 Vocabulary inflation

The stationary distribution of an inflated process is the same as the base
process's stationary distribution. Inflation affects only the observation space,
not the state dynamics.

## 9 Conformance Test Vectors

### 9.1 Encoding scheme

#### 9.1.1 Process definitions

Process definitions are encoded as JSON objects with a `type` discriminator:

**Base GHMM:**

```json
{
  "type": "ghmm",
  "transition_matrices": [[[...]]]
}
```

`transition_matrices` is a 3D array of shape [V, S, S]. An optional
`initial_state` field (1D array of dimension S) overrides the default
stationary distribution.

**HMM** (a GHMM with w = **1**):

```json
{
  "type": "hmm",
  "transition_matrices": [[[...]]]
}
```

The `hmm` type tag is a convenience for conformance testing, indicating that
the additional validity constraints of S3.2 item 3 apply (non-negative
entries, rows of T sum to 1). Mathematically, an HMM is a GHMM where
w = **1**; the separate type tag signals which validation rules to check,
not a distinct mathematical object.

**Factored process:**

```json
{
  "type": "factored",
  "factors": [
    {
      "component_type": "hmm",
      "transition_matrices": [[[[...]]]],
      "initial_state": [...],
      "vocab_size": 2
    }
  ],
  "structure": {
    "type": "independent"
  }
}
```

`transition_matrices` is a 4D array of shape [K, V, S, S]. The `structure`
object specifies the conditional dependency scheme with its parameters.

**Nonergodic mixture:**

```json
{
  "type": "nonergodic",
  "components": [...],
  "weights": [...],
  "vocab_maps": [[...], ...]
}
```

**Vocabulary inflation:**

```json
{
  "type": "inflated",
  "base_process": {...},
  "inflation_factor": 3
}
```

#### 9.1.2 State encoding

- **GHMM state:** a 1D array of dimension S
- **Factored state:** an ordered list of 1D arrays, one per factor
- **Nonergodic state:** an object with `component_beliefs` (1D array of
  dimension C) and `component_states` (list of state encodings, one per
  component)

#### 9.1.3 Structure definitions

```json
{"type": "independent"}

{"type": "sequential", "control_maps": [null, [0, 1, ...]]}

{"type": "fully_conditional", "control_maps": [[...], [...]]}

{"type": "conditional_transitions",
 "control_maps_transition": [[...], [...]],
 "emission_variant_indices": [0, 0],
 "emission_control_maps": [null, [1, 0]]}
```

For `conditional_transitions`, the emission mode is derived from the
`emission_control_maps` field: if present and any entry for i >= 1 is
non-null, emissions follow a sequential chain; if absent, emissions use
fixed `emission_variant_indices`.

**Initial state convention:** When a base process definition in a test vector
omits `initial_state`, the default is the stationary distribution (S3.1).
Tests in the `ghmm_stationary_distribution` category verify this computation
independently and should be validated first, since other test vectors
(particularly sequence probability) depend on correct stationary distribution
computation.

### 9.2 Tolerance

All numerical comparisons use relative tolerance to accommodate floating-point
differences across implementations and platforms. Each test vector specifies
its tolerance. A default of 1e-6 relative tolerance is used unless otherwise
noted.

### 9.3 Generation tests

Conformance vectors cover only deterministic operations. Sequence generation is
behaviorally specified (BOS/EOS/PAD layout, token positions) but not
numerically tested, because sampling depends on RNG implementation.

### 9.4 Test vectors

Test vectors are provided in the companion file `conformance-tests.json`.
Each test vector has these required fields:

- `id`: unique identifier
- `category`: which aspect is being tested
- `description`: human-readable description
- `operation`: which operation is being tested
- `input`: operation-specific input (state, observation sequence, etc.)
- `expected`: expected output (numerical value, array, or object)

And these optional fields:

- `process`: process definition (S9.1.1) — omitted for operation-only tests
  like token encoding
- `tolerance`: relative tolerance for numerical comparison (defaults to the
  file-level `default_tolerance` when omitted)
- `notes`: human-readable explanation of the expected value

Categories covered:

1. **ghmm_observation_distribution** — Observation probability distribution
   from a GHMM belief state
2. **ghmm_belief_update** — Belief state update after observing a token
3. **ghmm_sequence_probability** — Probability of an observation sequence
4. **ghmm_hmm_case** — GHMM with w = **1** (HMM case)
5. **ghmm_nontrivial_w** — GHMM with non-trivial normalizing eigenvector
6. **ghmm_stationary_distribution** — Stationary distribution computation
7. **factored_token_encoding** — Radix token encoding/decoding roundtrip
8. **factored_independent** — Independent factored joint distribution
9. **factored_sequential** — Sequential chain joint distribution
10. **factored_fully_conditional** — Fully conditional joint distribution
11. **factored_conditional_transitions** — Conditional transitions joint
    distribution
12. **fully_conditional_zero_fallback** — Uniform fallback when all mass is
    zero
13. **nonergodic_observation_distribution** — Mixture observation distribution
14. **nonergodic_belief_update** — Bayesian component belief update
15. **nonergodic_sequence_probability** — Mixture sequence probability
16. **nonergodic_zero_likelihood** — Zero-likelihood fallback (beliefs revert
    to prior)
17. **nonergodic_vocab_mapping** — Different vocabulary maps across components
18. **inflation_distribution** — Inflated distribution scaling
19. **inflation_probability** — Inflated sequence probability scaling
20. **generation_layout** — BOS/EOS/PAD token layout verification

## 10 Glossary

| Term | Definition |
|------|-----------|
| Belief state | A row vector representing the current probability distribution over hidden states, conditioned on observations so far |
| BOS | Beginning-of-sequence token; a framing token prepended to generated sequences |
| Component | One of the constituent processes in a nonergodic mixture |
| Composite token | A single integer encoding the joint observation of all factors in a factored process |
| Conditional dependency scheme | The rule governing how factors in a factored process influence each other's observation distributions and transition variant selection |
| Control map | An array mapping token indices (or radix-encoded token tuples) to transition matrix variant indices |
| EOS | End-of-sequence token; a framing token appended after the last body token |
| Factor | One of the constituent GHMMs in a factored process |
| GHMM | Generalized Hidden Markov Model; the fundamental process type in this spec |
| HMM | Hidden Markov Model; a GHMM where the normalizing eigenvector is the all-ones vector |
| Inflation factor | The multiplier K by which vocabulary inflation expands the observation space |
| Net transition matrix | T = sum_x T^(x); the sum over all observation-conditional matrices |
| Normalizing eigenvector | The right eigenvector w of T~ at eigenvalue 1, used for belief state normalization |
| Observation | A discrete token emitted by the process at each time step |
| PAD | Padding token; fills remaining positions when total_len > sequence_len + 2 |
| Principal eigenvalue | The largest real eigenvalue of the net transition matrix T |
| Radix encoding | Mixed-radix positional encoding of per-factor tokens into a composite token |
| Spectral normalization | Dividing all transition matrices by the principal eigenvalue to ensure T~ has eigenvalue 1 |
| Stationary distribution | The left eigenvector pi of T~ at eigenvalue 1, normalized to sum to 1; the invariant belief state |
| Transition matrix | T^(x), the S x S matrix governing state transitions when observation x is emitted |
| Variant | One of K_i alternative transition matrices for a factor, selected by the conditional dependency scheme |
| Vocabulary map | A mapping from a component's local token indices to global token indices in a nonergodic mixture |
