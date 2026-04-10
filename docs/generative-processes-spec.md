# Generative Processes Specification

**Version:** 1.0
**Status:** Draft

## 1 Introduction and Notation

### 1.1 Purpose

This document specifies the mathematical objects and observable behaviors that constitute a generative process system. The specification is intended to be sufficient for an independent ground-up reimplementation in any language or framework.

### 1.2 Scope

The core specification covers:

- Generalized hidden Markov models (GHMMs)
- Operations that must be possible on any generative process
- Composition schemes: factored processes and nonergodic mixtures
- Conditional dependency schemes for factored processes
- Conformance test vectors for verifying correctness

The core specification does **not** cover:

- Software architecture, programming language, or framework choices
- Algorithms or internal representations
- Implementation-specific conveniences or optimizations

Optional runtime interoperability conventions are collected in Appendix A and are not part of the mathematical or conformance requirements.

### 1.3 Notation

The following conventions are used within this document. They are not prescriptions for implementations.

| Symbol | Meaning |
|--------|---------|
| $V$ | Vocabulary size (number of distinct observations) |
| $S$ | State space size |
| $x$ | Observation token |
| $\mathcal{T}$ | Transition tensor in $\mathbb{R}^{V \times S \times S}$ |
| $T^{(x)}$ | The $x$-th $S \times S$ slice of $\mathcal{T}$ |
| $T$ | Net transition matrix: $T = \sum_x T^{(x)}$ |
| $\rho(T)$ | Spectral radius of $T$ |
| $\mathbf{w}$ | Normalizing right eigenvector of $T$ at eigenvalue 1 |
| $\boldsymbol{\pi}$ | Stationary left eigenvector of $T$ at eigenvalue 1, normalized to sum to 1 |
| $\boldsymbol{\eta}$ | GHMM state representative (row vector of dimension $S$) |
| $\widehat{\boldsymbol{\eta}}$ | Canonically normalized GHMM state representative |
| $\boldsymbol{\eta}_0$ | Initial state representative |
| $K_i$ | Number of variants for factor $i$ in a factored process |
| $F$ | Number of factors in a factored process |
| $C$ | Number of components in a nonergodic mixture |

**Conventions:**

- Mathematical indexing is **1-based** throughout the spec unless a section explicitly states otherwise.
- GHMM state representatives are **row vectors**. Two nonzero nonnegative row vectors $\boldsymbol{\eta}$ and $c \boldsymbol{\eta}$ with $c > 0$ represent the same predictive state.
- The formulas in §§3.4-3.6 are invariant under positive rescaling of the GHMM state representative.
- In the HMM subclass, the canonical representative coincides with the usual sum-to-1 hidden-state distribution.
- Input transition matrices use **row-vector convention**: the $(i, j)$ entry of $T^{(x)}$ is the nonnegative weight for transitioning from state $i$ to state $j$ while emitting observation $x$.

## 2 Operations

Given a generative process (base GHMM or composite), it must be possible to obtain the following results. The spec defines what results are required, not how they are organized or computed.

### 2.1 Observation probability distribution

Given a process state, compute the probability distribution over observations:

$$P(x \mid \text{state}) \quad \forall\, x$$

The result is a categorical distribution over the process vocabulary.

### 2.2 Observation sampling

Given a probability distribution over observations, sample an observation. This is standard categorical sampling and is not specific to generative processes.

### 2.3 State update

Given a prior process state and an observed token, compute the updated process state after conditioning on that observation.

### 2.4 Sequence probability

Given an observation sequence $x_1, \ldots, x_T$, compute the probability $P(x_1, \ldots, x_T)$ under the process's initial state.

### 2.5 Stationary objects

Some process types define stationary objects; others do not.

- For a base GHMM or HMM, the stationary objects are the stationary left eigenvector $\boldsymbol{\pi}$ and the associated canonical stationary state representative $\boldsymbol{\eta}_*$ of §3.8.
- For factored processes, see §4.7.
- For nonergodic mixtures, see §5.7.

The v1.0 conformance suite requires the `stationary_distribution` operation only for base `ghmm` and `hmm` process definitions.

## 3 Generalized Hidden Markov Model

A GHMM is a positive operator model. Its predictive state is represented by a nonzero nonnegative row vector up to positive scale. Only in the HMM subclass does the canonical representative coincide with an ordinary hidden-state probability distribution.

At each step, the current predictive state determines a distribution over observations. After observing a token, the state updates by applying the corresponding operator slice and re-normalizing in the $\mathbf{w}$ gauge.

### 3.1 Authoritative inputs

A GHMM is defined by transition matrices $T^{(x)} \in \mathbb{R}_{\ge 0}^{S \times S}$ for $x \in \{1, \ldots, V\}$ in the spectrally normalized gauge where the net matrix

$$
T = \sum_x T^{(x)}
$$

has spectral radius 1.

An optional initial state representative $\boldsymbol{\eta}_0$ may be provided. Any such representative must be a finite, nonzero, nonnegative row vector of dimension $S$. When omitted, the default predictive state is the stationary predictive state represented by $\boldsymbol{\pi}$ (§3.8).

### 3.2 Validity

The transition matrices $T^{(x)}$ must satisfy:

1. Every entry is finite and nonnegative.
2. The net transition matrix $T = \sum_x T^{(x)}$ has spectral radius 1.
3. The right eigenspace of $T$ at eigenvalue 1 is one-dimensional and admits a strictly positive eigenvector $\mathbf{w}$.
4. The left eigenspace of $T$ at eigenvalue 1 is one-dimensional and admits a nonnegative eigenvector $\boldsymbol{\pi}$.

These eigenvectors are normalized as follows:

$$
T \mathbf{w} = \mathbf{w}, \qquad \sum_i w_i = S
$$

and

$$
\boldsymbol{\pi} T = \boldsymbol{\pi}, \qquad \sum_i \pi_i = 1.
$$

A nonnegative irreducible $T$ is a sufficient condition for items 3-4, but implementations may verify them by any equivalent method.

*Non-normative note:* A producer may obtain the spectrally normalized form by dividing all $T^{(x)}$ by the Perron root of $\sum_x T^{(x)}$ before serialization.

#### 3.2.1 Hidden Markov Model

An HMM is the GHMM subclass with the additional constraint

$$
\sum_{x, s'} T^{(x)}[s, s'] = 1 \qquad \forall\, s.
$$

In that case,

$$
\mathbf{w} = \mathbf{1}.
$$

### 3.3 Derived quantities

From the net transition matrix $T$:

- **Normalizing eigenvector**

$$
T \mathbf{w} = \mathbf{w}, \qquad \sum_i w_i = S
$$

- **Stationary left eigenvector**

$$
\boldsymbol{\pi} T = \boldsymbol{\pi}, \qquad \sum_i \pi_i = 1
$$

### 3.4 Observation probability distribution

Given a GHMM state representative $\boldsymbol{\eta}$,

$$
P(x \mid \boldsymbol{\eta}) =
\frac{\boldsymbol{\eta} T^{(x)} \mathbf{w}}
{\boldsymbol{\eta} \cdot \mathbf{w}}.
$$

Because all matrices are nonnegative and $\mathbf{w}$ is strictly positive, this defines a valid categorical distribution for every admissible $\boldsymbol{\eta}$.

### 3.5 Belief state update

Given a state representative $\boldsymbol{\eta}$ and an observed token $x$,

$$
\boldsymbol{\eta}' =
\frac{\boldsymbol{\eta} T^{(x)}}
{\boldsymbol{\eta} T^{(x)} \cdot \mathbf{w}}.
$$

This returns a concrete representative of the posterior predictive state.

**Zero-denominator case:** If

$$
\boldsymbol{\eta} T^{(x)} \cdot \mathbf{w} = 0,
$$

the update is undefined. This corresponds to conditioning on an observation with zero probability under the current predictive state, or to numerical failure.

### 3.6 Sequence probability

Given an observation sequence $x_1, \ldots, x_T$ and an initial state representative $\boldsymbol{\eta}_0$,

$$
P(x_1, \ldots, x_T) =
\frac{\boldsymbol{\eta}_0 T^{(x_1)} T^{(x_2)} \cdots T^{(x_T)} \mathbf{w}}
{\boldsymbol{\eta}_0 \cdot \mathbf{w}}.
$$

For the empty sequence $\epsilon$,

$$
P(\epsilon) = 1.
$$

### 3.7 State representatives and canonical normalization

GHMM state representatives are defined up to positive scale. For any $c > 0$, $c \boldsymbol{\eta}$ represents the same predictive state as $\boldsymbol{\eta}$, and the quantities in §§3.4-3.6 are unchanged by this rescaling.

When a canonical representative is needed, use

$$
\widehat{\boldsymbol{\eta}} =
\frac{\boldsymbol{\eta}}{\boldsymbol{\eta} \cdot \mathbf{w}},
$$

which satisfies

$$
\widehat{\boldsymbol{\eta}} \cdot \mathbf{w} = 1.
$$

In the HMM subclass, this reduces to ordinary sum-to-1 normalization.

### 3.8 Stationary distribution

The stationary left eigenvector $\boldsymbol{\pi}$ is defined by

$$
\boldsymbol{\pi} T = \boldsymbol{\pi}, \qquad \sum_i \pi_i = 1.
$$

It represents the stationary predictive state. The canonical representative of this same predictive state is

$$
\boldsymbol{\eta}_* =
\frac{\boldsymbol{\pi}}{\boldsymbol{\pi} \cdot \mathbf{w}}.
$$

This canonical representative satisfies

$$
\boldsymbol{\eta}_* \cdot \mathbf{w} = 1
\qquad \text{and} \qquad
\boldsymbol{\eta}_* T = \boldsymbol{\eta}_*.
$$

When `initial_state` is omitted, the default predictive state is this stationary predictive state. Using $\boldsymbol{\pi}$ or $\boldsymbol{\eta}_*$ as its representative yields identical observable behavior. In the HMM subclass, $\mathbf{w} = \mathbf{1}$, so $\boldsymbol{\eta}_* = \boldsymbol{\pi}$.

## 4 Composition: Factored Processes

A factored process in v1.0 is defined by $F$ factors. Factor $i$ is a family of $K_i$ local GHMM or HMM variants

$$
\{T_{i \mid k}^{(x^i)}\}_{k=1}^{K_i}
$$

sharing a common local vocabulary size $V_i$ and local state dimension $S_i$. Each factor has its own state representative $\boldsymbol{\eta}^i$.

v1.0 factored processes are therefore GHMM/HMM-factor compositions. A more general factor-interface abstraction is reserved for a future version.

### 4.1 Authoritative inputs

A factored process is defined by:

- $F$ factor families, one per factor
- an emission selector for each factor

$$
\sigma_i^{\mathrm{emit}} :
\prod_{j \in \mathrm{deps}_i^{\mathrm{emit}}} \{1, \ldots, V_j\}
\to \{1, \ldots, K_i\}
$$

- a transition selector for each factor

$$
\sigma_i^{\mathrm{trans}} :
\prod_{j \in \mathrm{deps}_i^{\mathrm{trans}}} \{1, \ldots, V_j\}
\to \{1, \ldots, K_i\}
$$

- a composite-token encoding (§4.3.2)

The single-selector case is the special case

$$
\sigma_i^{\mathrm{emit}} = \sigma_i^{\mathrm{trans}}.
$$

For every factor, parent observations are ordered by increasing factor index when they are supplied to a selector. Write this ordered tuple as

$$
\mathbf{x}_{\mathrm{deps}_i^{\mathrm{emit}}}^{\uparrow}
= (x^j)_{j \in \mathrm{deps}_i^{\mathrm{emit}}}^{\uparrow}
$$

for emission selectors, and analogously for transition selectors.

If $K_i = 1$ and factor $i$ omits `initial_state`, its default state is the stationary predictive state of its sole local variant. If $K_i > 1$, the factor initial state must be explicit.

### 4.2 Validity

The only exact emission schemes normatively defined in v1.0 are the supported ones in §4.6.1: independent emissions and sequential-chain emissions. The **transition dependency graph** induced by $\mathrm{deps}_i^{\mathrm{trans}}$ is not required to be acyclic, because transitions are applied only after the full observation tuple for the current time step is known.

If the emission dependency graph is cyclic, the only defined v1.0 semantics is the fully conditional approximation of §4.6.3.

Except in the $Z = 0$ fallback branch of §4.6.3, a factor update denominator of zero is undefined. For every reachable prior factor-state tuple, any composite observation assigned positive probability by the current joint law must induce defined selected updates for all factors.

### 4.3 Observation probability distribution

The observation law of a factored process is a joint distribution over composite tokens, assembled from per-factor emission distributions according to the chosen emission dependency scheme.

#### 4.3.1 Per-factor emission distribution

For factor $i$, let

$$
k_i^{\mathrm{emit}} =
\sigma_i^{\mathrm{emit}}(\mathbf{x}_{\mathrm{deps}_i^{\mathrm{emit}}}^{\uparrow}).
$$

Its local emission distribution is

$$
P(x^i \mid \boldsymbol{\eta}^i, k_i^{\mathrm{emit}}) =
\frac{
  \boldsymbol{\eta}^i T_{i \mid k_i^{\mathrm{emit}}}^{(x^i)}
  \mathbf{w}_{i \mid k_i^{\mathrm{emit}}}
}{
  \boldsymbol{\eta}^i \cdot \mathbf{w}_{i \mid k_i^{\mathrm{emit}}}
}.
$$

Each $\mathbf{w}_{i \mid k}$ is derived from the corresponding variant family using the GHMM rules of §3.

#### 4.3.2 Composite token encoding

The composite-token encoding is part of the authoritative process definition.

- If no explicit encoding is supplied, the composite vocabulary is the full product space and uses the mixed-radix encoding of §4.3.3.
- If a strict subset of tuples is encoded, the process definition must provide an explicit bijection between composite token IDs and per-factor tuples.

Full-product encodings may include tuples that are unreachable and therefore always have zero probability. Sparse encodings may exclude them, but then the explicit map is mandatory. A composite token outside the declared encoding is outside the process vocabulary and therefore invalid.

#### 4.3.3 Radix encoding

When the composite vocabulary is the full product $V_{\text{composite}} = V_1 \cdots V_F$, a mixed-radix big-endian encoding provides a dense bijection in the mathematical 1-based convention.

**Encoding**:

$$
x = \sum_{i=1}^{F} (x^i - 1) \cdot m_i + 1
$$

where

$$
m_i = \prod_{j=i+1}^{F} V_j.
$$

**Decoding**:

$$
x^i = \left\lfloor \frac{x - 1}{m_i} \right\rfloor \bmod V_i + 1.
$$

**Invariant**:

$$
\operatorname{decode}(\operatorname{encode}(x^1, \ldots, x^F))
= (x^1, \ldots, x^F).
$$

Conformance JSON uses 0-based serialized token IDs; see §7.1 for the serialization convention.

#### 4.3.4 Joint distribution

For supported DAG emission topologies, the joint distribution is computed exactly by evaluating factors in emission topological order. For the fully conditional approximation, the joint is defined by §4.6.3.

### 4.4 State update

Given a composite observation $x_t$ corresponding to the tuple $(x_t^1, \ldots, x_t^F)$, compute for each factor

$$
k_i^{\mathrm{trans}} =
\sigma_i^{\mathrm{trans}}(\mathbf{x}_{t,\mathrm{deps}_i^{\mathrm{trans}}}^{\uparrow}),
$$

where

$$
\mathbf{x}_{t,\mathrm{deps}_i^{\mathrm{trans}}}^{\uparrow}
= (x_t^j)_{j \in \mathrm{deps}_i^{\mathrm{trans}}}^{\uparrow}.
$$

If the process is using the fully conditional approximation of §4.6.3 and the pre-normalization constant $Z$ for the current prior state tuple is zero, then all factor states remain unchanged after observing $x_t$.

Otherwise each factor updates by

$$
\boldsymbol{\eta}_t^i =
\frac{
  \boldsymbol{\eta}_{t-1}^i T_{i \mid k_i^{\mathrm{trans}}}^{(x_t^i)}
}{
  \boldsymbol{\eta}_{t-1}^i T_{i \mid k_i^{\mathrm{trans}}}^{(x_t^i)}
  \cdot \mathbf{w}_{i \mid k_i^{\mathrm{trans}}}
}.
$$

Outside the $Z = 0$ fallback branch, a zero denominator is undefined.

### 4.5 Sequence probability

Given a composite observation sequence $x_1, \ldots, x_T$ and initial per-factor states $(\boldsymbol{\eta}_0^1, \ldots, \boldsymbol{\eta}_0^F)$,

$$
P(x_1, \ldots, x_T) =
\prod_{t=1}^{T}
P_{\text{joint}}(x_t \mid \boldsymbol{\eta}_{t-1}^1, \ldots, \boldsymbol{\eta}_{t-1}^F),
$$

where $P_{\text{joint}}$ is the current joint observation law and the factor states update according to §4.4.

For the empty sequence $\epsilon$,

$$
P(\epsilon) = 1.
$$

### 4.6 Conditional dependencies

A factored process may use distinct selectors for same-step emissions and post-observation transitions. Emission selectors determine the joint observation law for the current time step. Transition selectors determine how factor states are updated after the full observation tuple is known. The single-selector case is the special case where the two selector families are identical.

#### 4.6.1 Emission topologies

##### 4.6.1.1 Independent

No factor's emission depends on any other factor's same-step emission. Then

$$
P(x^1, \ldots, x^F) =
\prod_{i=1}^{F} P(x^i \mid \boldsymbol{\eta}^i, \sigma_i^{\mathrm{emit}}(())).
$$

##### 4.6.1.2 Sequential chain

Factor 1 has no emission parents, and factor $i$ for $i \ge 2$ depends only on factor $i - 1$:

$$
P(x^1, \ldots, x^F) =
P(x^1 \mid \boldsymbol{\eta}^1, \sigma_1^{\mathrm{emit}}(()))
\prod_{i=2}^{F}
P(x^i \mid \boldsymbol{\eta}^i, \sigma_i^{\mathrm{emit}}(x^{i-1})).
$$

##### 4.6.1.3 Other DAG topologies

Other DAG topologies are reserved for a future version and are not normatively defined in v1.0. v1.0 normatively defines only:

- independent emissions
- sequential-chain emissions
- the fully conditional emission approximation
- hybrid conditional-transition schemes built on those emission modes

#### 4.6.2 Variant constraints

For every factor, the codomain of each selector must be contained in $\{1, \ldots, K_i\}$. Every selector output must index a valid variant family.

#### 4.6.3 Fully conditional approximation

When every factor's emission depends on the other factors' same-step emissions, the emission dependency graph is cyclic and the true joint is not available in closed form. v1.0 defines the following product-of-conditionals approximation:

$$
P_{\text{unnorm}}(x^1, \ldots, x^F) =
\prod_{i=1}^{F}
P\!\left(
  x^i \mid
  \boldsymbol{\eta}^i,
  \sigma_i^{\mathrm{emit}}((x^j)_{j \ne i}^{\uparrow})
\right)
$$

and

$$
P(x^1, \ldots, x^F) =
\frac{P_{\text{unnorm}}(x^1, \ldots, x^F)}{Z},
\qquad
Z = \sum_{\mathbf{x} \in \mathcal{X}_{\mathrm{enc}}} P_{\text{unnorm}}(\mathbf{x}).
$$

Here $\mathcal{X}_{\mathrm{enc}}$ is the set of per-factor tuples that are present in the authoritative composite-token encoding. For the default dense encoding, $\mathcal{X}_{\mathrm{enc}}$ is the full product space. For a sparse encoding, the sum is taken only over explicitly encoded tuples.

**Zero-mass fallback:** If $Z = 0$, the joint distribution over the chosen composite vocabulary falls back to uniform:

$$
P(x^1, \ldots, x^F) = \frac{1}{V_{\text{composite}}}.
$$

In this branch, observing any composite token leaves all factor states unchanged.

#### 4.6.4 Hybrid conditional-transitions scheme

A hybrid factored process may use one supported emission scheme (fixed or sequential in v1.0) together with an arbitrary transition-selector family, including selectors conditioned on all non-self observations in the current tuple. For v1.0 `conditional_transitions` serialization, factor $i$'s transition selector is evaluated on the ordered tuple of all factor observations except $x^i$, in increasing factor-index order. This scheme is normative in v1.0 and is serialized in the conformance tests with separate emission and transition control maps.

### 4.7 Default initial state and stationary objects

A factored process has a default initial state whenever every factor has a defined default initial state. In that case, the default composite initial state is the ordered tuple of the per-factor defaults.

For independent factors with a single variant each, this default composite initial state is stationary and equals the tuple of the per-factor stationary predictive states.

For other factored-process couplings, v1.0 does not define a general stationary object. The existence of a default initial state does not imply stationarity.

## 5 Composition: Nonergodic Mixture

A nonergodic mixture combines $C$ component processes into a single generative process with Bayesian component tracking.

### 5.1 Authoritative inputs

A nonergodic mixture is defined by:

- component processes, each independently defined under its own process type
- component weights $\boldsymbol{\alpha}_{\text{mix}} \in \mathbb{R}^C$ with $\alpha_{\text{mix},i} \ge 0$ for all $i$ and $\sum_i \alpha_{\text{mix},i} = 1$
- vocabulary mappings

For each component $i$, let

$$
\phi_i : \{1, \ldots, V_i\} \to \{1, \ldots, V_{\text{global}}\}
$$

be an injective local-to-global token map, and let

$$
S_i = \operatorname{Im}(\phi_i)
$$

denote its image. The images $S_i$ may overlap across components. The inverse $\phi_i^{-1}(x)$ is defined exactly when $x \in S_i$.

### 5.2 State structure

The state of a nonergodic mixture consists of:

- component beliefs $\boldsymbol{\beta} \in \mathbb{R}^C$ with $\beta_i \ge 0$ for all $i$ and $\sum_i \beta_i = 1$
- per-component states $(s^1, \ldots, s^C)$, using the state representation appropriate to each component process

### 5.3 Observation probability distribution

$$
P(x \mid \boldsymbol{\beta}, s^1, \ldots, s^C)
=
\sum_{i=1}^{C}
\beta_i \,
\mathbf{1}\{x \in S_i\} \,
P_i(\phi_i^{-1}(x) \mid s^i),
$$

where $P_i(\cdot \mid s^i)$ is component $i$'s local observation distribution under its own process semantics.

### 5.4 State update

Define

$$
\ell_i =
\mathbf{1}\{x \in S_i\} \,
P_i(\phi_i^{-1}(x) \mid s^i).
$$

If

$$
L = \sum_j \beta_j \ell_j > 0,
$$

then

$$
\beta_i' = \frac{\beta_i \ell_i}{L}.
$$

Each component state updates by its own process update rule:

$$
s^{i\prime} =
\begin{cases}
\operatorname{Update}_i(s^i, \phi_i^{-1}(x)) & \text{if } \ell_i > 0, \\
s^i & \text{otherwise.}
\end{cases}
$$

If $L = 0$, then $\boldsymbol{\beta}$ and all $s^i$ remain unchanged.

### 5.5 Sequence probability

Given an initial mixture state $(\boldsymbol{\beta}_0, s_0^1, \ldots, s_0^C)$,

$$
P(x_1, \ldots, x_T) =
\sum_{i=1}^{C}
\beta_{0,i} \,
\mathbf{1}\{\forall t,\ x_t \in S_i\} \,
P_i(\phi_i^{-1}(x_1), \ldots, \phi_i^{-1}(x_T) \mid s_0^i).
$$

For the empty sequence $\epsilon$,

$$
P(\epsilon) = 1.
$$

Under the default initial state of §5.7, $\boldsymbol{\beta}_0 = \boldsymbol{\alpha}_{\text{mix}}$.

### 5.6 Generation

Generation from an initial mixture state samples one component once from the initial component-belief vector $\boldsymbol{\beta}_0$, generates the full local sequence from that component, and maps each local token through $\phi_i$. Under the default initial state of §5.7, $\boldsymbol{\beta}_0 = \boldsymbol{\alpha}_{\text{mix}}$. There is no mid-sequence switching.

If a generation API exposes a final state, the normative final state is the public process state obtained by filtering the emitted global sequence under the mixture update rule above, not merely the latent chosen component. An implementation may additionally expose the latent chosen component as auxiliary output.

### 5.7 Stationary objects

A nonergodic mixture does not define a unique global stationary state for the Bayesian-tracking process.

Its default initial state is

$$
(\boldsymbol{\alpha}_{\text{mix}}, s_0^1, \ldots, s_0^C),
$$

where each component uses its own default initial state when not explicitly provided. This default is not generally stationary.

## 6 Addendum: Sequence Generation

The preceding sections define the mathematical objects and their properties. This section specifies additional behavior for generating finite sequences from those processes.

### 6.1 Sequence generation

Given a generative process and an initial state, generate a sequence of $n$ observations by repeatedly sampling from the current observation distribution and updating the state.

### 6.2 Sequence augmentation

Raw generated sequences may be augmented with framing tokens:

- **BOS**: prepended before the first body token
- **EOS**: appended after the last body token
- **PAD**: fills remaining positions to reach a target length

BOS, EOS, and PAD token indices must lie outside the process vocabulary. They do not participate in process-state updates.

When all augmentations are applied, the output layout is

$$
[\text{BOS},\, \text{body}_1, \ldots, \text{body}_n,\, \text{EOS},\, \text{PAD}, \ldots, \text{PAD}]
$$

padded to total length $L \ge n + 2$. There is no early termination; the process always runs for exactly $n$ body steps.

### 6.3 Batched generation

Sequences must be producible in batches: a collection of generated sequences, all with the same total output length $L$.

## 7 Conformance Test Vectors

### 7.1 Encoding scheme

**Index convention note:** The mathematics in this document is 1-based. JSON serialization is 0-based.

- Array positions in JSON are 0-based in the usual programming sense.
- Variant IDs stored in JSON scalar fields such as `control_maps` outputs and `emission_variant_indices` are also 0-based.
- Serialized token IDs in the conformance fixtures are 0-based unless a test explicitly states otherwise.

#### 7.1.1 Process definitions

Process definitions are encoded as JSON objects with a `type` discriminator.

**Base GHMM:**

```json
{
  "type": "ghmm",
  "transition_matrices": [[[...]]]
}
```

`transition_matrices` is a 3D array of shape `[V, S, S]` in row-vector convention and must already be in the spectrally normalized gauge of §3.1. An optional `initial_state` field provides a GHMM state representative. When omitted, the default predictive state is the stationary predictive state of §3.8.

**HMM:**

```json
{
  "type": "hmm",
  "transition_matrices": [[[...]]]
}
```

The `hmm` tag indicates that the HMM-specific validity constraints of §3.2.1 apply.

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

For each factor, `transition_matrices` is a 4D array of shape `[K, V, S, S]`. For v1.0 factored processes, each factor's `component_type` must be either `"ghmm"` or `"hmm"`. For each factor, each per-variant `[V, S, S]` block of `transition_matrices` must already be in the spectrally normalized gauge of §3.1. If `vocab_size` is present, it must equal the `V` dimension of that factor's `transition_matrices`. If a factor has `K = 1` and omits `initial_state`, it defaults to the stationary predictive state of its sole local variant. If a factor has `K > 1`, its `initial_state` must be explicit. A factored process may additionally include a `composite_encoding` object when it uses a sparse composite vocabulary.

**Nonergodic mixture:**

```json
{
  "type": "nonergodic",
  "components": [...],
  "weights": [...],
  "vocab_maps": [[...], ...]
}
```

Here the serialized `weights` field stores the mixture-weight vector $\boldsymbol{\alpha}_{\text{mix}}$ of §5.1. In `vocab_maps`, each inner array stores 0-based global token IDs. For component $i$, the length of `vocab_maps[i]` must equal that component's local vocabulary size, and the entries of `vocab_maps[i]` must be distinct.

#### 7.1.2 State encoding

- **GHMM state:** a 1D array of dimension $S$ representing a nonzero nonnegative row vector
- **Factored state:** an ordered list of factor-state encodings
- **Nonergodic state:** an object with `component_beliefs` and `component_states`

GHMM state outputs in conformance fixtures are serialized in canonical form

$$
\widehat{\boldsymbol{\eta}} =
\frac{\boldsymbol{\eta}}{\boldsymbol{\eta} \cdot \mathbf{w}},
$$

unless a test explicitly states otherwise. GHMM state inputs may use any valid nonzero nonnegative representative.

Whenever a GHMM state appears anywhere in an expected output state encoding, whether as a top-level GHMM state, as a factor state inside a factored state, or as a component state inside a nonergodic state, it is serialized in canonical form unless a test explicitly states otherwise. This requirement applies recursively.

#### 7.1.3 Structure definitions

```json
{"type": "independent"}

{"type": "independent", "variant_indices": [0, 2, 0]}

{"type": "sequential", "control_maps": [null, [0, 1, ...]]}

{"type": "fully_conditional", "control_maps": [[...], [...]]}

{"type": "conditional_transitions",
 "transition_control_maps": [[...], [...]],
 "emission_control_maps": [null, [1, 0]]}
```

`control_maps` is shorthand for the single-selector case where emission and transition selectors are identical.

The supported structure types determine the parent tuple for each selector:

- `independent`: every selector is parentless and constant. If `variant_indices` is omitted, every factor uses constant selector value `0`. If present, `variant_indices[i]` is the 0-based selected variant for factor `i` in the single-selector case.
- `sequential`: factor 1 has a constant emission selector; factor $i \ge 2$ uses parent tuple $(x^{i-1})$
- `fully_conditional`: factor $i$ uses the ordered tuple of all factor observations except $x^i$
- `conditional_transitions`: emissions are either constant (`emission_variant_indices`) or sequential (`emission_control_maps`), while transition selectors for factor $i$ use the ordered tuple of all factor observations except $x^i$

For `conditional_transitions`, emissions are determined by exactly one of `emission_variant_indices` or `emission_control_maps`, while transitions are determined by `transition_control_maps`.

For every factor, parent observations are serialized as 0-based local token IDs ordered by increasing parent factor index. If factor $i$ has ordered parents $p_1 < \cdots < p_m$ with local vocabulary sizes $V_{p_r}$ and observed serialized parent tokens $\tilde{x}^{p_r}$, then the flattened control-map lookup index is

$$
\operatorname{idx}
=
\sum_{r=1}^{m}
\tilde{x}^{p_r}
\prod_{q=r+1}^{m} V_{p_q}.
$$

Then the relevant control-map entry at that index returns a 0-based serialized variant ID. The required control-map length is therefore

$$
\prod_{r=1}^{m} V_{p_r}.
$$

For a parentless constant selector, the serialized form is:

- `variant_indices[i]` for the `independent` single-selector case
- a scalar entry in `emission_variant_indices` when independent fixed emissions are being used
- otherwise a length-1 array containing the selected 0-based variant ID

If a factored process uses a sparse composite vocabulary, its process definition must include an explicit composite-token encoding object, for example:

```json
"composite_encoding": {
  "type": "explicit",
  "token_to_tuple": [[0, 0], [0, 1], [1, 0]]
}
```

where the composite token ID is the 0-based array position.

**Initial state convention:** When a process definition omits `initial_state`, the process uses its own default initial-state rule. For a base GHMM, that is the stationary predictive state of §3.8. The `stationary_distribution` operation is defined in v1.0 conformance only for base `ghmm` and `hmm` process definitions. It returns $\boldsymbol{\pi}$, the stationary left eigenvector normalized to sum to 1.

### 7.2 Tolerance

All numerical comparisons use both relative and absolute tolerance. A computed scalar $a$ matches an expected scalar $b$ iff

$$
|a - b| \le \text{abs\_tol} + \text{rel\_tol} \, |b|.
$$

Unless otherwise noted, use

$$
\text{rel\_tol} = 10^{-6}, \qquad \text{abs\_tol} = 10^{-12}.
$$

For vectors, matrices, and tensors, apply this criterion elementwise.

### 7.3 Generation tests

Conformance vectors cover deterministic operations. Sequence-generation augmentations (§6) are behaviorally specified but not numerically tested, because sampling depends on RNG implementation.

### 7.4 Test vectors

Test vectors are provided in the companion file `conformance-tests.json`. Each test vector has these required fields:

Unless a category explicitly tests a specified fallback (`fully_conditional_zero_fallback` or `nonergodic_zero_likelihood`), conformance vectors use valid process definitions and valid operation inputs. Behavior on invalid process definitions, out-of-range selector outputs, composite tokens outside the declared encoding, and base-GHMM zero-denominator updates is outside v1.0 conformance.

- `id`: unique identifier
- `category`: which aspect is being tested
- `description`: human-readable description
- `operation`: which operation is being tested
- `input`: operation-specific input
- `expected`: expected output

And these optional fields:

- `process`: process definition (§7.1.1)
- `tolerance`: relative tolerance override
- `abs_tolerance`: absolute tolerance override
- `notes`: human-readable explanation of the expected value

Categories covered:

1. **ghmm_observation_distribution**: observation distribution from a GHMM
2. **ghmm_belief_update**: GHMM state update after observing a token
3. **ghmm_sequence_probability**: probability of an observation sequence
4. **ghmm_hmm_case**: HMM special case where $\mathbf{w} = \mathbf{1}$
5. **ghmm_nontrivial_w**: GHMM with nontrivial normalizing eigenvector
6. **ghmm_stationary_distribution**: stationary left eigenvector computation
7. **ghmm_projective_invariance**: invariance under positive rescaling of GHMM states
8. **factored_token_encoding**: token encoding and decoding
9. **factored_sparse_encoding**: explicit sparse composite encodings
10. **factored_independent**: independent factored joint distribution
11. **factored_sequential**: sequential-chain factored joint distribution
12. **factored_belief_update**: factored-process state updates
13. **factored_sequence_probability**: multi-step factored sequence probability
14. **factored_fully_conditional**: fully conditional approximation
15. **factored_conditional_transitions**: hybrid conditional-transition scheme
16. **fully_conditional_zero_fallback**: uniform fallback when $Z = 0$
17. **nonergodic_observation_distribution**: mixture observation distribution
18. **nonergodic_belief_update**: mixture Bayesian component update
19. **nonergodic_sequence_probability**: mixture sequence probability
20. **nonergodic_zero_likelihood**: zero-likelihood fallback
21. **nonergodic_vocab_mapping**: differing component vocabulary maps
22. **generation_layout**: BOS/EOS/PAD layout verification

## 8 Glossary

| Term | Definition |
|------|-----------|
| Belief state ($\boldsymbol{\eta}$) | The predictive state of a GHMM, represented by a nonzero nonnegative row vector up to positive scale. In the HMM subclass, the canonical representative is the usual hidden-state probability distribution. |
| BOS | Beginning-of-sequence token; a framing token prepended to generated sequences |
| Component | One constituent process in a nonergodic mixture |
| Composite token | A single integer encoding the joint observation tuple of all factors in a factored process |
| Control map | A lookup-table encoding of an emission selector or transition selector used in the conformance tests |
| Emission selector ($\sigma_i^{\mathrm{emit}}$) | A function selecting the emission variant for factor $i$ from the relevant same-step parent observations |
| EOS | End-of-sequence token; a framing token appended after the last body token |
| Factor | One constituent process in a factored process |
| GHMM | Generalized hidden Markov model; a positive operator model with projective state representatives |
| HMM | Hidden Markov model; the GHMM subclass where $\mathbf{w} = \mathbf{1}$ |
| Net transition matrix | $T = \sum_x T^{(x)}$ |
| Normalizing eigenvector | The strictly positive right eigenvector $\mathbf{w}$ of $T$ at eigenvalue 1 |
| Observation | A discrete token emitted by the process |
| PAD | Padding token used to extend generated sequences to a target length |
| Perron root | For a nonnegative matrix, the eigenvalue equal to its spectral radius |
| Radix encoding | Mixed-radix positional encoding of per-factor tokens into a composite token |
| Spectral radius | The maximum absolute value of the eigenvalues of a matrix |
| Stationary distribution | The normalized left eigenvector $\boldsymbol{\pi}$ of $T$ at eigenvalue 1. The associated canonical stationary state representative is $\boldsymbol{\eta}_* = \boldsymbol{\pi} / (\boldsymbol{\pi} \cdot \mathbf{w})$. |
| Transition selector ($\sigma_i^{\mathrm{trans}}$) | A function selecting the transition variant for factor $i$ after the full current observation tuple is known |
| Variant | One of the alternative transition-matrix families available to a factor |
| Vocabulary mapping ($\phi_i$) | An injective map from a component's local token indices to global token indices in a nonergodic mixture |

## Appendix A: Optional Runtime Interoperability Profile

This appendix is not part of the mathematical or conformance requirements. It describes optional runtime conventions that implementations may choose to support.

### A.1 Device profile

An implementation may choose to expose generated batches on CPU, NVIDIA GPUs (CUDA), or both.

### A.2 DLPack interoperability

An implementation may choose to expose generated tensors through [DLPack](https://dmlc.github.io/dlpack/latest/) so they can be consumed by other frameworks without copying.
