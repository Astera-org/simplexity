# tests/test_rpn_arithmetic_process.py
import math

import chex
import jax
import jax.numpy as jnp
import pytest

from simplexity.generative_processes.arithmetic_process import Operators, RPNArithmeticProcess, SpecialTokens

# ---------- Fixtures ----------


@pytest.fixture(scope="module")
def proc():
    # Small, nontrivial vocab: p=7 values, 3 operators, 4 specials => V=14
    return RPNArithmeticProcess(p=7, operators=[Operators.ADD, Operators.SUB, Operators.MUL], max_operations=5)


@pytest.fixture
def rng():
    return jax.random.PRNGKey(0)


# ---------- Helpers ----------


def make_binary_matrix_from_group_id(group_id: jnp.ndarray) -> jnp.ndarray:
    """6 x V binary matrix from group_id (0..5)."""
    G = 6
    return (group_id[None, :] == jnp.arange(G)[:, None]).astype(jnp.float32)  # (6, V)


# ---------- format_loss: core correctness ----------


def test_group_masks_match_naive_aggregation(proc: RPNArithmeticProcess, rng):
    """group_log_probs via masks == log(sum probs) via explicit grouping."""
    V = proc.vocab_size
    key = rng
    logits = jax.random.normal(key, (2, 3, V))  # (B,T,V)

    # Your path: group_masks + logsumexp
    log_probs = jax.nn.log_softmax(logits, axis=-1)  # (...,V)
    lp_exp = log_probs[..., None, :]  # (...,1,V)
    gmask = proc.group_masks[None, ...]  # (1,6,V)
    group_log_probs = jax.nn.logsumexp(lp_exp + gmask, axis=-1)  # (...,6)

    # Naive path: softmax then matmul with 6xV binary matrix
    probs = jax.nn.softmax(logits, axis=-1)  # (...,V)
    binary = make_binary_matrix_from_group_id(proc.group_id)  # (6,V)
    # (...,V) @ (V,6) => (...,6)
    group_probs = probs @ binary.T
    group_log_probs_naive = jnp.log(jnp.clip(group_probs, a_min=jnp.finfo(probs.dtype).tiny))

    chex.assert_trees_all_close(group_log_probs, group_log_probs_naive, atol=1e-3, rtol=1e-3)


def test_format_loss_indices_matches_naive(proc: RPNArithmeticProcess, rng):
    """Loss computed with masks equals a naive probability-space computation."""
    V = proc.vocab_size
    B, T = 4, 7
    key1, key2 = jax.random.split(rng)
    logits = jax.random.normal(key1, (B, T, V))
    labels = jax.random.randint(key2, (B, T), 0, V)

    pad_id = proc.tokens[SpecialTokens.PAD.value]
    # sprinkle some PAD to test masking
    labels = labels.at[0, :2].set(pad_id)

    loss = proc.format_loss(logits, labels, label_is_index=True, pad_id=pad_id)

    # Naive: probs -> aggregate -> pick target group -> mask PAD -> mean
    probs = jax.nn.softmax(logits, axis=-1)  # (B,T,V)
    binary = make_binary_matrix_from_group_id(proc.group_id)  # (6,V)
    group_probs = probs @ binary.T  # (B,T,6)
    y_group = jnp.take(proc.group_id, labels)  # (B,T)
    eps = jnp.finfo(probs.dtype).tiny
    picked = jnp.take_along_axis(group_probs, y_group[..., None], axis=-1)[..., 0]  # (B,T)
    nll = -jnp.log(jnp.clip(picked, a_min=eps))
    valid = labels != pad_id
    denom = jnp.maximum(1, jnp.sum(valid))
    loss_naive = jnp.sum(jnp.where(valid, nll, 0.0)) / denom

    chex.assert_trees_all_close(loss, loss_naive, atol=1e-3, rtol=1e-3)


def test_format_loss_onehot(proc: RPNArithmeticProcess, rng):
    """One-hot labels path works and equals index path for the same targets (no PAD)."""
    V = proc.vocab_size
    B, T = 3, 5
    key1, key2 = jax.random.split(rng)
    logits = jax.random.normal(key1, (B, T, V))
    y_idx = jax.random.randint(key2, (B, T), 0, V)
    labels_oh = jax.nn.one_hot(y_idx, V)

    loss_idx = proc.format_loss(logits, y_idx, label_is_index=True, pad_id=None)
    loss_oh = proc.format_loss(logits, labels_oh, label_is_index=False, pad_id=None)

    chex.assert_trees_all_close(loss_idx, loss_oh, atol=1e-6, rtol=1e-6)


def test_format_loss_jittable_and_vmap(proc: RPNArithmeticProcess, rng):
    V = proc.vocab_size
    B, T = 2, 6
    key1, key2 = jax.random.split(rng)
    logits = jax.random.normal(key1, (B, T, V))
    labels = jax.random.randint(key2, (B, T), 0, V)
    pad_id = proc.tokens[SpecialTokens.PAD.value]

    def f(L):
        return proc.format_loss(L, labels, label_is_index=True, pad_id=pad_id)

    loss_eager = f(logits)
    loss_jit = jax.jit(f)(logits)
    chex.assert_trees_all_close(loss_eager, loss_jit, atol=1e-6, rtol=1e-6)

    # vmap over batch: per-item losses should average to the unbatched mean if we do it manually
    def per_item(L_i, y_i):
        return proc.format_loss(L_i[None, ...], y_i[None, ...], label_is_index=True, pad_id=pad_id)

    vmapped = jax.vmap(per_item)(logits, labels)  # (B,)
    # Compare mean of per-item to single-call loss (they can differ if masking differs,
    # but here both use uniform masking logic, so they should match).
    chex.assert_trees_all_close(jnp.mean(vmapped), loss_eager, atol=1e-1, rtol=1e-1)


def test_format_loss_has_finite_grad(proc: RPNArithmeticProcess, rng):
    V = proc.vocab_size
    B, T = 2, 4
    key1, key2 = jax.random.split(rng)
    logits = jax.random.normal(key1, (B, T, V))
    labels = jax.random.randint(key2, (B, T), 0, V)
    pad_id = proc.tokens[SpecialTokens.PAD.value]

    def loss_fn(L):
        return proc.format_loss(L, labels, label_is_index=True, pad_id=pad_id)

    g = jax.grad(loss_fn)(logits)
    chex.assert_tree_all_finite(g)


# ---------- Token/group sanity ----------


def test_group_id_and_tokens_agree(proc: RPNArithmeticProcess):
    p = proc.p
    num_ops = len(proc.operators)
    V = proc.vocab_size

    # operands
    assert jnp.all(proc.group_id[:p] == 0)
    # operators
    assert jnp.all(proc.group_id[p : p + num_ops] == 1)
    # specials
    assert proc.group_id[p + num_ops + 0] == 2  # EQL
    assert proc.group_id[p + num_ops + 1] == 3  # BOE
    assert proc.group_id[p + num_ops + 2] == 4  # EOE
    assert proc.group_id[p + num_ops + 3] == 5  # PAD
    assert proc.group_id.shape[0] == V

    # masks: zero on in-group, -inf elsewhere
    g0 = proc.group_masks[0]  # operands
    assert jnp.all(jnp.isfinite(g0[:p]))
    assert jnp.all(~jnp.isfinite(g0[p:]))


# ---------- RPN behavior ----------


@pytest.mark.parametrize("k", [1, 2, 3, 4])
def test_random_sub_equation_is_valid(proc: RPNArithmeticProcess, rng, k):
    key = jax.random.fold_in(rng, k)
    n, sub = proc.random_sub_equation(key, k)
    assert n == (2 * k + 1)
    # All tokens must be values or operators (no specials/pad in sub-equation)
    assert proc.valid_sub_equation(sub, int(n))


def test_child_sub_equation_progress(proc: RPNArithmeticProcess, rng):
    key = rng
    n, sub = proc.random_sub_equation(key, 3)  # n=7
    new_n, new_sub = proc.child_sub_equation(sub)
    # Should never increase token count; unless already 1, it should usually reduce.
    assert new_n <= n
    assert new_sub.shape == sub.shape


def test_full_equation_structure(proc: RPNArithmeticProcess, rng):
    # Build a full equation; check BOE/EQL/EOE/PAD placement sanity.
    seq_len = 128
    # The generate method is vmapped over keys, so we need to pass a batch of keys
    keys = jax.random.split(rng, 1)  # Create a batch of 1 key
    _, seq = proc.generate(
        proc.initial_state, keys, seq_len, False
    )  # vmapped returns (state, batch) but our generate vmaps over keys; here we pass single key; unwrap:
    seq = seq[0]  # shape (seq_len,) - take the first (and only) sequence

    toks = proc.tokens
    BOE = toks[SpecialTokens.BOE.value]
    EOE = toks[SpecialTokens.EOE.value]
    EQL = toks[SpecialTokens.EQL.value]
    PAD = toks[SpecialTokens.PAD.value]

    # Exactly one BOE and one EOE present
    assert (seq == BOE).sum() == 1
    assert (seq == EOE).sum() == 1

    # BOE appears before EOE
    first_boe = int(jnp.argmax(seq == BOE))
    first_eoe = int(jnp.argmax(seq == EOE))
    assert first_boe < first_eoe

    # Everything after EOE should be PAD
    assert jnp.all(seq[first_eoe + 1 :] == PAD)

    # Before EOE: tokens should be operands/operators/EQL/BOE (no PAD)
    p = proc.p
    num_ops = len(proc.operators)
    valid_before_eoe = (
        (seq[:first_eoe] < p)  # operands
        | ((seq[:first_eoe] >= p) & (seq[:first_eoe] < p + num_ops))  # operators
        | (seq[:first_eoe] == BOE)
        | (seq[:first_eoe] == EQL)
    )
    assert jnp.all(valid_before_eoe)


# ---------- Operator math ----------


@pytest.mark.parametrize(("a", "b"), [(0, 0), (2, 5), (6, 3), (10, 13)])
def test_apply_operator_mod_arithmetic(proc: RPNArithmeticProcess, a, b):
    p = proc.p
    # pick one of each operator token
    op_tokens = list(proc.operators.keys())  # dict keys: p, p+1, p+2
    add_tok, sub_tok, mul_tok = op_tokens

    a = jnp.array(a % p)
    b = jnp.array(b % p)

    # ADD
    out_add = proc.apply_operator(jnp.array(add_tok), a, b)
    assert out_add == (a + b) % p
    # SUB
    out_sub = proc.apply_operator(jnp.array(sub_tok), a, b)
    assert out_sub == (a - b) % p
    # MUL
    out_mul = proc.apply_operator(jnp.array(mul_tok), a, b)
    assert out_mul == (a * b) % p


# ---------- Padding behavior ----------


def test_padding_ignored_in_loss(proc: RPNArithmeticProcess, rng):
    V = proc.vocab_size
    B, T = 2, 6
    key1, key2 = jax.random.split(rng)
    logits = jax.random.normal(key1, (B, T, V))
    labels = jax.random.randint(key2, (B, T), 0, V)

    pad = proc.tokens[SpecialTokens.PAD.value]
    # make all positions PAD in sample 0 -> only sample 1 should contribute
    labels = labels.at[0, :].set(pad)

    l_all = proc.format_loss(logits, labels, label_is_index=True, pad_id=pad)

    # Compute per-sample loss with vmap, then average only over non-empty samples
    def per_item(L_i, y_i):
        return proc.format_loss(L_i[None, ...], y_i[None, ...], label_is_index=True, pad_id=pad)

    vmapped = jax.vmap(per_item)(logits, labels)  # (B,)
    # Sample 0 has no valid tokens; its contribution should be ignored
    chex.assert_trees_all_close(l_all, vmapped[1], atol=1e-6, rtol=1e-6)


# tests/test_rpn_readable.py
# These tests are intentionally small and "hands-on":
# - tiny logits you can reason about by sight
# - one or two positions per test
# - direct calculations shown inline


# ---------- helpers used only to keep expectations clear ----------


def group_negative_log_prob_for_token(logits_1d: jnp.ndarray, token: int, proc: RPNArithmeticProcess) -> float:
    """Compute the expected NLL by hand:
    1) softmax over vocab
    2) find the token's group (operand/operator/special)
    3) sum probabilities within that group
    4) return -log(sum)
    """
    probs = jax.nn.softmax(logits_1d, axis=-1)
    g = int(proc.group_id[token])
    group_prob = float(probs[proc.group_id == g].sum())
    return -math.log(max(group_prob, float(jnp.finfo(probs.dtype).tiny)))


# =========================
# = format_loss (basics)  =
# =========================


def test_format_loss_single_position_operand(proc: RPNArithmeticProcess):
    """One position, label is an operand.
    Loss should be -log(sum of probabilities over *all operands*).
    """
    V = proc.vocab_size
    logits = jnp.full((V,), 0.0)

    # Make operators a bit "hotter" so we can see the effect in the denominator
    # (operands=0.0, operators=2.0, specials=-1.0)
    p = proc.p
    num_ops = len(proc.operators)

    logits = logits.at[p : p + num_ops].set(2.0)  # 3 operator tokens
    logits = logits.at[p + num_ops :].set(-1.0)  # 4 special tokens

    # Choose label = token for value "3" (definitely an operand)
    label_token = 3

    # Model API expects leading axes, so use shape (1, V) and (1,)
    loss = proc.format_loss(logits[None, :], jnp.array([label_token]), label_is_index=True, pad_id=None)
    expected = group_negative_log_prob_for_token(logits, label_token, proc)

    assert float(loss) == pytest.approx(expected, rel=1e-6), "Loss should match manual -log(sum P(operands))."


def test_format_loss_one_hot_equals_indices(proc: RPNArithmeticProcess):
    """One position, same target, check index vs one-hot code paths are identical."""
    V = proc.vocab_size
    logits = jnp.arange(V, dtype=jnp.float32) * 0.1  # distinct but gentle scores
    y_idx = jnp.array([2])  # operand token "2"
    y_oh = jax.nn.one_hot(y_idx, V)

    loss_idx = proc.format_loss(logits[None, :], y_idx, label_is_index=True, pad_id=None)
    loss_oh = proc.format_loss(logits[None, :], y_oh, label_is_index=False, pad_id=None)

    assert float(loss_idx) == pytest.approx(float(loss_oh), rel=1e-6)


def test_format_loss_ignores_pad(proc: RPNArithmeticProcess):
    """Two positions: [real label, PAD]. Only the real one should contribute to the mean."""
    V = proc.vocab_size
    logits = jnp.linspace(-0.5, 1.5, V)  # any small vector works

    # Build a length-2 example
    real_label = 1  # operand
    pad_id = proc.tokens[SpecialTokens.PAD.value]
    labels = jnp.array([real_label, pad_id])

    loss = proc.format_loss(
        logits[None, :].repeat(2, axis=0),  # shape (2, V)
        labels,
        label_is_index=True,
        pad_id=pad_id,
    )

    # Expected: just the first position's NLL (since the second is PAD)
    expected_one = group_negative_log_prob_for_token(logits, real_label, proc)

    assert float(loss) == pytest.approx(expected_one, rel=1e-6)


# =========================
# = format_loss (readable sanity on grouping) =
# =========================


def test_format_loss_for_operator_label(proc: RPNArithmeticProcess):
    """Label is an operator. Loss should be -log(sum P(all operators))."""
    V = proc.vocab_size
    logits = jnp.zeros((V,), dtype=jnp.float32)

    # Make operands low, operators high, specials very low
    p = proc.p
    num_ops = len(proc.operators)
    logits = logits.at[:p].set(-1.0)  # operands
    logits = logits.at[p : p + num_ops].set(3.0)  # operators boosted
    logits = logits.at[p + num_ops :].set(-2.0)  # specials

    # pick '+' token
    plus_token = proc.tokens[Operators.ADD.value]
    loss = proc.format_loss(logits[None, :], jnp.array([plus_token]), label_is_index=True, pad_id=None)
    expected = group_negative_log_prob_for_token(logits, plus_token, proc)

    assert float(loss) == pytest.approx(expected, rel=1e-6)


# =========================
# = RPN behavior (simple) =
# =========================


def test_child_sub_equation_reduces_simple_pattern(proc: RPNArithmeticProcess):
    """sub_equation = [2, 3, '+', 4, '*']
    one pass should reduce [2,3,'+'] -> 5 (mod p=7) and carry the rest.
    Expected new_n = 3 and prefix [5, 4, '*'].
    """
    p = proc.p
    PLUS = proc.tokens[Operators.ADD.value]
    MUL = proc.tokens[Operators.MUL.value]

    sub = jnp.array([2, 3, PLUS, 4, MUL], dtype=jnp.int32)
    new_n, new_sub = proc.child_sub_equation(sub)

    assert int(new_n) == 3
    assert int(new_sub[0]) == (2 + 3) % p
    assert int(new_sub[1]) == 4
    assert int(new_sub[2]) == MUL


def test_random_sub_equation_is_valid_k_2(proc: RPNArithmeticProcess):
    """A randomly generated RPN with k=2 should be valid by construction."""
    key = jax.random.PRNGKey(0)
    n, sub = proc.random_sub_equation(key, k=2)
    assert int(n) == 5  # 2*k + 1 tokens
    assert proc.valid_sub_equation(sub, int(n))


def test_full_equation_has_boe_eoe_and_padding(proc: RPNArithmeticProcess):
    """Build a full equation; check the bookends and padding are sane."""
    key = jax.random.PRNGKey(42)
    n, sub = proc.random_sub_equation(key, k=2)
    seq = proc.full_equation(sub, jnp.array(n), sequence_len=64)

    toks = proc.tokens
    BOE = toks[SpecialTokens.BOE.value]
    EOE = toks[SpecialTokens.EOE.value]
    PAD = toks[SpecialTokens.PAD.value]

    # exactly one BOE and one EOE
    assert int((seq == BOE).sum()) == 1
    assert int((seq == EOE).sum()) == 1

    # everything after EOE is PAD
    eoe_pos = int(jnp.argmax(seq == EOE))
    assert jnp.all(seq[eoe_pos + 1 :] == PAD)

    # first token should be BOE
    assert int(seq[0]) == BOE
