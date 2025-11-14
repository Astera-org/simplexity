from collections.abc import Sequence

import chex
import equinox as eqx
import jax
import jax.numpy as jnp



class EmbeddingFn(eqx.Module):
    embedding: eqx.nn.Embedding
    def __init__(self, vocab_size: int, embedding_size: int, key: chex.PRNGKey):
        self.embedding = eqx.nn.Embedding(vocab_size, embedding_size, key=key)

    def __call__(self, xs: jax.Array) -> jax.Array:
        # xs can be [] (scalar token id) or [T]
        if xs.ndim == 0:          # single token id
            return self.embedding(xs)
        elif xs.ndim == 1:        # [T]
            return eqx.filter_vmap(self.embedding)(xs)
        else:
            raise ValueError("EmbeddingFn expects scalar token id or [T] token ids.")


class LinearFn(eqx.Module):
    linear: eqx.nn.Linear
    def __init__(self, input_size: int, out_size: int, key: chex.PRNGKey):
        self.linear = eqx.nn.Linear(input_size, out_size, key=key)

    def __call__(self, xs: jax.Array) -> jax.Array:
        # xs can be [d] or [T, d]
        if xs.ndim == 1:
            return self.linear(xs)
        elif xs.ndim == 2:
            return eqx.filter_vmap(self.linear)(xs)
        else:
            raise ValueError("LinearFn expects [d] or [T, d].")
    

class RMSNorm(eqx.Module):
    """RMS Normalization layer."""

    w: jax.Array
    eps: float = 1e-5

    def __init__(self, d_model): 
        self.w = jnp.ones((d_model,))

    def __call__(self, x): 
        rms = jnp.sqrt(jnp.mean(x * x, axis=-1, keepdims=True) + self.eps)
        return self.w * (x / rms)

def depthwise_conv_time(conv: eqx.nn.Conv1d, u: jax.Array) -> jax.Array:
    """Causal depthwise conv over time, length-preserving."""
    # u: [T, d_inner] -> [1, d_inner, T]
    u_ = jnp.transpose(u[None, :, :], (0, 2, 1))
    k = conv.kernel_size[0]
    # causal left pad so output length == T
    u_ = jnp.pad(u_, ((0, 0), (0, 0), (k - 1, 0)))
    conv_out = conv(u_)                     # [1, d_inner, T]
    return jnp.transpose(conv_out, (0, 2, 1)).squeeze(0)  # [T, d_inner]

def depthwise_conv_step(conv: eqx.nn.Conv1d, u_t: jax.Array, conv_buf: jax.Array) -> tuple[jax.Array, jax.Array]:
    """Apply depthwise convolution step for streaming."""
    # u_t: [d_inner], conv_buf: [d_conv-1, d_inner]
    d_conv = conv.kernel_size[0]
    d_inner = u_t.shape[-1]

    # Prepare input buffer: [d_conv, d_inner]
    conv_input = jnp.concatenate([conv_buf, u_t[None, :]], axis=0)  # [d_conv, d_inner]
    conv_input = conv_input[None, :, :].transpose((0, 2, 1))  # [1, d_inner, d_conv]

    # Apply convolution
    conv_out = conv(conv_input)  # [1, d_inner, 1]
    conv_out = conv_out.squeeze((0, 2))  # [d_inner]

    # Update buffer
    new_conv_buf = jnp.concatenate([conv_buf[1:, :], u_t[None, :]], axis=0)  # [d_conv-1, d_inner]

    return conv_out, new_conv_buf

def selective_scan_full(
    u: jax.Array,        # [T, d_inner]
    delta: jax.Array,    # [T, d_inner]
    B: jax.Array,        # [T, d_inner, N]
    C: jax.Array,        # [T, d_inner, N]
    A: jax.Array,        # [N] (negative diag, e.g. -softplus(theta))
    D: jax.Array | None = None,  # optional skip coef per channel: [d_inner]
) -> jax.Array:
    T, d_inner = u.shape
    N = A.shape[0]

    def step(ssm_state, inputs):
        u_t, delta_t, B_t, C_t = inputs               # [d_inner], [d_inner], [d_inner, N], [d_inner, N]
        a = jnp.exp(delta_t[:, None] * A[None, :])    # [d_inner, N]
        ssm_state = a * ssm_state + B_t * u_t[:, None]
        y_t = jnp.einsum("in,in->i", ssm_state, C_t)  # [d_inner]
        if D is not None:
            y_t = y_t + D * u_t
        return ssm_state, y_t

    init = jnp.zeros((d_inner, N), dtype=u.dtype)
    _, ys = jax.lax.scan(step, init, (u, delta, B, C))
    return ys  # [T, d_inner]


class MambaBlock(eqx.Module):
    d_model: int = eqx.field(static=True)
    d_inner: int = eqx.field(static=True)
    N: int = eqx.field(static=True)
    d_conv: int = eqx.field(static=True)

    in_proj: eqx.nn.Linear          # d_model -> 2*d_inner
    delta_proj: eqx.nn.Linear       # d_inner -> d_inner
    B_proj: eqx.nn.Linear           # d_inner -> N
    C_proj: eqx.nn.Linear           # d_inner -> N
    out_proj: eqx.nn.Linear         # d_inner -> d_model
    A_log: jax.Array                # shape [N], negative via -softplus
    D: jax.Array                    # shape [d_inner], skip connections
    conv: eqx.nn.Conv1d             # depthwise, groups=d_inner
    norm: RMSNorm

    def __init__(self, d_model, d_inner, N, d_conv, key):
        k1,k2,k3,k4,k5,kc = jax.random.split(key, 6)
        self.in_proj   = eqx.nn.Linear(d_model, 2*d_inner, key=k1)
        self.delta_proj= eqx.nn.Linear(d_inner, d_inner, key=k2)
        self.B_proj = eqx.nn.Linear(d_inner, d_inner * N, key=k3)
        self.C_proj = eqx.nn.Linear(d_inner, d_inner * N, key=k4)
        self.out_proj  = eqx.nn.Linear(d_inner, d_model, key=k5)
        self.A_log     = jnp.zeros((N,))       # A = -softplus(A_log)
        self.D          = jnp.zeros((d_inner,), dtype=jnp.float32)   # starts “off”
        self.conv = eqx.nn.Conv1d(
            d_inner, d_inner, kernel_size=d_conv,
            stride=1, padding=0, groups=d_inner, use_bias=True, key=kc
        )
        self.norm = RMSNorm(d_model)

    def init_state(self, *, dtype=jnp.float32):
        """Per-layer streaming state: (ssm_state, conv_buf)."""
        ssm_state = jnp.zeros((self.d_inner, self.N), dtype=dtype)
        conv_buf  = jnp.zeros((self.d_conv - 1, self.d_inner), dtype=dtype)
        return (ssm_state, conv_buf)

    def __call__(self, xs: jax.Array) -> jax.Array:
        x = self.norm(xs)                                # [T, d_model]
        u, z = jnp.split(self.in_proj(x), 2, axis=-1)    # [T, d_inner] each
        u = depthwise_conv_time(self.conv, u)            # [T, d_inner]
        delta = jax.nn.softplus(self.delta_proj(u))      # [T, d_inner]

        T = xs.shape[0]
        B = self.B_proj(u).reshape(T, self.d_inner, self.N)  # [T, d_inner, N]
        C = self.C_proj(u).reshape(T, self.d_inner, self.N)  # [T, d_inner, N]
        y = selective_scan_full(
            u, delta, B, C, -jax.nn.softplus(self.A_log), D=self.D
        )                                                # [T, d_inner]

        h = jax.nn.sigmoid(z) * y
        return xs + self.out_proj(h)

    # streaming step (inference)
    def step(self, x_t, state):
        # state shapes: ssm_state [d_inner, N], conv_buf [d_conv-1, d_inner]
        ssm_state, conv_buf = state
        x = self.norm(x_t)                               # [d_model]
        u, z = jnp.split(self.in_proj(x), 2, axis=-1)    # [d_inner], [d_inner]
        u, conv_buf = depthwise_conv_step(self.conv, u, conv_buf)
        delta = jax.nn.softplus(self.delta_proj(u))      # [d_inner]
        B = self.B_proj(u).reshape(self.d_inner, self.N) # [d_inner, N]
        C = self.C_proj(u).reshape(self.d_inner, self.N) # [d_inner, N]
        A = -jax.nn.softplus(self.A_log)                 # [N]

        a = jnp.exp(delta[:, None] * A[None, :])         # [d_inner, N]
        ssm_state = a * ssm_state + B * u[:, None]       # [d_inner, N]
        y = jnp.einsum('in,in->i', ssm_state, C)         # [d_inner]
        y = y + self.D * u                               # <— D skip in streaming too

        h = jax.nn.sigmoid(z) * y
        out = x_t + self.out_proj(h)                     # residual
        return out, (ssm_state, conv_buf)


class MambaSSM(eqx.Module):
    """A Mamba-based SSM model."""

    vocab_size: int = eqx.field(static=True)
    layers: eqx.nn.Sequential

    def init_states(self, *, dtype=jnp.float32):
        """Return a tuple of per-layer states aligned with self.layers.layers."""
        states = []
        for layer in self.layers.layers:
            if isinstance(layer, MambaBlock):
                states.append(layer.init_state(dtype=dtype))
            else:
                states.append(None)
        return tuple(states)

    def __init__(self, vocab_size: int, embedding_size: int, hidden_sizes: Sequence[int], d_conv: int, N: int, key: chex.PRNGKey):
        self.vocab_size = vocab_size

        num_gru_layers = len(hidden_sizes)
        embedding_key, linear_key, *cell_keys = jax.random.split(key, num_gru_layers + 2)

        # Reference RNN:
        # layers = []
        # layers.append(EmbeddingFn(vocab_size, embedding_size, embedding_key))
        # input_size = embedding_size
        # for hidden_size, cell_key in zip(hidden_sizes, cell_keys, strict=True):
        #     gru_fn = GRUFn(input_size, hidden_size, cell_key)
        #     gru_layer = eqx.nn.Lambda(gru_fn)
        #     layers.append(gru_layer)
        #     input_size = hidden_size
        # linear_fn = LinearFn(input_size, vocab_size, linear_key)
        # linear_layer = eqx.nn.Lambda(linear_fn)
        # layers.append(linear_layer)
        # self.layers = eqx.nn.Sequential(layers)

        layers = []
        layers.append(EmbeddingFn(vocab_size, embedding_size, embedding_key))
        input_size = embedding_size
        for hidden_size, cell_key in zip(hidden_sizes, cell_keys, strict=True):
            mamba_block = MambaBlock(input_size, hidden_size, N, d_conv, cell_key)
            layers.append(mamba_block)
            input_size = hidden_size
        linear_fn = LinearFn(input_size, vocab_size, linear_key)
        linear_layer = eqx.nn.Lambda(linear_fn)
        layers.append(linear_layer)
        self.layers = eqx.nn.Sequential(layers)

    
    def stream_step(self, token_t: jax.Array, states):
        """
        Stream **one token** through the stack.
        token_t: scalar int token id  (shape [])
        states: tuple from init_states()
        returns: (logits_t [vocab], new_states)
        """
        new_states = []
        x = token_t
        # Embedding
        emb = self.layers.layers[0]
        x = emb(x)  # [d_model]

        # Blocks
        si = 1
        for layer in self.layers.layers[1:-1]:
            if isinstance(layer, MambaBlock):
                x, st = layer.step(x, states[si])
                new_states.append(st)
            else:
                # Shouldn't happen in your stack, but just in case
                x = layer(x)
                new_states.append(None)
            si += 1

        # Head
        head = self.layers.layers[-1]  # LinearFn wrapped in Lambda or the module itself
        x = head(x)  # [vocab]
        # Reinsert Nones for embedding/head to keep alignment with self.layers.layers
        new_states = (None, *tuple(new_states), None)
        return x, new_states

    def stream(self, tokens: jax.Array, states=None):
        """
        Stream a **sequence of tokens** (shape [T]) and return per-step logits and final states.
        """
        if states is None:
            states = self.init_states(dtype=tokens.dtype if tokens.dtype == jnp.float32 else jnp.float32)

        def _one(carry, tok_t):
            st = carry
            logits_t, st = self.stream_step(tok_t, st)
            return st, logits_t

        states, logits = jax.lax.scan(_one, states, tokens)  # logits: [T, vocab]
        return logits, states
            

    def __call__(self, xs: jax.Array) -> jax.Array:
        """Forward pass of the GRU RNN."""
        return self.layers(xs)


def build_mamba_ssm(vocab_size: int, embedding_size: int, num_layers: int, hidden_size: int, d_conv: int, N: int, seed: int) -> MambaSSM:
    """Build a Mamba SSM model."""
    hidden_sizes = [hidden_size] * num_layers
    key = jax.random.PRNGKey(seed)
    return MambaSSM(vocab_size, embedding_size, hidden_sizes, d_conv, N, key=key)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--vocab_size", type=int, default=200)
    parser.add_argument("--d_model", type=int, default=64)
    parser.add_argument("--num_layers", type=int, default=2)
    parser.add_argument("--d_inner", type=int, default=64)          # hidden_size per layer
    parser.add_argument("--ssm_order", type=int, default=16)        # N
    parser.add_argument("--d_conv", type=int, default=4)
    parser.add_argument("--seq_len", type=int, default=32)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--rollout_steps", type=int, default=16)
    args = parser.parse_args()

    key = jax.random.PRNGKey(args.seed)

    # Build model
    model = build_mamba_ssm(
        vocab_size=args.vocab_size,
        embedding_size=args.d_model,
        num_layers=args.num_layers,
        hidden_size=args.d_inner,
        d_conv=args.d_conv,
        N=args.ssm_order,
        seed=args.seed,
    )

    # Random toy data: a single sequence of token ids [T]
    key, tok_key = jax.random.split(key)
    tokens = jax.random.randint(tok_key, (args.seq_len,), 0, args.vocab_size, dtype=jnp.int32)

    # 1) Full-sequence forward
    logits_full = model(tokens)                  # [T, vocab]

    # 2) Streaming forward (teacher-forced), should match full
    logits_stream, _ = model.stream(tokens)      # [T, vocab]

    # Compare
    diff = logits_full - logits_stream
    mse = jnp.mean(diff * diff)
    max_abs = jnp.max(jnp.abs(diff))
    print(f"[check] logits parity: mse={float(mse):.6e}  max|Δ|={float(max_abs):.6e}")

    # 3) Tiny greedy rollout using stream_step
    def greedy_rollout(model, start_token: int, steps: int):
        states = model.init_states()
        tok = jnp.asarray(start_token, dtype=jnp.int32)
        seq = [int(tok)]
        for _ in range(steps):
            logits_t, states = model.stream_step(tok, states)  # [vocab]
            tok = jnp.argmax(logits_t).astype(jnp.int32)
            seq.append(int(tok))
        return seq

    start_token = int(tokens[0])
    out_seq = greedy_rollout(model, start_token=start_token, steps=args.rollout_steps)
    print(f"[gen] start={start_token}  rollout({args.rollout_steps}) -> {out_seq}")

    # Optional: assert tight parity
    assert float(max_abs) < 1e-4, "Streaming and full-sequence outputs diverged more than expected."
    print("[ok] streaming matches full-sequence within tolerance.")