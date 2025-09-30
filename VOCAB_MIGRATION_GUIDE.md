# Migration Guide: Automatic Vocab Size and Special Token Computation

## Overview

This PR introduces automatic computation of `vocab_size`, `bos_token`, and `eos_token` values for generative processes. This eliminates redundancy and reduces configuration errors.

## What Changed

### Before

Generative process configs required manual specification of:
- `vocab_size`: Total vocabulary including special tokens (error-prone!)
- `bos_token`: Numeric token ID for beginning-of-sequence (had to match vocab_size)
- `eos_token`: Numeric token ID for end-of-sequence

```yaml
# OLD FORMAT - DON'T USE
name: mess3_085
vocab_size: 4              # Manual: 3 (mess3) + 1 (BOS token)
bos_token: 3               # Manual: had to know this is vocab_size - 1
eos_token: null
instance:
  _target_: simplexity.generative_processes.builder.build_hidden_markov_model
  process_name: mess3
  x: 0.05
  a: 0.85
```

### After

Configs now use boolean flags, and all numeric values are computed automatically:

```yaml
# NEW FORMAT - USE THIS
name: mess3_085
use_bos: true              # Clear intent: "use beginning-of-sequence token"
use_eos: false             # Clear intent: "don't use end-of-sequence token"
instance:
  _target_: simplexity.generative_processes.builder.build_hidden_markov_model
  process_name: mess3
  x: 0.05
  a: 0.85
# vocab_size, bos_token, eos_token all computed automatically!
```

## How It Works

1. **Instantiate the generator**: The generator object knows its base `vocab_size` from the transition matrices
2. **Compute special token IDs**:
   - If `use_bos=true`: `bos_token = generator.vocab_size`
   - If `use_eos=true`: `eos_token = generator.vocab_size + (1 if use_bos else 0)`
3. **Compute total vocab**: `vocab_size = generator.vocab_size + num_special_tokens`
4. **Propagate to model**: Model configs use `d_vocab: ${training_data_generator.vocab_size}` (interpolation)

## Migration Steps

### For Simplexity Repo Configs

✅ **Already done!** All configs in `simplexity/configs/generative_process/` have been updated.

### For Simplex-Research Repo Configs

Update your generative process configs:

```bash
cd /path/to/simplex-research
```

For each `configs/generative_process/*.yaml` file:

1. **Remove** the `vocab_size` line
2. **Replace** `bos_token: N` with `use_bos: true` (or `use_bos: false` if it was `null`)
3. **Replace** `eos_token: N` with `use_eos: true` (or `use_eos: false` if it was `null`)

### Example: basic_mess3

**File**: `/Users/adamimos/Documents/GitHub/simplex-research/basic_mess3/configs/generative_process/mess3_085.yaml`

```yaml
# BEFORE
name: mess3_085
vocab_size: 4
bos_token: 3
eos_token:

# AFTER
name: mess3_085
use_bos: true
use_eos: false
```

## Benefits

1. **Single source of truth**: Base vocab comes from the generator code, not config
2. **Impossible to mismatch**: BOS/EOS tokens always get the right IDs
3. **Clearer intent**: "Do I want a BOS token?" vs "What ID should BOS be?"
4. **Automatic d_vocab**: Model vocab size always includes special tokens correctly
5. **Less error-prone**: No manual arithmetic (3 + 1 = 4)

## Validation

After migrating your configs, run:

```bash
uv run python your_train_script.py --config-name your_config
```

The system will automatically:
- Read `use_bos` and `use_eos` flags
- Compute the correct token IDs
- Set the total vocab_size
- Pass everything to your model

## Edge Cases

**Q: What if I need custom token IDs?**
A: The computed values can still be overridden manually in code if absolutely necessary.

**Q: Does this work with composite/nonergodic processes?**
A: Yes! The `vocab_size` property works for all `GenerativeProcess` types.

**Q: What if I don't use special tokens?**
A: Just set `use_bos: false` and `use_eos: false`. The vocab_size will equal the base generator vocab.

## Examples

### No special tokens (binary process)
```yaml
name: zero_one_random
use_bos: false
use_eos: false
# Computes: vocab_size=2, bos_token=null, eos_token=null
```

### BOS only (standard next-token prediction)
```yaml
name: mess3
use_bos: true
use_eos: false
# Computes: vocab_size=4, bos_token=3, eos_token=null
```

### Both tokens (sequence-to-sequence)
```yaml
name: days_of_week
use_bos: true
use_eos: true
# Computes: vocab_size=13, bos_token=11, eos_token=12
```

## Need Help?

- Check existing configs in `simplexity/configs/generative_process/` for examples
- Run tests: `uv run pytest tests/test_vocab_computation.py -v`
- See the implementation in `simplexity/run.py::compute_vocab_and_special_tokens()`
