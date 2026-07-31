# Vendored generators modules

Verbatim copies of [generators](https://github.com/ealt/generators) at commit `b1242ea`,
apart from rewriting the module-level import to this package's path:

| File | Upstream path |
| --- | --- |
| `ghmm_process.py` | `generators/ghmm/process.py` |
| `utils.py` | `generators/utils.py` |

They are checked in, rather than imported, because that is generators' distribution model — the
copy is the point. It also keeps this test suite independent of generators being installed.

`../adapter.py` binds these module-level functions to
`simplexity.run_management.protocols.GenerativeProcess`, and
`../../end_to_end/test_vendored_process_training.py` trains against it under `@managed_run()`.

To pick up upstream improvements, diff against the current upstream module and re-copy. Do not add
simplexity-specific behaviour here; that belongs in the adapter.
