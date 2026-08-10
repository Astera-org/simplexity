"""Generative process implementations, pending deprecation.

Generative processes are moving out of simplexity and into consumers' own projects, sourced from
[generators](https://github.com/ealt/generators), which distributes process modules by
copying rather than by import. Run management instantiates a vendored process on equal terms with
one implemented here: see `simplexity.run_management.protocols.GenerativeProcessProtocol` for the
contract, and `docs/generators_migration.md` for the migration.

Nothing here is removed or altered yet. Full deprecation waits until generators reaches feature
parity with this package; `docs/design/generators_instantiation.md` tracks the outstanding gaps.
"""

import warnings

warnings.warn(
    "simplexity.generative_processes is pending deprecation in favour of processes vendored from "
    "generators (https://github.com/ealt/generators). It remains fully supported until "
    "generators reaches feature parity; see docs/generators_migration.md.",
    PendingDeprecationWarning,
    stacklevel=2,
)
