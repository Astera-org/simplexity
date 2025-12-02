"""Conditional structures for factored generative processes.

This module provides various conditional dependency structures that define
how factors in a factored generative process depend on each other.

Available structures:
- Independent: No conditional dependencies between factors
- SequentialConditional: One-way chain dependencies (factor i depends on i-1)
- FullyConditional: Mutual dependencies between all factors
- ConditionalTransitions: Hybrid structure (independent/sequential emissions, mutual transitions)
"""

from simplexity.generative_processes.structures.conditional_transitions import (
    ConditionalTransitions,
)
from simplexity.generative_processes.structures.fully_conditional import (
    FullyConditional,
)
from simplexity.generative_processes.structures.protocol import (
    ConditionalContext,
    ConditionalStructure,
)
from simplexity.generative_processes.structures.sequential_conditional import (
    SequentialConditional,
)

__all__ = [
    "ConditionalContext",
    "ConditionalStructure",
    "ConditionalTransitions",
    "FullyConditional",
    "SequentialConditional",
]
