import enum
from collections.abc import Callable, Sequence

import chex
import equinox as eqx
import jax
import jax.numpy as jnp


class Operators(enum.Enum):
    """The operators that can be used in the arithmetic process."""

    ADD = "+"
    SUB = "-"
    MUL = "*"

    # def __lt__(self, other):
    #     if not isinstance(other, Operators):
    #         return NotImplemented
    #     return self.value < other.value


class SpecialTokens(enum.Enum):
    """The special tokens that can be used in the arithmetic process."""

    VAL = "<val>"  # Value (operand)
    OPR = "<opr>"  # Operator
    EQL = "="  # Equals
    BOE = "<boe>"  # Beginning of equation
    EOE = "<eoe>"  # End of equation
    PAD = "<pad>"  # Padding


class ArithmeticProcess(eqx.Module):
    """A generative process that generates arithmetic expressions."""

    p: int
    operators: dict[int, Operators]
    tokens: dict[str, int]
    operator_functions: dict[str, Callable[[jax.Array, jax.Array], jax.Array]] = eqx.static_field()

    def __init__(self, p: int, operators: Sequence[Operators]):
        self.p = p
        self.operators = {p + i: operator for i, operator in enumerate(operators)}
        num_operators = len(self.operators)
        self.tokens = {
            **{str(i): i for i in range(p)},
            **{operator.value: token for token, operator in self.operators.items()},
            **{
                SpecialTokens.EQL.value: p + num_operators,
                SpecialTokens.BOE.value: p + num_operators + 1,
                SpecialTokens.EOE.value: p + num_operators + 2,
                SpecialTokens.PAD.value: p + num_operators + 3,
            },
        }

        # Create operator functions mapping
        operator_function_map = {
            Operators.ADD.value: lambda x, y: jnp.mod(jnp.add(x, y), self.p),
            Operators.SUB.value: lambda x, y: jnp.mod(jnp.subtract(x, y), self.p),
            Operators.MUL.value: lambda x, y: jnp.mod(jnp.multiply(x, y), self.p),
        }
        self.operator_functions = {operator.value: operator_function_map[operator.value] for operator in operators}

    def is_operand(self, token: jax.Array) -> jax.Array:
        """Check if a token is an operand."""
        return token < self.p

    def is_operator(self, token: jax.Array) -> jax.Array:
        """Check if a token is an operator."""
        return (token >= self.p) & (token < self.p + len(self.operators))

    def operator(self, token: jax.Array) -> Callable[[jax.Array, jax.Array], jax.Array]:
        """Get the operator function for a token."""
        return self.operator_functions[self.operators[int(token)].value]

    def diagram(self, tree: jax.Array) -> str:
        """Produce a diagram from a template."""
        safe_values = {
            Operators.ADD.value: "#43;",
            Operators.SUB.value: "#45;",
            Operators.MUL.value: "x",
        }
        lines = [
            "```mermaid",
            "graph TD",
            "",
        ]
        operators = []
        operands = []
        for idx, token in enumerate(tree):
            node = f"node{idx}"
            if self.is_operand(token):
                lines.append(f"{node}[{token}]")
                operands.append(idx)
            elif self.is_operator(token):
                operator = self.operators[int(token)]
                value = safe_values[operator.value]
                lines.append(f"{node}[{value}]")
                operators.append(idx)
        for idx in operators + operands:
            parent = (idx - 1) // 2
            if parent < 0:
                continue
            lines.append(f"node{parent} --> node{idx}")
        font = "color:#000,font-weight:bold"
        shape = "rx:40,ry:40"
        lines.extend(
            [
                "",
                f"classDef operand fill:#b3d9ff,stroke:#1a75ff,stroke-width:2px,{font},{shape};",
                f"classDef operator fill:#ffcccc,stroke:#cc0000,stroke-width:2px,{font},{shape};",
                "",
            ]
        )
        if operators:
            lines.append(f"class {','.join([f'node{idx}' for idx in operators])} operator;")
        if operands:
            lines.append(f"class {','.join([f'node{idx}' for idx in operands])} operand;")
        lines.append("")
        lines.append("```")
        return "\n".join(lines)

    def child(self, tree: jax.Array) -> jax.Array:
        """Produce a child template from a template."""

        def left_child(idx: int) -> int:
            return 2 * idx + 1

        def right_child(idx: int) -> int:
            return 2 * idx + 2

        output = jnp.full(tree.shape, self.tokens[SpecialTokens.PAD.value])
        stack = [0]
        while stack:
            idx = stack.pop()
            token = tree[idx]
            if self.is_operator(token):
                left_idx = left_child(idx)
                right_idx = right_child(idx)
                left_token = tree[left_idx]
                right_token = tree[right_idx]
                if self.is_operand(left_token) & self.is_operand(right_token):
                    output = output.at[idx].set(self.operator(token)(left_token, right_token))
                else:
                    output = output.at[idx].set(token)
                    stack.append(left_idx)
                    stack.append(right_idx)
            elif self.is_operand(token):
                output = output.at[idx].set(token)

        return output

    def generate(self, state: jax.Array, key: chex.PRNGKey, sequence_len: int) -> jax.Array:
        """Produce an equation from a format."""
        ...
