import enum
from abc import ABC, abstractmethod
from collections.abc import Callable, Sequence

import chex
import equinox as eqx
import jax
import jax.numpy as jnp

from simplexity.data_structures.stack import Stack


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


class ArithmeticProcess(eqx.Module, ABC):
    """A generative process that generates arithmetic expressions.

    This class is abstract and cannot be instantiated directly.
    """

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

    @abstractmethod
    def random_sub_equation(self, key: chex.PRNGKey, k: int) -> tuple[int, jax.Array]:
        """Produce a random sub-equation."""
        ...

    @abstractmethod
    def child_sub_equation(self, sub_equation: jax.Array) -> tuple[int, jax.Array]:
        """Produce a child sub-equation."""
        ...

    def full_equation(self, sub_equation: jax.Array, n: int, sequence_len: int) -> jax.Array:
        """Produce a random RPN sequence."""
        equation = jnp.full(sequence_len, self.tokens[SpecialTokens.PAD.value])
        equation = equation.at[0].set(self.tokens[SpecialTokens.BOE.value])
        i = 1
        equation = equation.at[i : i + n].set(sub_equation)
        i += n
        while n > 1:
            equation = equation.at[i].set(self.tokens[SpecialTokens.EQL.value])
            i += 1
            n, sub_equation = self.child_sub_equation(sub_equation)
            equation = equation.at[i : i + n].set(sub_equation[:n])
            i += n
        equation = equation.at[i].set(self.tokens[SpecialTokens.EOE.value])
        return equation

    def random_equation(self, key: chex.PRNGKey, k: int, sequence_len: int) -> jax.Array:
        """Produce a random RPN sequence."""
        n, sub_equation = self.random_sub_equation(key, k)
        return self.full_equation(sub_equation, n, sequence_len)


class BinaryTreeArithmeticProcess(ArithmeticProcess):
    """A generative process that generates arithmetic expressions in RPN format."""

    def __init__(self, p: int, operators: Sequence[Operators]):
        super().__init__(p, operators)

    @staticmethod
    def left_child(idx: int) -> int:
        """Get the left child of an index."""
        return 2 * idx + 1

    @staticmethod
    def right_child(idx: int) -> int:
        """Get the right child of an index."""
        return 2 * idx + 2

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

    def random_sub_equation(self, key: chex.PRNGKey, k: int) -> tuple[int, jax.Array]:
        """Produce a random RPN sub-equation."""
        n = 2 ** (k + 1) - 1
        sub_equation = jnp.full(n, self.tokens[SpecialTokens.PAD.value])
        operand_key, operator_key, key = jax.random.split(key, 3)
        operands = jax.random.randint(operand_key, (k + 1,), 0, self.p)
        operators = jax.random.randint(operator_key, (k,), self.p, self.p + len(self.operators))
        operator_idxs = jnp.zeros(n, dtype=jnp.bool_)
        leaf_idxs = jnp.zeros(n, dtype=jnp.bool_).at[0].set(True)
        while jnp.sum(operator_idxs) < k:
            key, leaf_key = jax.random.split(key)
            leaf_idx = int(jax.random.choice(leaf_key, jnp.where(leaf_idxs)[0]))
            operator_idxs = operator_idxs.at[leaf_idx].set(True)
            leaf_idxs = leaf_idxs.at[leaf_idx].set(False)
            leaf_idxs = leaf_idxs.at[self.left_child(leaf_idx)].set(True)
            leaf_idxs = leaf_idxs.at[self.right_child(leaf_idx)].set(True)
        sub_equation = sub_equation.at[operator_idxs].set(operators)
        sub_equation = sub_equation.at[leaf_idxs].set(operands)
        max_level = int(jnp.floor(jnp.log2(jnp.max(leaf_idxs) + 1)))
        n = 2 ** (max_level + 1) - 1
        return n, sub_equation

    def child_sub_equation(self, sub_equation: jax.Array) -> tuple[int, jax.Array]:
        """Produce a child RPN sub-equation."""
        n = sub_equation.shape[0]
        output = jnp.full(n, self.tokens[SpecialTokens.PAD.value])
        stack = Stack(max_size=n, default_element=jnp.array(0, dtype=jnp.int32))
        stack = stack.push(jnp.array(0, dtype=jnp.int32))
        max_idx = 0
        while not stack.is_empty:
            stack, idx = stack.pop()
            max_idx = max(max_idx, idx)
            token = sub_equation[idx]
            if self.is_operator(token):
                left_idx = self.left_child(int(idx))
                right_idx = self.right_child(int(idx))
                left_token = sub_equation[left_idx]
                right_token = sub_equation[right_idx]
                if self.is_operand(left_token) & self.is_operand(right_token):
                    output = output.at[idx].set(self.operator(token)(left_token, right_token))
                else:
                    output = output.at[idx].set(token)
                    stack = stack.push(jnp.array(left_idx, dtype=jnp.int32))
                    stack = stack.push(jnp.array(right_idx, dtype=jnp.int32))
            elif self.is_operand(token):
                output = output.at[idx].set(token)

        max_level = int(jnp.floor(jnp.log2(max_idx + 1)))
        n = 2 ** (max_level + 1) - 1
        return n, output
