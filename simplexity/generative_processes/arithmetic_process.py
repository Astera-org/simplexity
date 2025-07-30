import enum
from abc import ABC, abstractmethod
from collections.abc import Callable, Sequence

import chex
import equinox as eqx
import jax
import jax.numpy as jnp

from simplexity.data_structures.stack import Stack


class Operators(enum.Enum):
    """Mathematical operators that can be used in arithmetic expressions."""

    ADD = "+"
    SUB = "-"
    MUL = "*"


class SpecialTokens(enum.Enum):
    """Special tokens used to structure arithmetic expressions and sequences."""

    VAL = "<val>"  # Value (operand)
    OPR = "<opr>"  # Operator
    EQL = "="  # Equals
    BOE = "<boe>"  # Beginning of equation
    EOE = "<eoe>"  # End of equation
    PAD = "<pad>"  # Padding


class ArithmeticProcess(eqx.Module, ABC):
    """Abstract base class for generative processes that create arithmetic expressions.

    This class provides the foundation for generating arithmetic expressions
    in different formats (e.g., binary tree, RPN). It handles token management,
    operator functions, and provides abstract methods for subclasses to implement
    specific generation strategies.

    This class is abstract and cannot be instantiated directly.
    """

    p: int
    operators: dict[int, Operators]
    tokens: dict[str, int]
    operator_functions: dict[str, Callable[[jax.Array, jax.Array], jax.Array]] = eqx.static_field()

    def __init__(self, p: int, operators: Sequence[Operators]):
        """Initialize the arithmetic process.

        Args:
            p: The modulus for arithmetic operations (values are in range [0, p-1])
            operators: Sequence of operators to use in expressions
        """
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
        """Check if a token represents an operand (numeric value).

        Args:
            token: Token to check

        Returns:
            Boolean array indicating if the token is an operand
        """
        return token < self.p

    def is_operand_or_operator(self, token: jax.Array) -> jax.Array:
        """Check if a token represents an operand or operator.

        Args:
            token: Token to check

        Returns:
            Boolean array indicating if the token is an operand or operator
        """
        return token < self.p + len(self.operators)

    def is_operator(self, token: jax.Array) -> jax.Array:
        """Check if a token represents an operator.

        Args:
            token: Token to check

        Returns:
            Boolean array indicating if the token is an operator
        """
        return self.is_operand_or_operator(token) & ~self.is_operand(token)

    def operator(self, token: jax.Array) -> Callable[[jax.Array, jax.Array], jax.Array]:
        """Get the operator function corresponding to a token.

        Args:
            token: Token representing an operator

        Returns:
            Function that performs the corresponding arithmetic operation
        """
        return self.operator_functions[self.operators[int(token)].value]

    @abstractmethod
    def random_sub_equation(self, key: chex.PRNGKey, k: int) -> tuple[int, jax.Array]:
        """Generate a random sub-equation.

        Args:
            key: JAX PRNG key for random number generation
            k: Complexity parameter (typically represents tree depth or expression size)

        Returns:
            Tuple of (size, sub_equation) where size is the number of meaningful tokens
            and sub_equation is the array representation of the expression
        """
        ...

    @abstractmethod
    def child_sub_equation(self, sub_equation: jax.Array) -> tuple[int, jax.Array]:
        """Generate a child sub-equation by evaluating the given sub-equation.

        This method typically performs one step of evaluation, reducing the expression
        by computing operations where both operands are known.

        Args:
            sub_equation: The parent sub-equation to evaluate

        Returns:
            Tuple of (size, evaluated_sub_equation) where size is the number of
            meaningful tokens in the evaluated result
        """
        ...

    def full_equation(self, sub_equation: jax.Array, n: int, sequence_len: int) -> jax.Array:
        """Generate a complete equation sequence from a sub-equation.

        Creates a full equation by repeatedly evaluating the sub-equation until
        it reduces to a single value, adding equals signs between each evaluation step.

        Args:
            sub_equation: Initial sub-equation to start with
            n: Number of meaningful tokens in the sub-equation
            sequence_len: Total length of the output sequence

        Returns:
            Complete equation sequence with beginning/end markers and equals signs
        """
        equation = jnp.full(sequence_len, self.tokens[SpecialTokens.PAD.value])
        equation = equation.at[0].set(self.tokens[SpecialTokens.BOE.value])
        i = 1
        equation = equation.at[i : i + n].set(sub_equation[:n])
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
        """Generate a complete random arithmetic equation.

        Args:
            key: JAX PRNG key for random number generation
            k: Complexity parameter for the sub-equation
            sequence_len: Total length of the output sequence

        Returns:
            Complete equation sequence with random arithmetic expression
        """
        n, sub_equation = self.random_sub_equation(key, k)
        return self.full_equation(sub_equation, n, sequence_len)

    @abstractmethod
    def valid_sub_equation(self, sub_equation: jax.Array, n: int) -> jax.Array:
        """Check if a sub-equation is valid according to the implementation's rules.

        Args:
            sub_equation: The sub-equation to validate
            n: Number of meaningful tokens in the sub-equation

        Returns:
            Boolean array indicating if the sub-equation is valid
        """
        ...


class BinaryTreeArithmeticProcess(ArithmeticProcess):
    """Generative process that creates arithmetic expressions in binary tree format.

    This implementation represents arithmetic expressions as complete binary trees
    stored in array format, where each node at index i has children at indices
    2*i+1 and 2*i+2. Operators are placed at internal nodes and operands at leaves.
    """

    def __init__(self, p: int, operators: Sequence[Operators]):
        """Initialize the binary tree arithmetic process.

        Args:
            p: The modulus for arithmetic operations
            operators: Sequence of operators to use in expressions
        """
        super().__init__(p, operators)

    @staticmethod
    def parent(idx: int) -> int:
        """Get the parent index of a given node index.

        Args:
            idx: Node index

        Returns:
            Parent node index
        """
        return (idx - 1) // 2

    @staticmethod
    def left_child(idx: int) -> int:
        """Get the left child index of a given node index.

        Args:
            idx: Node index

        Returns:
            Left child node index
        """
        return 2 * idx + 1

    @staticmethod
    def right_child(idx: int) -> int:
        """Get the right child index of a given node index.

        Args:
            idx: Node index

        Returns:
            Right child node index
        """
        return 2 * idx + 2

    def diagram(self, tree: jax.Array) -> str:
        """Generate a Mermaid diagram representation of the binary tree.

        Creates a visual representation of the arithmetic expression tree
        suitable for rendering in Markdown with Mermaid support.

        Args:
            tree: Array representation of the binary tree

        Returns:
            Mermaid diagram code as a string
        """
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
        """Generate a random binary tree sub-equation.

        Creates a complete binary tree with k operators randomly placed at internal
        nodes and k+1 operands at leaf nodes. The tree is stored in array format
        where each node at index i has children at indices 2*i+1 and 2*i+2.

        Args:
            key: JAX PRNG key for random number generation
            k: Number of operators (determines tree depth and complexity)

        Returns:
            Tuple of (size, sub_equation) where size is the actual tree size
            and sub_equation contains the tree in array format
        """
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
        # Find the maximum index that contains a non-PAD token
        max_idx = jnp.max(jnp.where(sub_equation != self.tokens[SpecialTokens.PAD.value])[0])
        max_level = int(jnp.floor(jnp.log2(max_idx + 1)))
        n = 2 ** (max_level + 1) - 1
        return n, sub_equation

    def child_sub_equation(self, sub_equation: jax.Array) -> tuple[int, jax.Array]:
        """Generate a child sub-equation by evaluating the binary tree.

        Performs one step of evaluation by computing operations where both
        operands are known values. Uses a stack-based traversal to process
        the tree in depth-first order.

        Args:
            sub_equation: The parent binary tree sub-equation to evaluate

        Returns:
            Tuple of (size, evaluated_sub_equation) where size is the actual
            tree size after evaluation and evaluated_sub_equation contains
            the partially evaluated tree
        """
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

    def valid_sub_equation(self, sub_equation: jax.Array, n: int) -> jax.Array:
        """Check if a binary tree sub-equation is valid.

        Validates that the sub-equation follows binary tree structure rules:
        - Root must be an operand or operator
        - Internal nodes must be operators
        - Leaf nodes must be operands
        - Parent-child relationships must be consistent
        - Padding tokens must be in unused positions

        Args:
            sub_equation: The sub-equation to validate
            n: Number of meaningful tokens in the sub-equation

        Returns:
            Boolean array indicating if the sub-equation is valid
        """
        if n < 1:
            return jnp.array(False)
        if sub_equation.shape[0] < n:
            return jnp.array(False)
        if not self.is_operand_or_operator(sub_equation[0]):
            return jnp.array(False)
        if not jnp.all(sub_equation[n:] == self.tokens[SpecialTokens.PAD.value]):
            return jnp.array(False)
        # valid parents
        for idx in range(1, sub_equation.shape[0]):
            if self.is_operand_or_operator(sub_equation[idx]):
                if not self.is_operator(sub_equation[self.parent(idx)]):
                    return jnp.array(False)
            elif sub_equation[idx] != self.tokens[SpecialTokens.PAD.value]:
                return jnp.array(False)
        # valid children
        for idx in range(self.parent(n)):
            token = sub_equation[idx]
            left_idx = self.left_child(int(idx))
            right_idx = self.right_child(int(idx))
            left_token = sub_equation[left_idx]
            right_token = sub_equation[right_idx]
            if self.is_operator(token):
                if not (self.is_operand_or_operator(left_token) and self.is_operand_or_operator(right_token)):
                    return jnp.array(False)
            else:
                if not (
                    left_token == self.tokens[SpecialTokens.PAD.value]
                    and right_token == self.tokens[SpecialTokens.PAD.value]
                ):
                    return jnp.array(False)
        return jnp.array(True)
