"""
Tests for RPNArithmeticProcess - Reverse Polish Notation arithmetic expressions.

This module tests the RPNArithmeticProcess class which generates arithmetic expressions
in Reverse Polish Notation (postfix notation). In RPN, operators follow their operands,
making it suitable for stack-based evaluation.

Example: (2 + 3) * 4 becomes 2 3 + 4 * in RPN
"""

import equinox as eqx
import jax
import jax.numpy as jnp
import pytest

from simplexity.generative_processes.arithmetic_process import Operators, RPNArithmeticProcess

TOKENS = {
    "0": 0,
    "1": 1,
    "2": 2,
    "3": 3,
    "4": 4,
    "+": 5,
    "-": 6,
    "=": 7,
    "<boe>": 8,
    "<eoe>": 9,
    "<pad>": 10,
}

# RPN: 2 0 + 3 1 - 4 + -
# This represents: (2 + 0) - ((3 - 1) + 4)
# Evaluation steps:
# 1. 2 0 + -> 2
# 2. 3 1 - -> 2
# 3. 2 4 + -> 1
# 4. 2 1 - -> 1
BASE_RPN = jnp.array(
    [
        TOKENS["2"],
        TOKENS["0"],
        TOKENS["+"],
        TOKENS["3"],
        TOKENS["1"],
        TOKENS["-"],
        TOKENS["4"],
        TOKENS["+"],
        TOKENS["-"],
        TOKENS["<pad>"],
        TOKENS["<pad>"],
        TOKENS["<pad>"],
        TOKENS["<pad>"],
        TOKENS["<pad>"],
        TOKENS["<pad>"],
    ]
)

# RPN: 2 2 4 + -
# This represents: 2 - (2 + 4)
# Evaluation steps:
# 1. 2 4 + -> 1
# 2. 2 1 - -> 1
CHILD_RPN = jnp.array(
    [
        TOKENS["2"],
        TOKENS["2"],
        TOKENS["4"],
        TOKENS["+"],
        TOKENS["-"],
        TOKENS["<pad>"],
        TOKENS["<pad>"],
        TOKENS["<pad>"],
        TOKENS["<pad>"],
        TOKENS["<pad>"],
        TOKENS["<pad>"],
        TOKENS["<pad>"],
        TOKENS["<pad>"],
        TOKENS["<pad>"],
        TOKENS["<pad>"],
    ]
)

# RPN: 2 1 -
# This represents: 2 - 1
# Evaluation: 2 1 - -> 1
GRANDCHILD_RPN = jnp.array(
    [
        TOKENS["2"],
        TOKENS["1"],
        TOKENS["-"],
        TOKENS["<pad>"],
        TOKENS["<pad>"],
        TOKENS["<pad>"],
        TOKENS["<pad>"],
        TOKENS["<pad>"],
        TOKENS["<pad>"],
        TOKENS["<pad>"],
        TOKENS["<pad>"],
        TOKENS["<pad>"],
        TOKENS["<pad>"],
        TOKENS["<pad>"],
        TOKENS["<pad>"],
    ]
)

# RPN: 1
# This represents: 1
# Final result
SOLUTION_RPN = jnp.array(
    [
        TOKENS["1"],
        TOKENS["<pad>"],
        TOKENS["<pad>"],
        TOKENS["<pad>"],
        TOKENS["<pad>"],
        TOKENS["<pad>"],
        TOKENS["<pad>"],
        TOKENS["<pad>"],
        TOKENS["<pad>"],
        TOKENS["<pad>"],
        TOKENS["<pad>"],
        TOKENS["<pad>"],
        TOKENS["<pad>"],
        TOKENS["<pad>"],
        TOKENS["<pad>"],
    ]
)


def test_initialization():
    process = RPNArithmeticProcess(p=5, operators=[Operators.ADD, Operators.SUB])
    assert process.p == 5
    assert process.tokens == TOKENS


def test_operations():
    process = RPNArithmeticProcess(p=5, operators=[Operators.ADD, Operators.SUB])
    assert process.operator_functions[Operators.ADD.value](jnp.array(2), jnp.array(3)) == 0
    assert process.operator_functions[Operators.SUB.value](jnp.array(2), jnp.array(3)) == 4


def test_is_operand():
    process = RPNArithmeticProcess(p=5, operators=[Operators.ADD, Operators.SUB])
    assert process.is_operand(jnp.array(TOKENS["0"]))
    assert process.is_operand(jnp.array(TOKENS["1"]))
    assert process.is_operand(jnp.array(TOKENS["2"]))
    assert process.is_operand(jnp.array(TOKENS["3"]))
    assert process.is_operand(jnp.array(TOKENS["4"]))
    assert not process.is_operand(jnp.array(TOKENS["+"]))
    assert not process.is_operand(jnp.array(TOKENS["-"]))
    assert not process.is_operand(jnp.array(TOKENS["="]))
    assert not process.is_operand(jnp.array(TOKENS["<boe>"]))
    assert not process.is_operand(jnp.array(TOKENS["<eoe>"]))
    assert not process.is_operand(jnp.array(TOKENS["<pad>"]))


def test_is_operand_or_operator():
    process = RPNArithmeticProcess(p=5, operators=[Operators.ADD, Operators.SUB])
    assert process.is_operand_or_operator(jnp.array(TOKENS["0"]))
    assert process.is_operand_or_operator(jnp.array(TOKENS["1"]))
    assert process.is_operand_or_operator(jnp.array(TOKENS["2"]))
    assert process.is_operand_or_operator(jnp.array(TOKENS["3"]))
    assert process.is_operand_or_operator(jnp.array(TOKENS["4"]))
    assert process.is_operand_or_operator(jnp.array(TOKENS["+"]))
    assert process.is_operand_or_operator(jnp.array(TOKENS["-"]))
    assert not process.is_operand_or_operator(jnp.array(TOKENS["="]))
    assert not process.is_operand_or_operator(jnp.array(TOKENS["<boe>"]))
    assert not process.is_operand_or_operator(jnp.array(TOKENS["<eoe>"]))
    assert not process.is_operand_or_operator(jnp.array(TOKENS["<pad>"]))


def test_is_operator():
    process = RPNArithmeticProcess(p=5, operators=[Operators.ADD, Operators.SUB])
    assert not process.is_operator(jnp.array(TOKENS["0"]))
    assert not process.is_operator(jnp.array(TOKENS["1"]))
    assert not process.is_operator(jnp.array(TOKENS["2"]))
    assert not process.is_operator(jnp.array(TOKENS["3"]))
    assert not process.is_operator(jnp.array(TOKENS["4"]))
    assert process.is_operator(jnp.array(TOKENS["+"]))
    assert process.is_operator(jnp.array(TOKENS["-"]))
    assert not process.is_operator(jnp.array(TOKENS["="]))
    assert not process.is_operator(jnp.array(TOKENS["<boe>"]))
    assert not process.is_operator(jnp.array(TOKENS["<eoe>"]))
    assert not process.is_operator(jnp.array(TOKENS["<pad>"]))


def test_child_simple_add():
    process = RPNArithmeticProcess(p=5, operators=[Operators.ADD, Operators.SUB])
    n, child_rpn = process.child_sub_equation(BASE_RPN)
    assert n == 5
    assert jnp.all(child_rpn == CHILD_RPN)

    n, child_rpn = process.child_sub_equation(CHILD_RPN)
    assert n == 3
    assert jnp.all(child_rpn == GRANDCHILD_RPN)

    n, child_rpn = process.child_sub_equation(GRANDCHILD_RPN)
    assert n == 1
    assert jnp.all(child_rpn == SOLUTION_RPN)


def test_full_equation():
    process = RPNArithmeticProcess(p=5, operators=[Operators.ADD, Operators.SUB])
    equation = process.full_equation(BASE_RPN, 9, 32)

    # Build the expected array of size 32
    meaningful_tokens = jnp.concatenate(
        [
            jnp.array([TOKENS["<boe>"]]),
            BASE_RPN[:9],
            jnp.array([TOKENS["="]]),
            CHILD_RPN[:5],
            jnp.array([TOKENS["="]]),
            GRANDCHILD_RPN[:3],
            jnp.array([TOKENS["="]]),
            SOLUTION_RPN[:1],
            jnp.array([TOKENS["<eoe>"]]),
        ]
    )
    padding_tokens = jnp.full(32 - len(meaningful_tokens), TOKENS["<pad>"])
    expected = jnp.concatenate([meaningful_tokens, padding_tokens])

    assert jnp.all(equation == expected)


def test_random_sub_equation():
    process = RPNArithmeticProcess(p=5, operators=[Operators.ADD, Operators.SUB])
    key = jax.random.PRNGKey(0)
    k = 3
    n, sub_equation = process.random_sub_equation(key, k)
    assert process.valid_sub_equation(sub_equation, n)


def test_random_sub_equation_jit_vmap():
    process = RPNArithmeticProcess(p=5, operators=[Operators.ADD, Operators.SUB])
    key = jax.random.PRNGKey(0)
    k = 3
    batch_size = 10
    keys = jax.random.split(key, batch_size)
    n, sub_equations = eqx.filter_jit(eqx.filter_vmap(process.random_sub_equation))(keys, k)
    assert sub_equations.shape == (batch_size, 2 * k + 1)
    assert jnp.all(jnp.array([process.valid_sub_equation(sub_equation, n) for sub_equation in sub_equations]))


def test_random_equation():
    process = RPNArithmeticProcess(p=5, operators=[Operators.ADD, Operators.SUB])
    key = jax.random.PRNGKey(0)
    k = 3
    equation = process.random_equation(key, k, 32)
    assert equation.shape == (32,)


def test_valid_sub_equation():
    """Test basic validation cases."""
    process = RPNArithmeticProcess(p=5, operators=[Operators.ADD, Operators.SUB])
    assert process.valid_sub_equation(BASE_RPN, 9)
    assert process.valid_sub_equation(CHILD_RPN, 5)
    assert process.valid_sub_equation(GRANDCHILD_RPN, 3)
    assert process.valid_sub_equation(SOLUTION_RPN, 1)
    assert not process.valid_sub_equation(BASE_RPN, 0)
    assert not process.valid_sub_equation(BASE_RPN, 32)
    assert not process.valid_sub_equation(CHILD_RPN.at[5].set(2), 5)
    assert not process.valid_sub_equation(jnp.array([TOKENS["<boe>"], 1, TOKENS["<eoe>"]]), 1)


@pytest.mark.parametrize(
    ("operators", "description", "rpn_array", "n", "should_be_valid", "reason"),
    [
        # Test with default operators (ADD, SUB)
        ([Operators.ADD, Operators.SUB], "Valid: 1 2 + 3 +", jnp.array([1, 2, 5, 3, 5]), 5, True, "Standard valid RPN"),
        (
            [Operators.ADD, Operators.SUB],
            "Valid: 1 2 3 + +",
            jnp.array([1, 2, 3, 5, 5]),
            5,
            True,
            "Valid with multiple operands",
        ),
        (
            [Operators.ADD, Operators.SUB],
            "Valid: 1 2 3 4 + + +",
            jnp.array([1, 2, 3, 4, 5, 5, 5]),
            7,
            True,
            "Longer valid expression",
        ),
        ([Operators.ADD, Operators.SUB], "Valid: 1", jnp.array([1]), 1, True, "Single operand"),
        ([Operators.ADD, Operators.SUB], "Valid: 1 2 +", jnp.array([1, 2, 5]), 3, True, "Simple binary operation"),
        # Invalid cases - insufficient operands for operators
        (
            [Operators.ADD, Operators.SUB],
            "Invalid: 1 + 2 3 +",
            jnp.array([1, 5, 2, 3, 5]),
            5,
            False,
            "First + lacks operands",
        ),
        ([Operators.ADD, Operators.SUB], "Invalid: + 1 2 +", jnp.array([5, 1, 2, 5]), 4, False, "Starts with operator"),
        (
            [Operators.ADD, Operators.SUB],
            "Invalid: 1 2 + +",
            jnp.array([1, 2, 5, 5]),
            4,
            False,
            "Second + lacks operands",
        ),
        (
            [Operators.ADD, Operators.SUB],
            "Invalid: 1 2 + 3 + +",
            jnp.array([1, 2, 5, 3, 5, 5]),
            6,
            False,
            "Last + lacks operands",
        ),
        (
            [Operators.ADD, Operators.SUB],
            "Invalid: + + 1 2",
            jnp.array([5, 5, 1, 2]),
            4,
            False,
            "Two operators at start",
        ),
        (
            [Operators.ADD, Operators.SUB],
            "Invalid: 1 + + 2",
            jnp.array([1, 5, 5, 2]),
            4,
            False,
            "Consecutive operators",
        ),
        # Invalid cases - wrong final stack size
        ([Operators.ADD, Operators.SUB], "Invalid: 1 2", jnp.array([1, 2]), 2, False, "Two operands, no operator"),
        (
            [Operators.ADD, Operators.SUB],
            "Invalid: 1 2 3",
            jnp.array([1, 2, 3]),
            3,
            False,
            "Three operands, no operator",
        ),
        ([Operators.ADD, Operators.SUB], "Invalid: 1 2 + 3", jnp.array([1, 2, 5, 3]), 4, False, "Extra operand at end"),
        (
            [Operators.ADD, Operators.SUB],
            "Invalid: 1 2 + 3 4",
            jnp.array([1, 2, 5, 3, 4]),
            5,
            False,
            "Two extra operands at end",
        ),
        # Invalid cases - single tokens
        ([Operators.ADD, Operators.SUB], "Invalid: +", jnp.array([5]), 1, False, "Single operator"),
        (
            [Operators.ADD, Operators.SUB],
            "Invalid: 1 +",
            jnp.array([1, 5]),
            2,
            False,
            "Operand + operator, no second operand",
        ),
        # Invalid cases - boundary conditions
        ([Operators.ADD, Operators.SUB], "Invalid: empty", jnp.array([]), 0, False, "Empty sequence"),
        (
            [Operators.ADD, Operators.SUB],
            "Invalid: n > array length",
            jnp.array([1, 2, 5]),
            5,
            False,
            "n exceeds array length",
        ),
        ([Operators.ADD, Operators.SUB], "Invalid: n < 1", jnp.array([1, 2, 5]), 0, False, "n is zero"),
        # Invalid cases - special tokens (should not be in RPN)
        (
            [Operators.ADD, Operators.SUB],
            "Invalid: contains BOE",
            jnp.array([TOKENS["<boe>"], 1, 2, 5]),
            4,
            False,
            "Contains BOE token",
        ),
        (
            [Operators.ADD, Operators.SUB],
            "Invalid: contains EOE",
            jnp.array([1, 2, 5, TOKENS["<eoe>"]]),
            4,
            False,
            "Contains EOE token",
        ),
        (
            [Operators.ADD, Operators.SUB],
            "Invalid: contains EQL",
            jnp.array([1, 2, 5, TOKENS["="]]),
            4,
            False,
            "Contains EQL token",
        ),
        # Edge cases that revealed the original bug
        (
            [Operators.ADD, Operators.SUB],
            "Counterexample: + 1 2",
            jnp.array([5, 1, 2]),
            3,
            False,
            "Starts with operator",
        ),
        (
            [Operators.ADD, Operators.SUB],
            "Invalid: + 1 2 3",
            jnp.array([5, 1, 2, 3]),
            4,
            False,
            "Starts with operator, ends with extra operand",
        ),
        (
            [Operators.ADD, Operators.SUB],
            "Invalid: + 1 + 2",
            jnp.array([5, 1, 5, 2]),
            4,
            False,
            "Two operators, insufficient operands",
        ),
        (
            [Operators.ADD, Operators.SUB],
            "Invalid: 1 + 2 + 3",
            jnp.array([1, 5, 2, 5, 3]),
            5,
            False,
            "Second + lacks operands",
        ),
        # Test with only ADD operator
        ([Operators.ADD], "Valid: 1 2 +", jnp.array([1, 2, 5]), 3, True, "Simple binary operation with ADD only"),
        ([Operators.ADD], "Valid: 1 2 3 + +", jnp.array([1, 2, 3, 5, 5]), 5, True, "Multiple operations with ADD only"),
        ([Operators.ADD], "Invalid: + 1 2", jnp.array([5, 1, 2]), 3, False, "Starts with operator (ADD only)"),
        (
            [Operators.ADD],
            "Invalid: 1 +",
            jnp.array([1, 5]),
            2,
            False,
            "Operand + operator, no second operand (ADD only)",
        ),
        # Test with multiple operators (ADD, SUB, MUL)
        (
            [Operators.ADD, Operators.SUB, Operators.MUL],
            "Valid: 1 2 +",
            jnp.array([1, 2, 5]),
            3,
            True,
            "Simple binary operation with multiple operators",
        ),
        (
            [Operators.ADD, Operators.SUB, Operators.MUL],
            "Valid: 1 2 -",
            jnp.array([1, 2, 6]),
            3,
            True,
            "Subtraction with multiple operators",
        ),
        (
            [Operators.ADD, Operators.SUB, Operators.MUL],
            "Valid: 1 2 *",
            jnp.array([1, 2, 7]),
            3,
            True,
            "Multiplication with multiple operators",
        ),
        (
            [Operators.ADD, Operators.SUB, Operators.MUL],
            "Invalid: + 1 2",
            jnp.array([5, 1, 2]),
            3,
            False,
            "Starts with operator (multiple operators)",
        ),
        (
            [Operators.ADD, Operators.SUB, Operators.MUL],
            "Invalid: - 1 2",
            jnp.array([6, 1, 2]),
            3,
            False,
            "Starts with subtraction (multiple operators)",
        ),
    ],
)
def test_valid_sub_equation_comprehensive(operators, description, rpn_array, n, should_be_valid, reason):
    """Comprehensive test of RPN validation covering all edge cases, error conditions, and operator configurations."""
    process = RPNArithmeticProcess(p=5, operators=operators)
    result = process.valid_sub_equation(rpn_array, n)
    assert result == should_be_valid, f"{description}: expected {should_be_valid}, got {result} ({reason})"


def test_valid_sub_equation_vmap_compatibility():
    """Test that validation works correctly with vmap operations."""
    process = RPNArithmeticProcess(p=5, operators=[Operators.ADD, Operators.SUB])

    # Create a batch of test cases: some valid, some invalid (all with n=3)
    test_equations = jnp.array(
        [
            [1, 2, 5, 10, 10],  # Valid: 1 2 +
            [5, 1, 2, 10, 10],  # Invalid: + 1 2
            [1, 2, 5, 10, 10],  # Valid: 1 2 +
            [1, 5, 2, 10, 10],  # Invalid: 1 + 2
            [1, 2, 5, 10, 10],  # Valid: 1 2 +
        ]
    )

    # Expected results: [True, False, True, False, True]
    expected = jnp.array([True, False, True, False, True])
    jitted_vmapped_validation = eqx.filter_jit(eqx.filter_vmap(process.valid_sub_equation))
    jitted_results = jitted_vmapped_validation(test_equations, 3)
    assert jnp.all(jitted_results == expected), f"Jitted results: expected {expected}, got {jitted_results}"
