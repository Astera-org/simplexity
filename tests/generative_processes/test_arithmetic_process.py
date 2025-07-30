import jax
import jax.numpy as jnp

from simplexity.generative_processes.arithmetic_process import BinaryTreeArithmeticProcess, Operators

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

# 0 1 2 3 4 5 6 7 8 9 A B C D E
# - + + 2 0 - 4 _ _ _ _ 3 1 _ _
#
#        -
#    +       +
#  2   0   -   4
# _ _ _ _ 3 1 _ _
#
# (2 + 0) - ((3 - 1) + 4)
BASE_TREE = jnp.array(
    [
        TOKENS["-"],
        TOKENS["+"],
        TOKENS["+"],
        TOKENS["2"],
        TOKENS["0"],
        TOKENS["-"],
        TOKENS["4"],
        TOKENS["<pad>"],
        TOKENS["<pad>"],
        TOKENS["<pad>"],
        TOKENS["<pad>"],
        TOKENS["3"],
        TOKENS["1"],
        TOKENS["<pad>"],
        TOKENS["<pad>"],
    ]
)
# 0 1 2 3 4 5 6 7 8 9 A B C D E
# - 2 + _ _ 2 4 _ _ _ _ _ _ _ _
#
#        -
#    2       +
#  _   _   2   4
# _ _ _ _ _ _ _ _
#
# 2 - (2 + 4)
CHILD_TREE = jnp.array(
    [
        TOKENS["-"],
        TOKENS["2"],
        TOKENS["+"],
        TOKENS["<pad>"],
        TOKENS["<pad>"],
        TOKENS["2"],
        TOKENS["4"],
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
# 0 1 2 3 4 5 6 7 8 9 A B C D E
# - 2 1 _ _ _ _ _ _ _ _ _ _ _ _
#
#        -
#    2       1
#  _   _   _   _
# _ _ _ _ _ _ _ _
#
# 2 - 6
GRANDCHILD_TREE = jnp.array(
    [
        TOKENS["-"],
        TOKENS["2"],
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
    ]
)
# 0 1 2 3 4 5 6 7 8 9 A B C D E
# 1 _ _ _ _ _ _ _ _ _ _ _ _ _ _
#
#        1
#    _       _
#  _   _   _   _
# _ _ _ _ _ _ _ _
#
# 1
SOLUTION_TREE = jnp.array(
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
    process = BinaryTreeArithmeticProcess(p=5, operators=[Operators.ADD, Operators.SUB])
    assert process.p == 5
    assert process.tokens == TOKENS


def test_operations():
    process = BinaryTreeArithmeticProcess(p=5, operators=[Operators.ADD, Operators.SUB])
    assert process.operator_functions[Operators.ADD.value](jnp.array(2), jnp.array(3)) == 0
    assert process.operator_functions[Operators.SUB.value](jnp.array(2), jnp.array(3)) == 4


def test_is_operand():
    process = BinaryTreeArithmeticProcess(p=5, operators=[Operators.ADD, Operators.SUB])
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
    process = BinaryTreeArithmeticProcess(p=5, operators=[Operators.ADD, Operators.SUB])
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
    process = BinaryTreeArithmeticProcess(p=5, operators=[Operators.ADD, Operators.SUB])
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


def test_diagram():
    process = BinaryTreeArithmeticProcess(p=5, operators=[Operators.ADD, Operators.SUB])

    base_diagram = process.diagram(BASE_TREE)
    with open("tests/generative_processes/goldens/equation_trees/base_equation.md") as f:
        expected = f.read().strip()
        assert base_diagram == expected

    child_diagram = process.diagram(CHILD_TREE)
    with open("tests/generative_processes/goldens/equation_trees/child_equation.md") as f:
        expected = f.read().strip()
        assert child_diagram == expected


def test_child_simple_add():
    process = BinaryTreeArithmeticProcess(p=5, operators=[Operators.ADD, Operators.SUB])
    n, child_tree = process.child_sub_equation(BASE_TREE)
    assert n == 7
    assert jnp.all(child_tree == CHILD_TREE)

    n, child_tree = process.child_sub_equation(CHILD_TREE)
    assert n == 3
    assert jnp.all(child_tree == GRANDCHILD_TREE)

    n, child_tree = process.child_sub_equation(GRANDCHILD_TREE)
    assert n == 1
    assert jnp.all(child_tree == SOLUTION_TREE)


def test_full_equation():
    process = BinaryTreeArithmeticProcess(p=5, operators=[Operators.ADD, Operators.SUB])
    equation = process.full_equation(BASE_TREE, 15, 32)
    expected = jnp.concatenate(
        [
            jnp.array([TOKENS["<boe>"]]),
            BASE_TREE[:15],
            jnp.array([TOKENS["="]]),
            CHILD_TREE[:7],
            jnp.array([TOKENS["="]]),
            GRANDCHILD_TREE[:3],
            jnp.array([TOKENS["="]]),
            SOLUTION_TREE[:1],
            jnp.array([TOKENS["<eoe>"]]),
            jnp.array([TOKENS["<pad>"]]),
        ]
    )
    assert jnp.all(equation == expected)


def test_random_sub_equation():
    process = BinaryTreeArithmeticProcess(p=5, operators=[Operators.ADD, Operators.SUB])
    key = jax.random.PRNGKey(0)
    k = 3
    n, sub_equation = process.random_sub_equation(key, k)
    assert process.valid_sub_equation(sub_equation, n)


def test_random_equation():
    process = BinaryTreeArithmeticProcess(p=5, operators=[Operators.ADD, Operators.SUB])
    key = jax.random.PRNGKey(0)
    k = 3
    equation = process.random_equation(key, k, 32)
    assert equation.shape == (32,)


def test_valid_sub_equation():
    process = BinaryTreeArithmeticProcess(p=5, operators=[Operators.ADD, Operators.SUB])
    assert process.valid_sub_equation(BASE_TREE, 15)
    assert process.valid_sub_equation(CHILD_TREE, 7)
    assert process.valid_sub_equation(GRANDCHILD_TREE, 3)
    assert process.valid_sub_equation(SOLUTION_TREE, 1)
    assert not process.valid_sub_equation(BASE_TREE, 0)
    assert not process.valid_sub_equation(BASE_TREE, 32)
    assert not process.valid_sub_equation(CHILD_TREE.at[8].set(2), 7)
    assert not process.valid_sub_equation(jnp.array([TOKENS["<boe>"], 1, TOKENS["<eoe>"]]), 1)
