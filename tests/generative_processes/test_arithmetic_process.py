import jax.numpy as jnp

from simplexity.generative_processes.arithmetic_process import BinaryTreeArithmeticProcess, Operators

# 0 1 2 3 4 5 6 7 8 9 A B C D E
# - + + 2 0 - 4 _ _ _ _ 2 1 _ _
#
#        -
#    +       +
#  2   0   -   4
# _ _ _ _ 3 1 _ _
#
# (2 + 0) - ((3 - 1) + 4)
BASE_TREE = jnp.array([6, 5, 5, 2, 0, 6, 4, 10, 10, 10, 10, 3, 1, 10, 10])
# 0 1 2 3 4 5 6 7 8 9 A B C D E
# - 2 + _ _ 2 4 _ _ _ _ _ _ _ _
#
#        -
#    2       +
#  _   _   2   4
# _ _ _ _ _ _ _ _
#
# 2 - (2 + 4)
CHILD_TREE = jnp.array([6, 2, 5, 10, 10, 2, 4, 10, 10, 10, 10, 10, 10, 10, 10])
# 0 1 2 3 4 5 6 7 8 9 A B C D E
# - 2 1 _ _ _ _ _ _ _ _ _ _ _ _
#
#        -
#    2       1
#  _   _   _   _
# _ _ _ _ _ _ _ _
#
# 2 - 6
GRANDCHILD_TREE = jnp.array([6, 2, 1, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10])
# 0 1 2 3 4 5 6 7 8 9 A B C D E
# 1 _ _ _ _ _ _ _ _ _ _ _ _ _ _
#
#        1
#    _       _
#  _   _   _   _
# _ _ _ _ _ _ _ _
#
# 1
SOLUTION_TREE = jnp.array([1, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10])


def test_initialization():
    process = BinaryTreeArithmeticProcess(p=5, operators=[Operators.ADD, Operators.SUB])
    assert process.p == 5
    tokens = {
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
    assert process.tokens == tokens


def test_operations():
    process = BinaryTreeArithmeticProcess(p=5, operators=[Operators.ADD, Operators.SUB])
    assert process.operator_functions[Operators.ADD.value](jnp.array(2), jnp.array(3)) == 0
    assert process.operator_functions[Operators.SUB.value](jnp.array(2), jnp.array(3)) == 4


def test_is_operand():
    process = BinaryTreeArithmeticProcess(p=5, operators=[Operators.ADD, Operators.SUB])
    for i in range(5):
        assert isinstance(process.tokens[str(i)], int)
        assert process.is_operand(jnp.array(i))
    for i in range(5, 11):
        assert not process.is_operand(jnp.array(i))


def test_is_operator():
    process = BinaryTreeArithmeticProcess(p=5, operators=[Operators.ADD, Operators.SUB])
    for i in range(0, 5):
        assert not process.is_operator(jnp.array(i))

    plus = jnp.array(process.tokens[Operators.ADD.value])
    assert process.is_operator(plus)
    minus = jnp.array(process.tokens[Operators.SUB.value])
    assert process.is_operator(minus)

    for i in range(7, 11):
        assert not process.is_operator(jnp.array(i))


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
