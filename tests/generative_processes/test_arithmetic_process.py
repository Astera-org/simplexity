import jax.numpy as jnp

from simplexity.generative_processes.arithmetic_process import ArithmeticProcess, Operators


def test_initialization():
    process = ArithmeticProcess(p=5, operators=[Operators.ADD, Operators.SUB])
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
    process = ArithmeticProcess(p=5, operators=[Operators.ADD, Operators.SUB])
    assert process.operator_functions[Operators.ADD.value](jnp.array(2), jnp.array(3)) == 0
    assert process.operator_functions[Operators.SUB.value](jnp.array(2), jnp.array(3)) == 4


def test_is_operand():
    process = ArithmeticProcess(p=5, operators=[Operators.ADD, Operators.SUB])
    for i in range(5):
        assert isinstance(process.tokens[str(i)], int)
        assert process.is_operand(jnp.array(i))
    for i in range(5, 11):
        assert not process.is_operand(jnp.array(i))


def test_is_operator():
    process = ArithmeticProcess(p=5, operators=[Operators.ADD, Operators.SUB])
    for i in range(0, 5):
        assert not process.is_operator(jnp.array(i))

    plus = jnp.array(process.tokens[Operators.ADD.value])
    assert process.is_operator(plus)
    minus = jnp.array(process.tokens[Operators.SUB.value])
    assert process.is_operator(minus)

    for i in range(7, 11):
        assert not process.is_operator(jnp.array(i))


def test_diagram():
    process = ArithmeticProcess(p=5, operators=[Operators.ADD, Operators.SUB])
    tree = jnp.array([6, 5, 5, 2, 0, 6, 4, 10, 10, 10, 10, 3, 1, 10, 10, 10])
    diagram = process.diagram(tree)
    with open("tests/generative_processes/goldens/equation_trees/base_equation.md") as f:
        expected = f.read().strip()
        assert diagram == expected
    child_tree = jnp.array([6, 2, 5, 10, 10, 2, 4, 10, 10, 10, 10, 10, 10, 10, 10, 10])
    child_diagram = process.diagram(child_tree)
    with open("tests/generative_processes/goldens/equation_trees/child_equation.md") as f:
        expected = f.read().strip()
        assert child_diagram == expected


def test_child_simple_add():
    process = ArithmeticProcess(p=5, operators=[Operators.ADD, Operators.SUB])
    tree = jnp.array([6, 5, 5, 2, 0, 6, 4, 10, 10, 10, 10, 3, 1, 10, 10, 10])
    output = process.child(tree)
    expected = jnp.array([6, 2, 5, 10, 10, 2, 4, 10, 10, 10, 10, 10, 10, 10, 10, 10])
    assert jnp.all(output == expected)
