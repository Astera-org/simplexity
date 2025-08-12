"""Factory functions for creating arithmetic processes from configs."""

from simplexity.generative_processes.arithmetic_process import (
    ArithmeticProcess,
    BinaryTreeArithmeticProcess,
    Operators,
    RPNArithmeticProcess,
)


def create_arithmetic_process(
    p: int,
    max_operations: int,
    operators: list[str],
    representation: str,
) -> ArithmeticProcess:
    """Create an arithmetic process from parameters.

    This function handles the conversion of operator strings to enum values
    and instantiates the appropriate arithmetic process class.

    Args:
        p: The modulus for arithmetic operations
        max_operations: The maximum number of operations to take in the equation
        operators: List of operator strings like ["+", "-", "*"]
        representation: Type of process ("binary_tree" or "rpn")

    Returns:
        Instantiated arithmetic process

    Raises:
        ValueError: If the target class is not supported or operators are invalid
    """
    # Convert operator strings to enum values
    operator_map = {
        "+": Operators.ADD,
        "-": Operators.SUB,
        "*": Operators.MUL,
    }

    operator_enums = []
    for op_str in operators:
        if op_str not in operator_map:
            raise ValueError(f"Unsupported operator: {op_str}. Supported operators: {list(operator_map.keys())}")
        operator_enums.append(operator_map[op_str])

    # Instantiate the appropriate class
    if representation == "binary_tree":
        return BinaryTreeArithmeticProcess(p=p, operators=operator_enums, max_operations=max_operations)
    elif representation == "rpn":
        return RPNArithmeticProcess(p=p, operators=operator_enums, max_operations=max_operations)
    else:
        raise ValueError(f"Unsupported process type: {representation}")


def register_arithmetic_processes() -> None:
    """Register arithmetic process factory functions with Hydra."""
    # This function can be called to register custom resolvers if needed
    # For now, we'll use the factory function directly
    pass
