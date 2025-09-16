import ast
import inspect
import pytest
from glob import glob
import re

allwed_nodes = (
    ast.FunctionDef,
    ast.ClassDef,
    ast.Import,
    ast.ImportFrom
)


def is_main_block(node):
    """
    Check if the given AST node is an if __name__ == "__main__": block.
    """
    if isinstance(node, ast.If) and isinstance(node.test, ast.Compare):
        if (
            isinstance(node.test.left, ast.Name) and 
            node.test.left.id == "__name__" and
            len(node.test.ops) == 1 and 
            isinstance(node.test.ops[0], ast.Eq) and
            len(node.test.comparators) == 1 and 
            isinstance(node.test.comparators[0], ast.Constant) and
            node.test.comparators[0].value == "__main__"
        ):
            return True
    return False


def test_filename():
    pass

def get_filename():
    """
    This function gets the filename of the student's assignment. If not found, looks for assignment.py instead
    """
    assignment_files = glob("assignment*")
    default_name = "assignment.py"
    pattern = re.compile(r"^assignment_\d{7}\.py$")
    valid_assignment_files = [f for f in assignment_files
                                if pattern.match(f)] 
    
    if len(valid_assignment_files)==0:
        if default_name in assignment_files:
            raise AssertionError(
                "Please rename the assignment file with your student id eg: assignment.py --> assignment_1234567.py"
            )
        else:
            raise AssertionError(
                "No valid assignment files found..."
                )
    if len(valid_assignment_files)>1:
        raise AssertionError(
            f"Multiple assignment files found. Please keep just 1 file"
            )
    filename = valid_assignment_files[0]
    print(f"Using {filename}")

    return filename

def test_code_outside_functions():
    """
    This function checks that there is no code outside of functions or classes.
    """
    filename = get_filename()
    with open(filename,"r") as f:
        student_module = f.read()

    tree = ast.parse(student_module)

    for node in tree.body:
        if not isinstance(
            node, 
            allwed_nodes
        ):
            # Check if this is the if __name__ == "__main__": block
            if is_main_block(node):
                continue

            raise AssertionError(
                f"Code found outside of functions or classes at line number: "
                f"{node.lineno}"
            )


def test_function_signatures():
    """
    This function checks that the function signatures match the expected ones.
    """
    filename = get_filename()
    student =  __import__(filename.strip(".py"))
    
    expected_signatures = {
        "similarity_matrix": ["matrix", "k=5", "axis=0"],
        "user_based_cf": [
            "user_id",
            "movie_id",
            "user_similarity",
            "user_item_matrix",
            "k=5"
        ],
        "item_based_cf": [
            "user_id",
            "movie_id",
            "item_similarity",
            "user_item_matrix",
            "k=5"
        ],
        "matrix_factorization":[
            "utility_matrix: numpy.ndarray",
            "feature_dimension=2",
            "learning_rate=0.001",
            "regularization=0.02",
            "n_steps=2000"
        ]
    }

    for func_name, expected_params in expected_signatures.items():
        if '.' in func_name:
            class_name, method_name = func_name.split('.')
            func = getattr(student, class_name).__dict__[method_name]
        else:
            func = getattr(student, func_name)

        sig = inspect.signature(func)
        params = [str(param) for param in sig.parameters.values()]

        assert params == expected_params, (
            f"Function '{func_name}' has incorrect signature. "
            f"Expected: {expected_params}, Found: {params}"
        )

if __name__ == "__main__":
    pytest.main([__file__])