import difflib
from tokenizer import tokenize_code, normalize_code
import ast


def calculate_similarity(tokenized_code1, tokenized_code2):
    """
    Calculates similarity between two pieces of code using tokenized sequences.
    """

    similarity = difflib.SequenceMatcher(None, tokenized_code1, tokenized_code2).ratio()
    return similarity


def detect_clone_type(code1, code2):
    """
    Detects the type of code clone between two pieces of code.
    """
    # Normalize code by removing whitespace
    normalized_code1 = normalize_code(code1)
    normalized_code2 = normalize_code(code2)

    # Type-1: Identical except for whitespace, formatting, or comments
    if normalized_code1 == normalized_code2:
        return "Type-1 Clone: Identical except for whitespace, formatting, or comments"

    # Tokenize the code
    tokenized_code1 = tokenize_code(code1)
    tokenized_code2 = tokenize_code(code2)

    # Type-2: Structurally identical but differ in identifiers, literals, or function names
    if tokenized_code1 == tokenized_code2:
        return "Type-2 Clone: Structurally identical with minor differences (e.g., identifiers)"

    # Check similarity

    similarity = calculate_similarity(tokenized_code1, tokenized_code2)
    print(f"Similarity: {similarity}")

    # Type-3: Near-miss copies with small modifications (e.g., added/removed/changed statements)
    if similarity > 0.5:  # Lower threshold for near-miss similarity
        return "Type-3 Clone: Near-miss copy with small modifications"

    # Type-4: Semantically similar but structurally different
    try:
        tree1 = ast.parse(code1)
        tree2 = ast.parse(code2)
        if ast.dump(tree1) == ast.dump(tree2):
            return "Type-4 Clone: Semantically similar but structurally different"
    except SyntaxError:
        pass  # Handle invalid code gracefully

    return "No Significant Similarity"


def detect_type_from_files(file1, start1, end1, file2, start2, end2):
    """
    Detects the type of code clone between two pieces of code from files.
    start1, end1, start2, end2 are line numbers (1-based, inclusive start, exclusive end).
    """
    with open(file1, "r") as f:
        lines1 = f.readlines()
        code1 = "".join(lines1[start1 - 1 : end1])

    with open(file2, "r") as f:
        lines2 = f.readlines()
        code2 = "".join(lines2[start2 - 1 : end2])

    return detect_clone_type(code1, code2)
