import tokenize
import keyword
import os
from io import BytesIO


def tokenize_code(code):
    """
    Tokenize the source code using the Python tokenize module.

    It returns a string listing the names of the generated tokens, separated by spaces.
    This normalizes the code (ignoring variable names, literals, comments, etc.) so that similarities in the structure can be detected.

    Tokens are processed as follows:
    - ENCODING and ENDMARKER tokens are ignored.
    - Comment and line break tokens (NEWLINE, NL) are ignored.
    - NAME tokens are checked: if they correspond to a keyword (according to the keyword module), the keyword is returned (in uppercase), and if not, they are replaced by the "ID" token.
    - For all other tokens, the standard name is used (e.g., NUMBER, STRING, etc.)
    """
    tokens_list = []
    code_bytes = code.encode("utf-8")
    readline = BytesIO(code_bytes).readline

    try:
        for tok in tokenize.tokenize(readline):
            token_type = tok.type
            token_string = tok.string

            if token_type in (tokenize.ENCODING, tokenize.ENDMARKER):
                continue

            if token_type in (tokenize.COMMENT, tokenize.NEWLINE, tokenize.NL):
                continue

            if token_type == tokenize.NAME:
                if keyword.iskeyword(token_string):
                    tokens_list.append(token_string.upper())
                else:
                    tokens_list.append("ID")
            else:
                tokens_list.append(tokenize.tok_name[token_type])
    except tokenize.TokenError as e:
        print("Error tokenizando el código:", e)

    return " ".join(tokens_list)


def safe_tokenize_code(code):
    try:
        return tokenize_code(code)
    except (tokenize.TokenError, IndentationError):
        cleaned_code = "\n".join(line.lstrip() for line in code.splitlines())
        wrapped_code = "def _dummy():\n"
        wrapped_code += "\n".join("    " + line for line in cleaned_code.splitlines())
        try:
            return tokenize_code(wrapped_code)
        except Exception as e:
            print("Tokenization failed:", e)
            return ""


def normalize_code(code):
    """
    Normalize the code by removing comments, whitespace, and newlines.
    Returns a string of token names representing the code structure.
    """

    def _normalize(c):
        tokens_list = []
        code_bytes = c.encode("utf-8")
        readline = BytesIO(code_bytes).readline

        for tok in tokenize.tokenize(readline):
            token_type = tok.type
            # Ignore encoding, end markers, comments, newlines, and whitespace
            if token_type in (
                tokenize.ENCODING,
                tokenize.ENDMARKER,
                tokenize.COMMENT,
                tokenize.NEWLINE,
                tokenize.NL,
                tokenize.INDENT,
                tokenize.DEDENT,
            ):
                continue
            tokens_list.append(tok.string)
        return " ".join(tokens_list)

    try:
        return _normalize(code)
    except (tokenize.TokenError, IndentationError):
        # Try wrapping in a dummy function if code fragment fails
        cleaned_code = "\n".join(line.lstrip() for line in code.splitlines())
        wrapped_code = "def _dummy():\n" + "\n".join(
            "    " + line for line in cleaned_code.splitlines()
        )
        try:
            return _normalize(wrapped_code)
        except Exception as e:
            print("Normalization failed:", e)
            return ""


def save_preprocessed_code(
    original_path,
    tokenized_code,
    source_base="output_dataset",
    out_base="code_preprocessed",
):
    """
    Save the tokenized content in a path that preserves the same structure relative to the original base directory (source_base), in the out_base folder.
    """
    rel_path = os.path.relpath(original_path, source_base)
    rel_path_txt = os.path.splitext(rel_path)[0] + ".txt"
    new_path = os.path.join(out_base, rel_path_txt)
    os.makedirs(os.path.dirname(new_path), exist_ok=True)
    with open(new_path, "w", encoding="utf-8") as f:
        f.write(tokenized_code)
    print(f"Preprocesado guardado en: {new_path}")
