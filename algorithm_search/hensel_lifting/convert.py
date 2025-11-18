import re
import ast


def parse_matrix_string(matrix_str):
    """Parse matrix string to array"""
    matrix_str = matrix_str.strip()
    if matrix_str.startswith('{') and matrix_str.endswith('}'):
        matrix_str = matrix_str[1:-1]
    
    rows = []
    row_pattern = r'\{([^}]+)\}'
    for match in re.finditer(row_pattern, matrix_str):
        row_str = match.group(1)
        row = [int(x.strip()) for x in row_str.split(',')]
        rows.append(row)
    
    return rows


def matrix_to_expression_parts(a_matrix, b_matrix, c_matrix):
    """Convert three matrices to expression parts"""
    m, k = len(a_matrix), len(a_matrix[0])
    k2, n = len(b_matrix), len(b_matrix[0])
    n2, m2 = len(c_matrix), len(c_matrix[0])
    
    # Validate dimension consistency
    assert k == k2, f"A columns({k}) must equal B rows({k2})"
    assert n == n2 and m == m2, f"C matrix dimension({n2}x{m2}) should match result dimension({n}x{m})"
    
    # Build expression parts
    a_parts = []
    b_parts = []
    c_parts = []
    
    for i in range(m):
        for j in range(k):
            coeff = a_matrix[i][j]
            if coeff != 0:
                var = f"a{i+1}{j+1}"
                if coeff == 1:
                    a_parts.append(var)
                elif coeff == -1:
                    a_parts.append(f"-{var}")
                else:
                    a_parts.append(f"{coeff}*{var}")
    
    for i in range(k):
        for j in range(n):
            coeff = b_matrix[i][j]
            if coeff != 0:
                var = f"b{i+1}{j+1}"
                if coeff == 1:
                    b_parts.append(var)
                elif coeff == -1:
                    b_parts.append(f"-{var}")
                else:
                    b_parts.append(f"{coeff}*{var}")
    
    for i in range(m):
        for j in range(n):
            coeff = c_matrix[i][j]
            if coeff != 0:
                var = f"c{i+1}{j+1}"
                if coeff == 1:
                    c_parts.append(var)
                elif coeff == -1:
                    c_parts.append(f"-{var}")
                else:
                    c_parts.append(f"{coeff}*{var}")
    
    return a_parts, b_parts, c_parts


def format_expression_part(parts):
    """Format variable list to expression part"""
    if not parts:
        return "0"
    elif len(parts) == 1:
        return parts[0]
    else:
        result = parts[0]
        for part in parts[1:]:
            if part.startswith('-'):
                result += part
            else:
                result += '+' + part
        return f"({result})"


def matrices_to_expression(a_matrix, b_matrix, c_matrix):
    """Convert three matrices to mathematical expression"""
    a_parts, b_parts, c_parts = matrix_to_expression_parts(a_matrix, b_matrix, c_matrix)
    
    a_expr = format_expression_part(a_parts)
    b_expr = format_expression_part(b_parts)
    c_expr = format_expression_part(c_parts)
    
    return f"{a_expr}*{b_expr}*{c_expr}"


def parse_term_from_string(term_str):
    """Parse three matrices from string"""
    term_str = term_str.strip()
    if term_str.startswith('{') and term_str.endswith('}'):
        term_str = term_str[1:-1]
    
    # Find boundaries of three matrices
    brace_count = 0
    matrix_starts = []
    i = 0
    while i < len(term_str):
        if term_str[i] == '{':
            if brace_count == 0:
                matrix_starts.append(i)
            brace_count += 1
        elif term_str[i] == '}':
            brace_count -= 1
        i += 1
    
    # Extract three matrix strings
    matrix_strings = []
    for j in range(len(matrix_starts)):
        start = matrix_starts[j]
        if j < len(matrix_starts) - 1:
            # Find end position before next matrix starts
            brace_count = 0
            end = start
            for k in range(start, len(term_str)):
                if term_str[k] == '{':
                    brace_count += 1
                elif term_str[k] == '}':
                    brace_count -= 1
                    if brace_count == 0:
                        end = k + 1
                        break
        else:
            end = len(term_str)
        
        matrix_str = term_str[start:end]
        matrix_str = matrix_str.rstrip(',').strip()
        matrix_strings.append(matrix_str)
    
    # Parse each matrix
    matrices = []
    for matrix_str in matrix_strings:
        matrix = parse_matrix_string(matrix_str)
        matrices.append(matrix)
    
    return matrices[0], matrices[1], matrices[2]


def split_terms_robust(content):
    """Robust method for splitting terms"""
    # Remove outermost braces
    if content.startswith('{') and content.endswith('}'):
        content = content[1:-1]
    
    terms = []
    i = 0
    
    while i < len(content):
        # Skip whitespace and commas
        while i < len(content) and content[i] in ', \n\t':
            i += 1
        
        if i >= len(content):
            break
        
        # Find complete term: {{{...}}, {{...}}, {{...}}}
        if content[i] == '{':
            start = i
            brace_count = 0
            in_matrix = False
            matrix_count = 0
            
            while i < len(content):
                char = content[i]
                
                if char == '{':
                    brace_count += 1
                    # Check if matrix starts {{
                    if i + 1 < len(content) and content[i + 1] == '{':
                        if not in_matrix:
                            matrix_count += 1
                            in_matrix = True
                elif char == '}':
                    brace_count -= 1
                    # Check if matrix ends }}
                    if i > 0 and content[i - 1] == '}':
                        in_matrix = False
                
                i += 1
                
                # Term ends when braces are balanced and 3 matrices seen
                if brace_count == 0 and matrix_count == 3:
                    term = content[start:i]
                    terms.append(term)
                    break
        else:
            i += 1
    
    return terms


def process_file_reverse(input_file, output_file):
    """Convert matrix format to expression format"""
    with open(input_file, 'r') as file:
        content = file.read().strip()
    
    print(f"Input content length: {len(content)}")
    print(f"Input preview: {content[:200]}...")
    
    terms = split_terms_robust(content)
    print(f"Found {len(terms)} terms")
    
    for i, term in enumerate(terms[:3]):  # Show preview of first 3 terms
        print(f"Term {i+1} preview: {term[:100]}...")
    
    # Convert each term
    expressions = []
    for i, term in enumerate(terms):
        try:
            print(f"\nProcessing term {i+1}...")
            a_matrix, b_matrix, c_matrix = parse_term_from_string(term)
            expression = matrices_to_expression(a_matrix, b_matrix, c_matrix)
            expressions.append(expression)
            print(f"Term {i+1}: {expression}")
        except Exception as e:
            print(f"Error processing term {i+1}: {e}")
            print(f"Term content: {term[:100]}...")
            import traceback
            traceback.print_exc()
    
    # Write to output file
    with open(output_file, 'w') as file:
        for expr in expressions:
            file.write(expr + '\n')
    
    print(f"\nConverted {len(expressions)} terms to {output_file}")


if __name__ == "__main__":
    import sys
    
    # Get file paths from command line arguments
    if len(sys.argv) >= 3:
        input_file = sys.argv[1]
        output_file = sys.argv[2]
    else:
        print("Usage: python convert.py <input_file> <output_file>")
        print("Example: python convert.py 333-a11-mod2-lifted.txt output.txt")
        sys.exit(1)
    
    print(f"Processing file: {input_file}")
    print(f"Output file: {output_file}")
    
    try:
        process_file_reverse(input_file, output_file)
    except FileNotFoundError:
        print(f"File {input_file} not found. Please check the file path.")
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
