import sys
import re


def parse_expression_line(line):
    """Parse expression line into individual terms"""
    line = line.strip()
    if not line:
        return []
    
    # Handle parentheses - remove outer parentheses if present
    if line.startswith('(') and line.endswith(')'):
        line = line[1:-1]
    
    # Split by + and - while keeping the signs
    terms = []
    current_term = ""
    i = 0
    
    while i < len(line):
        char = line[i]
        if char in ['+', '-'] and current_term and not current_term.endswith('*'):
            # Found a separator
            terms.append(current_term.strip())
            current_term = char if char == '-' else ""
        else:
            current_term += char
        i += 1
    
    # Add the last term
    if current_term:
        terms.append(current_term.strip())
    
    # Clean up terms - remove empty ones and leading +
    cleaned_terms = []
    for term in terms:
        term = term.strip()
        if term and term != '+':
            if term.startswith('+'):
                term = term[1:]
            cleaned_terms.append(term)
    
    return cleaned_terms


def contains_variable(term, var_pattern):
    """Check if term contains the specified variable pattern like a12"""
    # Remove parentheses and spaces
    term = term.replace('(', '').replace(')', '').replace(' ', '')
    return var_pattern in term


def count_a_variables(term):
    """Count how many 'a' variables are in the term"""
    # Find all patterns like a11, a12, etc.
    a_pattern = r'a\d\d'
    matches = re.findall(a_pattern, term)
    return len(matches)


def remove_variable_from_expression(line, x, y):
    """Remove axy terms from expression line"""
    var_pattern = f"a{x}{y}"
    original_line = line.strip()
    
    # Parse the line into terms
    terms = parse_expression_line(line)
    
    if not terms:
        return line
    
    # Filter out terms containing the variable
    filtered_terms = []
    total_a_vars = 0
    removed_a_vars = 0
    
    for term in terms:
        a_count = count_a_variables(term)
        total_a_vars += a_count
        
        if contains_variable(term, var_pattern):
            removed_a_vars += a_count
            # Skip this term (remove it)
            continue
        else:
            filtered_terms.append(term)
    
    # If all 'a' variables were removed, return None (delete entire line)
    if total_a_vars > 0 and removed_a_vars == total_a_vars:
        return None
    
    # If no terms left, return None
    if not filtered_terms:
        return None
    
    # If only one term and no changes were made, return original line
    if len(terms) == 1 and len(filtered_terms) == 1:
        return original_line
    
    # Reconstruct the expression
    result = ""
    for i, term in enumerate(filtered_terms):
        if i == 0:
            # First term
            result = term
        else:
            # Subsequent terms
            if term.startswith('-'):
                result += term
            else:
                result += '+' + term
    
    # Only add parentheses if there are multiple terms AND the original had parentheses
    if len(filtered_terms) > 1 and original_line.startswith('(') and original_line.endswith(')'):
        result = f"({result})"
    
    return result


def process_file(input_file, output_file, x, y):
    """Process the expression file"""
    try:
        with open(input_file, 'r') as f:
            lines = f.readlines()
    except FileNotFoundError:
        print(f"Error: File {input_file} not found.")
        return False
    
    print(f"Processing {len(lines)} lines...")
    print(f"Removing variable a{x}{y} from expressions")
    
    processed_lines = []
    removed_lines = 0
    
    for i, line in enumerate(lines):
        line = line.strip()
        if not line:
            continue
            
        result = remove_variable_from_expression(line, x, y)
        
        if result is None:
            removed_lines += 1
            print(f"Line {i+1}: Removed entire line (only contained a{x}{y})")
        else:
            processed_lines.append(result)
            if result != line.strip():
                print(f"Line {i+1}: {line.strip()} -> {result}")
    
    # Write results to output file
    with open(output_file, 'w') as f:
        for line in processed_lines:
            f.write(line + '\n')
    
    print(f"\nProcessing complete:")
    print(f"  Original lines: {len(lines)}")
    print(f"  Removed lines: {removed_lines}")
    print(f"  Remaining lines: {len(processed_lines)}")
    print(f"  Output written to: {output_file}")
    
    return True


def main():
    if len(sys.argv) != 5:
        print("Usage: python reduce.py <input_file> <output_file> <x> <y>")
        print("Example: python reduce.py 333.exp 333_reduced.exp 1 1")
        print("This will remove all terms containing a11 from the expressions")
        sys.exit(1)
    
    input_file = sys.argv[1]
    output_file = sys.argv[2]
    
    try:
        x = int(sys.argv[3])
        y = int(sys.argv[4])
    except ValueError:
        print("Error: x and y must be integers")
        sys.exit(1)
    
    print(f"Input file: {input_file}")
    print(f"Output file: {output_file}")
    print(f"Target variable: a{x}{y}")
    
    success = process_file(input_file, output_file, x, y)
    
    if not success:
        sys.exit(1)


if __name__ == "__main__":
    main()
