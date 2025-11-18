import ast
import numpy as np
import re

def parse_data_file(file_path):
    """Parse strategy data"""
    with open(file_path, 'r') as f:
        content = f.read().strip()
    
    content = content.replace('{', '[').replace('}', ']')
    
    try:
        data = ast.literal_eval(content)
    except (SyntaxError, ValueError) as e:
        print(f"Parse error: {e}")
        return None
    
    if not isinstance(data, list) or len(data) == 0:
        print("Parsed data is not a list or empty")
        return None
    
    return data

def determine_dimensions(block):
    """Determine matrix dimensions from strategy block"""
    m = len(block[0]) 
    r = len(block[0][0]) 
    n = len(block[1][0])
    
    return m, r, n

def parse_sparse_position(file_path):
    """Parse sparse position from filename"""
    match = re.search(r'a(\d)(\d)', file_path)
    if match:
        return (int(match.group(1)) - 1, int(match.group(2)) - 1)
    else:
        return None

def apply_strategy(A, B, strategy, debug=False):
    """Apply AlphaTensor strategy for matrix multiplication"""
    m, r, n = determine_dimensions(strategy[0])
    C = np.zeros((m, n))
    
    for idx, block in enumerate(strategy):
        # U matrix: A combination coefficients (m×r)
        U = np.array(block[0], dtype=int)
        # V matrix: B combination coefficients (r×n)
        V = np.array(block[1], dtype=int)
        # W matrix: C combination coefficients (n×m)
        W = np.array(block[2], dtype=int)
        
        # Calculate combination coefficients
        u_factor = np.sum(U * A)
        v_factor = np.sum(V * B)
        term = u_factor * v_factor
        
        # Update result matrix (W is n×m, transpose to m×n)
        contribution = term * W.T
        C += contribution
        
        if debug:
            print(f"Block {idx + 1} contribution:")
            print(f"  U: {U.tolist()}")
            print(f"  V: {V.tolist()}")
            print(f"  W: {W.tolist()}")
            print(f"  Inner product: U·A={u_factor}, V·B={v_factor}")
            print(f"  Scalar term: {term}")
            print(f"  Contribution matrix: \n{contribution}")
            print(f"  Current C: \n{C}")
            print("-"*50)
    
    return C

def verify_strategy(strategy, sparse_index, num_tests=3):
    """Verify if strategy correctly computes matrix multiplication"""
    m, r, n = determine_dimensions(strategy[0])
    print(f"Matrix multiplication dimensions: A({m}×{r}) * B({r}×{n}) → C({m}×{n})")
    print(f"Strategy contains {len(strategy)} multiplication terms")
    
    if sparse_index:
        sparse_desc = f"position({sparse_index[0]+1},{sparse_index[1]+1}) (a{sparse_index[0]+1}{sparse_index[1]+1})"
    else:
        sparse_desc = "no specific sparse position"
    print(f"Sparse setting: {sparse_desc}")
    
    test_cases = []
    
    # Add test cases
    for test_idx in range(num_tests):
        # Generate random matrices based on strategy dimensions
        A = np.random.randint(0, 10, (m, r))
        B = np.random.randint(0, 10, (r, n))
        
        # Apply sparse setting
        if sparse_index and sparse_index[0] < m and sparse_index[1] < r:
            print(f"Setting A[{sparse_index[0]},{sparse_index[1]}] to 0 (a{sparse_index[0]+1}{sparse_index[1]+1})")
            A[sparse_index[0], sparse_index[1]] = 0
        
        C_std = A @ B
        test_cases.append((f"Test{test_idx+1}", A, B, C_std))
    
    # Execute all tests
    all_passed = True
    for idx, (name, A, B, C_std) in enumerate(test_cases):
        print(f"\n{name}:")
        print(f"A = \n{A}")
        print(f"B = \n{B}")
        print(f"Standard result (A@B) = \n{C_std}")
        
        # Strategy computation (enable debug for first test)
        debug = (idx == 0)
        C_opt = apply_strategy(A, B, strategy, debug=debug)
        
        # Show strategy result
        print(f"Strategy result: \n{np.round(C_opt, 4)}")
        
        # Calculate absolute error
        abs_error = np.abs(C_std - C_opt)
        max_error = np.max(abs_error)
        is_correct = max_error < 1e-6
        
        if not is_correct:
            print("❌ Verification failed - Error analysis:")
            # Find all error positions
            errors = []
            for i in range(m):
                for j in range(n):
                    if abs_error[i, j] > 1e-6:
                        errors.append((i, j, C_std[i, j], C_opt[i, j], abs_error[i, j]))
            
            # Show error positions
            for error in errors:
                print(f"Position({error[0]},{error[1]}): "
                      f"expected={error[2]}, "
                      f"computed={error[3]:.6f}, "
                      f"error={error[4]:.6f}")
            
            # Visualize differences
            print("\nDifference matrix (✗ indicates error):")
            diff_matrix = np.where(abs_error > 1e-6, '✗', '✓')
            for row in diff_matrix:
                print(" ".join(row))
            
            all_passed = False
        else:
            print(f"✅ Test passed (max error: {max_error:.6f})")
    
    if all_passed:
        print("\n✅ All tests passed")
    else:
        print("\n❌ Some tests failed")
    return all_passed

def analyze_and_verify(file_path):
    """Analyze strategy file and verify"""
    # Parse file
    data = parse_data_file(file_path)
    if data is None:
        return False
    
    print(f"Parse successful: {len(data)} blocks total")
    
    # Print structure analysis of first block
    block0 = data[0]
    m, r, n = determine_dimensions(block0)
    print("\nFirst block structure analysis:")
    print(f"  Matrix U (A combination): {len(block0[0])} rows × {len(block0[0][0])} cols")
    print(f"  Matrix V (B combination): {len(block0[1])} rows × {len(block0[1][0])} cols")
    print(f"  Matrix W (C combination): {len(block0[2])} rows × {len(block0[2][0])} cols")
    
    # Parse sparse position from filename
    sparse_index = parse_sparse_position(file_path)
    
    # Verify strategy
    print("\nStarting strategy verification...")
    return verify_strategy(data, sparse_index, num_tests=3)

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        file_path = sys.argv[1]
    else:
        file_path = "/path/to/your/strategy.txt"

    print("="*50)
    print(f"Verifying strategy file: {file_path}")
    print("="*50)

    analyze_and_verify(file_path)