# AlphaSparseLLM: Discovering Faster Sparse Matrix Multiplication Algorithms on multi-core CPU for LLM Inference
The deployment of pruned Large Language Models (LLMs) on CPUs presents a cost-efficient edge inference solution but encounters challenges due to the unique sparsity characteristics of pruned weights, including moderate sparsity mismatch (60%-80%), unstructured patterns, layerwise sparsity variance, and tall-skinny matrices in decoding. We propose AlphaSparseLLM, a hardware-aware framework for accelerating pruned LLM inference on multi-core CPUs. AlphaSparseLLM offers: (1) an efficient algorithm search through random walks and sparse Hensel lifting for sparsity-adapted fast matrix multiplication (FMM) algorithms; (2) a dualpath execution workflow that dynamically routes unstructured non-zero to specialized SpMM kernels while processing denser blocks with identified FMM algorithms, enabling fine-grained load balancing; (3) CPU-specific optimizations such as NUMA-aware threading, prefetching, and vectorization. Evaluations on LLaMA-2 and Qwen-2 models indicate throughput improvements of 1.03x to 1.12x over CPU libraries (GGML, MKL, oneDNN, LIBXSMM) and state-of-the-art FMM frameworks.

## Algorithms

The folder `algorithms` contains optimal sparse matrix multiplication strategies for size between 2×2×2 and 6×6×6 in both modular arithmetic and standard arithmetic found by AlphaSparseLLM. More specifically, for matrix multiplication AB=C, in each matrix multiplication size, we sequentially introduced sparse elements at all positions of matrix A (i.e., setting them to 0, and they are not included in subsequent calculations), once at a time, and provided the fast multiplication strategies with least multiplication times for each size.

- `modular` folder contains sparse matrix multiplication algorithms in moduler arithmetic (i.e., each position in the matrix is either 0 or 1, except for the sparse positions that are determined to be 0.)
- `standard` folder contains sparse matrix multiplication algorithms in standard arithmetic

### How to interpret the algorithms

Take a  $2\times 2\times 2$ sparse matrix multiplication algorithm with the (1,1) element of matrix $\textbf{A}$ being 0 as an example. In our representation method corresponding to low-rank decomposition of three-dimensional spatial tensors, it would be written as:

```
(a22)*(b11+b22)*(c11+c22)
(a21+a22)*(b11)*(c12-c22)
(a22)*(b21-b11)*(c11+c12)
(a12)*(b22)*(-c11+c21)
(a21)*(b11+b12)*(c22)
(a12-a22)*(b21+b22)*(c11)
```

For the part contains a and b, each row corresponds to an intermediate variable $M_i$ calculated by adding or subtracting elements from A and B:
$$
M_1=A_{22}(B_{11}+B_{22})\\
M_2=(A_{21}+A_{22})B_{11}\\
M_3=A_{22}(B_{21}-B_{11})\\
M_4=A_{12}B_{22}\\
M_5=A_{21}(B_{11}+B_{12})\\
M_6=(A_{12}-A_{22})(B_{21}+B_{22})
$$
For the part contains c, each row represents the role of the intermediate variable $M_i$ when calculating a specific element of the final matrix C. The coefficient in front of $C_{xy}$ represents the actual coefficient of $M_i$ when computing the element $(y,x)$ of matrix C. (To be consistent with the notation for tensor low-rank decomposition, the subscript of c needs to be **transposed**)
$$
C_{11}=M_1+M_3-M_4+M_6\\
C_{12}=M_4\\
C_{21}=M_2+M_3\\
C_{22}=M_1-M_2+M_5
$$
The decomposition algorithm above is equivalent to the matrix multiplication calculation below:
$$
\left(
\begin{array}{cc}
 c _ { 11 } & c _ { 12 } \\
 c _ { 21 } & c _ { 22 }
\end{array}
\right)
=
\left(
\begin{array}{cc}
0 & a _ { 12 } \\
 a _ { 21 } & a _ { 22 }
\end{array}
\right)
\cdot 
\left(
\begin{array}{cc}
 b _ { 11 } & b _ { 12 } \\
 b _ { 21 } & b _ { 22 }
\end{array}
\right)
$$


## Algorithm Search

In the `Algorithm Search` folder, we include the code for a whole pipeline that can automatically discover fast matrix multiplication strategy with constraints of sparsity.

As introduced in the paper, the efficient algorithm search for sparse-friendly fast matrix multiplication contains the following steps:

- IsReducible
- Reduce
- Flip
- Hensel Lifting

The entire pipeline must begin with a multiplication strategy for a modular of a known corresponding size without the introduction of sparsity, with the representation same as in the `Algorithms` Part. This initial strategy can be either a naive O(n³) approach or an optimized one discovered through mathematical methods. Since the flip operation must be performed in a modular context, if a standard strategy is used, it must first be converted into a modular strategy. This conversion is done by unifying the coefficient preceding each element (a, b, or c) to 1 if it is odd, and to 0 if it is even. The result is a final strategy representation consisting solely of 0s and 1s. For example:

```sh
# a naive strategy
(a11)*(b11)*(c11)
(a12)*(b21)*(c11)
(a11)*(b12)*(c21)
(a12)*(b22)*(c21)
(a21)*(b11)*(c12)
(a22)*(b21)*(C12)
(a21)*(b12)*(c22)
(a22)*(b22)*(c22)

# an optimized strategy (same as the naive strategy above in modular context)
(a12)*(b11+b12+b21+b22)*(c11)
(a11+a21)*(b12+b22)*(c11+c21)
(a11+a12)*(b11+b12)*(c21+c22)
(a21)*(b11)*(c11+c12+c21+c22)
(a11+a12+a21+a22)*(b22)*(c22)
(a11+a12+a21)*(b11+b12+b22)*(c11+c21+c22)
```

Both can serve as an initial strategy input for the algorithm discovery procedure of size 2\*2\*2.

### IsReducible & Reduce

First, run `reduce.py` to examine whether the strategy can be reduced under sparsity constraints (i.e. one element in matrix $\textbf{A}$ is determined to be 0), and reduce the strategy to a sparse-friendly one. Under most circumstances, this step contributes to the reduction of one or more ranks.

Usage: `reduce.py <input_file> <output_file> <x> <y>` for setting element $a_{xy}$ of the strategy in `input_file` to be 0.

### Flip

- First, compile the source code to generate the executable file using a Makefile:

```bash
cd flip/code
make
```

- Next, prepare the strategy file (after reduction) and run the flip procedure:

  Usage: `./flip <strategy file> <m> <n> <k> <pathlength> 1` for a strategy of size m\*n\*k (if m=n=k, parameter n and k can be omitted)

In the `flip` folder, we include an example strategy file `333-a11.exp`, with the $(1,1)$ element reduced for a 3\*3\*3 strategy. By executing `./flip 333-a11.exp 3 100000 1`，we can get an optimized modular strategy with a reduced rank of 21. The optimized strategy would be generated under the same path. If no strategy is generated, it indicates that the rank of strategy cannot be further reduced within the given pathlength.

### Hensel Lifting

The hensel lifting code is presented in a wolfram notebook `lift.wlnb`, and has to be run using Mathematica software. In the notebook, you should set the following parameters before running:

```mathematica
filePath = "\\path\\to\\your\\folder\\333-a11-mod2.exp";
k = 3; 
outputPath = "\\path\\to\\your\\folder\\333-a11-mod2-lifted.txt";
```

- `filePath` is the **absolute path** where the strategy generated after Flip that needs to be lifted is stored
- `k` is the target of lifting, which means the program would lift the strategy from a mod-2 one to a mod-$2^k$ one.
- `outputPath` is the **absolute path** where you store the lifted result, which is presented in a tensor format.

After lifting, a result file will be generated, which is a matrix multiplication strategy for target modulus 2k. We include two scripts for you to further process the lifted results:

- `convert.py`: converting the tensor representation to normal representation introduced in *Algorithms* Part. Usage: `python convert.py <input_file> <output_file>`
- `ground_truth.py`：inspecting whether the lifted result is the ground-truth strategy for sparse matrix multiplication (i.e. the strategy is correct not only under modular arithmetic precision, but also in the sense of linear algebra, regardless of precision). Usage: `python ground_truth.py <streategy file path>`
