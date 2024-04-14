
# Common command line options

- `--block-dim` defaults to `1024` threads per block
- `--grid-dim` defaults to `1` threads per block
- `--grid-level` by default Faial only checks block-level data-races; this option enables grid-level analysis.
- `-I` to add include directories
- `-D` to define a macro
- `--kernel` by default Faial analyzes all kernels in the input; this option restricts the analysis to a single kernel.
