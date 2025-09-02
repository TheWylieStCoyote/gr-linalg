# gr-linalg

** ALPHA RC1 **

Linear algebra blocks for GNU Radio built on Eigen. Provides synchronous (stream) blocks and message/PDU variants for common matrix and vector operations and decompositions.

## Features

- Matrix operations: add, subtract, multiply, transpose, trace, determinant, inverse, pseudo-inverse, rank, norms
- Vector operations: dot product, norms, normalize, Kronecker product
- Decompositions: Cholesky (LLT/LDLT), QR, LU, Eigenvalues/vectors, SVD
- Two execution models:
  - Sync stream blocks (gr::block/gr::sync_block)
  - Message/PDU blocks (PMT-based), with named message ports
- C++17, Eigen, GNU Radio runtime integration, Boost.Test unit tests

See `docs/LINEAR_ALGEBRA_OPERATIONS.md` for operation details.

## Requirements

- GNU Radio (3.10+ recommended)
- CMake
- Eigen3
- Boost (for tests)
- A C++17 compiler (GCC/Clang)

## Build and install

```bash
# From the repository root
mkdir -p build
cd build
cmake ..
cmake --build . -j
# Optional install
sudo cmake --install .
# On some systems, you may need to refresh the linker cache afterwards
# sudo ldconfig
```

Notes:

- If Python or GRC bindings are available on your system, CMake will attempt to enable them; otherwise they may be disabled automatically. The YAML block definitions live under `grc/`.
- Use `-DCMAKE_INSTALL_PREFIX=/your/prefix` to install to a custom location.

## Running tests

```bash
# From build directory
ctest --output-on-failure -j
```

You can also run individual test executables directly (they are placed under `build/lib/`, e.g. `build/lib/linalg_qa_matrix_add.cc`).

If you observe teardown hangs due to ControlPort/Thrift on your system, the tests use this environment to disable it:

```bash
export GR_DONT_LOAD_PREFS=0
export GR_CONF_CONTROLPORT_ON=0
export GR_RPCMANAGER_ENABLED=0
```

## Using the blocks

### In GNU Radio Companion (GRC)

- Install the module and ensure GNU Radio can find the OOT module path.
- The block YAML definitions are under `grc/` and will appear under a "Linalg" category (exact name depends on your setup).
- Most blocks accept shape parameters (rows, cols). For stream blocks, the item size (vlen) must match rows*cols for the chosen scalar type.

### In C++ (quick start)

```cpp
#include <gnuradio/top_block.h>
#include <gnuradio/blocks/vector_source_f.h>
#include <gnuradio/blocks/vector_sink_f.h>
#include <gnuradio/linalg/matrix_add.h>

int main() {
  auto tb = gr::make_top_block("linalg_example");

  // 2x2 matrices flattened with vlen=4
  auto srcA = gr::blocks::vector_source_f::make({1.f,2.f,3.f,4.f}, false, 4);
  auto srcB = gr::blocks::vector_source_f::make({5.f,6.f,7.f,8.f}, false, 4);
  auto add  = gr::linalg::matrix_add_sync<float>::make({2,2});
  auto sink = gr::blocks::vector_sink_f::make(4);

  tb->connect(srcA, 0, add,  0);
  tb->connect(srcB, 0, add,  1);
  tb->connect(add,  0, sink, 0);

  tb->run();

  // Results available in sink->data()
  return 0;
}
```

### PDU/message blocks

- PDU blocks expose named message ports (defaults like `in_0`, `out_0`, etc.).
- Send PMT PDUs `{meta, data}` where `data` is a typed PMT uniform vector matching the element type.

## Project layout

- `include/gnuradio/linalg/` – Public C++ headers (APIs)
- `lib/` – Block implementations, base classes, and unit tests
- `grc/` – GRC block definitions (YAML)
- `docs/` – Additional docs and notes
- `examples/` – Example flowgraphs (C++/Python/GRC)

## Development notes

- Base classes:
  - `linalg_base<T>` provides common shape, naming, and validation utilities
  - `linalg_base_sync<T>` wires stream I/O and calls `operation(...)`
  - `linalg_base_pdu<T>` registers message ports and policies for PDU blocks
- Many operations require square or shape-compatible matrices; shape checks are enforced during construction and/or `operation(...)`.
- New blocks should follow the patterns used by `matrix_add` and `matrix_determinant` for sync and PDU variants.

## License

GPL-3.0-or-later. See the SPDX header in source files.
