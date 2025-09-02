#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
GNU Radio block management for Linear Algebra Performance Profiler.

Handles discovery, instantiation, and connection of GNU Radio linear algebra
blocks for performance testing.
"""

import sys
import os
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
import numpy as np
from gnuradio import gr, blocks

from .models import BlockInfo


class BlockManager:
    """Manages GNU Radio linear algebra blocks for performance testing"""

    def __init__(self):
        # Set up path to find gnuradio.linalg C++ bindings
        build_path = (
            Path(__file__).parent.parent.parent
            / "build"
            / "test_modules"
            / "gnuradio"
            / "linalg"
        )
        if build_path.exists() and str(build_path) not in sys.path:
            sys.path.insert(0, str(build_path))

        # Try to import the C++ bindings
        try:
            import linalg_python

            self.linalg_python = linalg_python
            self.bindings_available = True
        except ImportError as e:
            print(f"Warning: Could not import linalg_python bindings: {e}")
            self.linalg_python = None
            self.bindings_available = False

        # Block categories and their available blocks
        self.block_categories = {
            "matrix_basic": [
                "matrix_add",
                "matrix_subtract",
                "matrix_multiply",
                "matrix_transpose",
                "matrix_determinant",
                "matrix_trace",
            ],
            "matrix_advanced": [
                "matrix_inverse",
                "matrix_norm",
                "matrix_condition_number",
                "matrix_pseudo_inverse",
                "matrix_reshape",
                "matrix_solve",
                "matrix_exp",
                "matrix_hermitian",
                "matrix_kronecker_product",
            ],
            "matrix_elementwise": [
                "matrix_elementwise_multiply",
                "matrix_elementwise_divide",
                "matrix_diag",
                "matrix_power",
            ],
            "matrix_generators": ["eye", "ones", "zeros", "matrix_source_const"],
            "vector_operations": [
                "vector_norm",
                "vector_normalize",
                "vector_cross_product",
                "vector_angle",
                "vector_outer_product",
                "dot_product",
                "vector_correlate",
            ],
            "decompositions": [
                "decomp_svd",
                "decomp_eigen",
                "decomp_lu",
                "decomp_qr",
                "decomp_cholesky",
                "decomp_schur",
                "decomp_hessenberg",
            ],
            "solvers": ["matrix_solve", "solve_least_squares", "solve_iterative"],
        }

        # Precision type mappings
        self.precision_types = {
            "f": ("float", np.float32, blocks.vector_source_f, blocks.vector_sink_f),
            "c": (
                "complex_float",
                np.complex64,
                blocks.vector_source_c,
                blocks.vector_sink_c,
            ),
        }

    def get_available_blocks(self) -> Dict[str, List[str]]:
        """Get all available block categories and their blocks"""
        return self.block_categories.copy()

    def get_blocks_by_category(self, category: str) -> List[str]:
        """Get blocks for a specific category"""
        return self.block_categories.get(category, [])

    def get_all_block_names(self) -> List[str]:
        """Get flat list of all block names"""
        all_blocks = []
        for blocks_list in self.block_categories.values():
            all_blocks.extend(blocks_list)
        return list(set(all_blocks))  # Remove duplicates

    def get_block_function(self, block_name: str, precision: str):
        """Get the block factory function for given block and precision"""
        if not self.bindings_available:
            return None

        try:
            # Try sync version first (most common)
            sync_name = f"{block_name}_sync_{precision}"
            if hasattr(self.linalg_python, sync_name):
                return getattr(self.linalg_python, sync_name)

            # Try PDU version
            pdu_name = f"{block_name}_pdu_{precision}"
            if hasattr(self.linalg_python, pdu_name):
                return getattr(self.linalg_python, pdu_name)

            # Try without sync/pdu for special cases
            simple_name = f"{block_name}_{precision}"
            if hasattr(self.linalg_python, simple_name):
                return getattr(self.linalg_python, simple_name)

            return None

        except Exception as e:
            print(f"Error getting block function for {block_name}_{precision}: {e}")
            return None

    def create_block(self, block_name: str, precision: str, rows: int, cols: int):
        """Create a GNU Radio block instance"""
        block_func = self.get_block_function(block_name, precision)
        if not block_func:
            raise ValueError(f"Block function not found: {block_name}_{precision}")

        # Create block using appropriate method and parameters
        try:
            if block_name in ["matrix_add", "matrix_subtract"]:
                # These use .make() with shape and number of inputs
                block = block_func.make([rows, cols], 2)  # 2 inputs for add/subtract
            elif block_name in [
                "matrix_transpose",
                "matrix_determinant",
                "matrix_trace",
                "matrix_rank",
            ]:
                # These use constructor directly with shape
                block = block_func([rows, cols])
            elif block_name == "matrix_multiply":
                # Matrix multiply needs two shapes
                block = block_func.make([rows, cols], [cols, rows])
            elif block_name in ["vector_norm", "vector_normalize"]:
                # Vector operations with vector length and norm type
                block = block_func.make([max(rows, cols)], 2)  # L2 norm
            elif block_name == "vector_cross_product":
                # Cross product with bool parameter
                block = block_func(True)  # True for normalize
            elif block_name == "vector_angle":
                # Vector angle with shape and degrees flag
                block = block_func.make([max(rows, cols)], False)  # False for radians
            elif block_name == "dot_product":
                # Dot product with two vector shapes
                length = max(rows, cols)
                block = block_func.make([length], [length])
            elif block_name in ["eye", "ones", "zeros"]:
                # Matrix generators
                block = block_func.make([rows, cols])
            elif "decomp_" in block_name:
                # Decomposition blocks
                block = block_func([rows, cols])
            elif block_name in [
                "matrix_inverse",
                "matrix_norm",
                "matrix_condition_number",
                "matrix_pseudo_inverse",
                "matrix_hermitian",
                "matrix_exp",
            ]:
                # Advanced matrix operations with .make()
                block = block_func.make([rows, cols])
            elif block_name == "matrix_reshape":
                # Reshape with input and output shapes
                block = block_func.make([rows, cols], [rows * cols, 1])
            elif block_name == "matrix_solve":
                # Linear system solver
                if not hasattr(self.linalg_python, "solver_method"):
                    raise ValueError("solver_method enum not available")
                block = block_func.make(
                    [rows, cols], [rows, 1], self.linalg_python.solver_method.AUTO, 1e-6
                )
            elif block_name == "solve_least_squares":
                # Least squares solver
                if not hasattr(self.linalg_python, "least_squares_method"):
                    raise ValueError("least_squares_method enum not available")
                block = block_func.make(
                    [rows, cols],
                    [rows, 1],
                    self.linalg_python.least_squares_method.AUTO,
                    1e-6,
                )
            elif block_name == "solve_iterative":
                # Iterative solver
                block = block_func.make([rows, cols], [rows, 1])
            else:
                # Default: try with shape parameter
                block = block_func([rows, cols])

            return block

        except Exception as e:
            raise ValueError(f"Failed to create {block_name}_{precision}: {str(e)}")

    def get_vector_source_sink(self, precision: str, length: int):
        """Get appropriate vector source and sink for precision type"""
        if precision not in self.precision_types:
            raise ValueError(f"Unsupported precision type: {precision}")

        _, dtype, source_class, sink_class = self.precision_types[precision]
        return source_class, sink_class, dtype

    def create_test_data(self, precision: str, shape: Tuple[int, ...]) -> np.ndarray:
        """Create test data for given precision and shape"""
        if precision not in self.precision_types:
            raise ValueError(f"Unsupported precision type: {precision}")

        _, dtype, _, _ = self.precision_types[precision]

        # Create deterministic test data
        np.random.seed(42)  # Ensure reproducible results

        if dtype == np.complex64:
            # Complex data: real + imaginary parts
            real_part = np.random.uniform(0.1, 2.0, shape).astype(np.float32)
            imag_part = np.random.uniform(0.1, 2.0, shape).astype(np.float32)
            data = real_part + 1j * imag_part
        else:
            # Real data: uniform distribution
            data = np.random.uniform(0.1, 2.0, shape).astype(dtype)

        return data

    def is_vector_operation(self, block_name: str) -> bool:
        """Check if block operates on vectors rather than matrices"""
        vector_ops = [
            "vector_norm",
            "vector_normalize",
            "vector_cross_product",
            "vector_angle",
            "vector_outer_product",
            "dot_product",
            "vector_correlate",
        ]
        return block_name in vector_ops

    def requires_two_inputs(self, block_name: str) -> bool:
        """Check if block requires two input streams"""
        two_input_blocks = [
            "matrix_add",
            "matrix_subtract",
            "matrix_multiply",
            "vector_cross_product",
            "vector_angle",
            "dot_product",
            "vector_outer_product",
            "matrix_solve",
            "solve_least_squares",
            "solve_iterative",
            "matrix_kronecker_product",
        ]
        return block_name in two_input_blocks

    def is_source_block(self, block_name: str) -> bool:
        """Check if block is a source (no inputs)"""
        source_blocks = ["eye", "ones", "zeros", "matrix_source_const"]
        return block_name in source_blocks
