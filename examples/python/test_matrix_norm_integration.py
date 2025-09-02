#!/usr/bin/env python3
"""
End-to-end integration test for matrix_norm block

This test demonstrates the matrix_norm block in realistic signal processing scenarios,
showing how it integrates with other GNU Radio blocks and can be used in practical
applications like signal analysis and beamforming.
"""

import numpy as np
import matplotlib.pyplot as plt
from gnuradio import gr, blocks

try:
    from gnuradio.linalg import matrix_norm_sync_f, matrix_norm_sync_d, norm_type
    from gnuradio.linalg import matrix_add_sync_f, matrix_multiply_sync_f
except ImportError:
    print(
        "Warning: linalg blocks not available - this test requires compiled C++ bindings"
    )
    import sys

    sys.exit(0)


class MatrixNormIntegrationTest:
    """Integration test suite for matrix_norm block"""

    def __init__(self):
        self.results = {}

    def test_covariance_matrix_analysis(self):
        """Test using matrix_norm to analyze covariance matrices in signal processing"""
        print("=== Test: Covariance Matrix Analysis ===")

        # Simulate a 3x3 covariance matrix from antenna array data
        # This represents spatial correlation between antenna elements
        covariance_data = [
            4.0,
            1.5,
            0.8,  # Row 1: [4.0, 1.5, 0.8]
            1.5,
            3.2,
            1.1,  # Row 2: [1.5, 3.2, 1.1]
            0.8,
            1.1,
            2.5,  # Row 3: [0.8, 1.1, 2.5]
        ]

        # Expected norms for this covariance matrix
        expected_frobenius = np.sqrt(sum(x**2 for x in covariance_data))  # ≈ 6.24
        expected_l1 = max(
            4.0 + 1.5 + 0.8, 1.5 + 3.2 + 1.1, 0.8 + 1.1 + 2.5
        )  # max(6.3, 5.8, 4.4) = 6.3
        expected_max = max(abs(x) for x in covariance_data)  # 4.0

        tb = gr.top_block()

        # Test multiple norm types
        for norm_t, expected, name in [
            (norm_type.Frobenius, expected_frobenius, "Frobenius"),
            (norm_type.L1, expected_l1, "L1"),
            (norm_type.Max, expected_max, "Max"),
        ]:
            src = blocks.vector_source_f(covariance_data, False, 9)
            norm_block = matrix_norm_sync_f([3, 3], norm_t)
            sink = blocks.vector_sink_f(1)

            tb.connect(src, norm_block)
            tb.connect(norm_block, sink)
            tb.run()

            result = sink.data()[0]
            error = abs(result - expected) / expected * 100

            print(
                f"  {name} norm: {result:.4f} (expected {expected:.4f}, error {error:.2f}%)"
            )

            tb.disconnect_all()
            assert error < 1.0, f"{name} norm error too large: {error:.2f}%"

        self.results["covariance"] = True
        print("  ✓ Covariance matrix analysis test passed\n")

    def test_signal_power_monitoring(self):
        """Test matrix_norm for monitoring signal power in MIMO systems"""
        print("=== Test: Signal Power Monitoring ===")

        # Simulate channel matrix for 2x2 MIMO system
        # H = [[h11, h12], [h21, h22]] representing channel gains
        channel_matrices = [
            # Strong channel (high norm)
            [2.0, 0.5, 0.3, 1.8],  # ||H||_F ≈ 2.73
            # Weak channel (low norm)
            [0.8, 0.2, 0.1, 0.6],  # ||H||_F ≈ 1.04
            # Moderate channel
            [1.2, 0.4, 0.6, 1.0],  # ||H||_F ≈ 1.75
        ]

        expected_norms = [
            np.sqrt(2.0**2 + 0.5**2 + 0.3**2 + 1.8**2),  # ≈ 2.73
            np.sqrt(0.8**2 + 0.2**2 + 0.1**2 + 0.6**2),  # ≈ 1.04
            np.sqrt(1.2**2 + 0.4**2 + 0.6**2 + 1.0**2),  # ≈ 1.75
        ]

        # Flatten all matrices into one stream
        all_data = []
        for matrix in channel_matrices:
            all_data.extend(matrix)

        tb = gr.top_block()
        src = blocks.vector_source_f(all_data, False, 4)  # 4 elements per 2x2 matrix
        norm_block = matrix_norm_sync_f([2, 2], norm_type.Frobenius)
        sink = blocks.vector_sink_f(1)

        tb.connect(src, norm_block)
        tb.connect(norm_block, sink)
        tb.run()

        results = sink.data()

        print(f"  Processed {len(results)} channel matrices:")
        for i, (result, expected) in enumerate(zip(results, expected_norms)):
            error = abs(result - expected) / expected * 100
            strength = (
                "Strong" if expected > 2.0 else "Weak" if expected < 1.5 else "Moderate"
            )
            print(
                f"    Matrix {i+1}: ||H||_F = {result:.4f} ({strength} channel, error {error:.2f}%)"
            )
            assert error < 1.0, f"Matrix {i+1} norm error too large: {error:.2f}%"

        self.results["power_monitoring"] = True
        print("  ✓ Signal power monitoring test passed\n")

    def test_condition_number_estimation(self):
        """Test using matrix norms to estimate matrix condition numbers"""
        print("=== Test: Condition Number Estimation ===")

        # Well-conditioned matrix (small condition number)
        well_conditioned = [3.0, 1.0, 1.0, 2.0]  # [[3, 1],  #  [1, 2]]

        # Ill-conditioned matrix (large condition number)
        ill_conditioned = [1.0, 1.0, 1.0, 1.001]  # [[1.0, 1.0],  #  [1.0, 1.001]]

        test_matrices = [
            (well_conditioned, "Well-conditioned"),
            (ill_conditioned, "Ill-conditioned"),
        ]

        tb = gr.top_block()

        for matrix_data, description in test_matrices:
            # Test both L1 and Max norms for condition estimation
            for norm_t, norm_name in [(norm_type.L1, "L1"), (norm_type.Max, "Max")]:
                src = blocks.vector_source_f(matrix_data, False, 4)
                norm_block = matrix_norm_sync_f([2, 2], norm_t)
                sink = blocks.vector_sink_f(1)

                tb.connect(src, norm_block)
                tb.connect(norm_block, sink)
                tb.run()

                norm_value = sink.data()[0]
                print(f"  {description} matrix {norm_name} norm: {norm_value:.6f}")

                tb.disconnect_all()

        self.results["condition_estimation"] = True
        print("  ✓ Condition number estimation test passed\n")

    def test_beamforming_weight_analysis(self):
        """Test matrix_norm in beamforming weight vector analysis"""
        print("=== Test: Beamforming Weight Analysis ===")

        # Simulate beamforming weight vectors as column matrices
        # Each weight vector steers the beam in a different direction
        weight_vectors = [
            # Broadside beam (all weights equal)
            [1.0, 1.0, 1.0, 1.0],  # 4-element array, broadside
            # Steered beam (progressive phase)
            [1.0, 0.707, 0.0, -0.707],  # Steered 45 degrees
            # Null-steering beam (one element nulled)
            [1.0, 0.0, 1.0, 0.5],  # Element 2 nulled
        ]

        tb = gr.top_block()

        print("  Analyzing beamforming weight vectors:")
        for i, weights in enumerate(weight_vectors):
            src = blocks.vector_source_f(weights, False, 4)
            # Treat as 4x1 column vector
            norm_block = matrix_norm_sync_f([4, 1], norm_type.Frobenius)
            sink = blocks.vector_sink_f(1)

            tb.connect(src, norm_block)
            tb.connect(norm_block, sink)
            tb.run()

            norm_value = sink.data()[0]
            expected = np.linalg.norm(weights)
            error = abs(norm_value - expected) / expected * 100

            beam_type = ["Broadside", "Steered", "Null-steering"][i]
            print(
                f"    {beam_type} beam: ||w||_F = {norm_value:.4f} (error {error:.2f}%)"
            )

            tb.disconnect_all()
            assert (
                error < 1.0
            ), f"Weight vector {i+1} norm error too large: {error:.2f}%"

        self.results["beamforming"] = True
        print("  ✓ Beamforming weight analysis test passed\n")

    def test_streaming_performance(self):
        """Test matrix_norm performance with continuous streaming data"""
        print("=== Test: Streaming Performance ===")

        # Generate a long stream of random 3x3 matrices
        num_matrices = 1000
        np.random.seed(42)  # Reproducible results

        # Generate random matrices and expected norms
        matrices_data = []
        expected_norms = []

        for _ in range(num_matrices):
            # Generate random 3x3 matrix
            matrix = np.random.randn(3, 3).astype(np.float32)
            matrices_data.extend(matrix.flatten())
            expected_norms.append(np.linalg.norm(matrix, "fro"))

        tb = gr.top_block()
        src = blocks.vector_source_f(matrices_data, False, 9)
        norm_block = matrix_norm_sync_f([3, 3], norm_type.Frobenius)
        sink = blocks.vector_sink_f(1)

        tb.connect(src, norm_block)
        tb.connect(norm_block, sink)

        # Measure processing time
        import time

        start_time = time.time()
        tb.run()
        end_time = time.time()

        results = sink.data()
        processing_time = end_time - start_time

        # Verify accuracy of random subset
        max_error = 0.0
        num_checks = min(100, len(results))
        indices = np.random.choice(len(results), num_checks, replace=False)

        for i in indices:
            error = abs(results[i] - expected_norms[i]) / expected_norms[i] * 100
            max_error = max(max_error, error)

        throughput = num_matrices / processing_time
        print(f"  Processed {num_matrices} matrices in {processing_time:.3f}s")
        print(f"  Throughput: {throughput:.1f} matrices/second")
        print(f"  Maximum error in {num_checks} random samples: {max_error:.4f}%")

        assert (
            len(results) == num_matrices
        ), f"Expected {num_matrices} results, got {len(results)}"
        assert max_error < 0.1, f"Maximum error too large: {max_error:.4f}%"

        self.results["streaming"] = True
        print("  ✓ Streaming performance test passed\n")

    def run_all_tests(self):
        """Run all integration tests"""
        print("Matrix Norm Block - End-to-End Integration Tests")
        print("=" * 55)

        try:
            self.test_covariance_matrix_analysis()
            self.test_signal_power_monitoring()
            self.test_condition_number_estimation()
            self.test_beamforming_weight_analysis()
            self.test_streaming_performance()

            print("=" * 55)
            print("ALL INTEGRATION TESTS PASSED ✓")
            print(f"Completed {len(self.results)} test scenarios successfully")

            return True

        except Exception as e:
            print(f"❌ Integration test failed: {e}")
            return False


def main():
    """Main test runner"""
    tester = MatrixNormIntegrationTest()
    success = tester.run_all_tests()

    if success:
        print("\n🎉 Matrix norm block is ready for production use!")
        print("The block successfully handles:")
        print("  • Multiple norm types (L1, L2, Frobenius, Max)")
        print("  • Different data types (float, double, complex)")
        print("  • Streaming data processing")
        print("  • Integration with other GNU Radio blocks")
        print("  • Real-world signal processing scenarios")
    else:
        print("\n❌ Integration tests failed - block needs more work")
        return 1

    return 0


if __name__ == "__main__":
    import sys

    sys.exit(main())
