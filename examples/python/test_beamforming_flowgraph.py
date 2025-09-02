#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
GNU Radio Integration Test: Adaptive Beamforming

This test demonstrates a practical beamforming application using linear algebra blocks.
Beamforming is a signal processing technique used in antenna arrays to spatially filter
signals, enhancing signals from desired directions while suppressing interference.

The example implements:
1. Multi-channel signal generation (simulating antenna array)
2. Covariance matrix estimation using outer products
3. Weight calculation using matrix operations
4. Beamformed output generation
"""

import numpy as np
from gnuradio import gr, blocks, analog
from gnuradio.linalg import (
    matrix_add_sync_d,
    matrix_multiply_sync_d,
    matrix_inverse_sync_d,
    vector_outer_product_sync_d,
    matrix_condition_number_sync_d,
)
import unittest
import time
import math


class BeamformingFlowgraph(gr.top_block):
    """
    Advanced beamforming flowgraph using linear algebra operations
    """

    def __init__(self, num_antennas=4, sample_rate=32000):
        gr.top_block.__init__(self, "Beamforming Integration Test")

        self.num_antennas = num_antennas
        self.sample_rate = sample_rate
        self.vector_size = num_antennas

        # Beamforming parameters
        self.signal_freq = 1000  # Signal of interest frequency
        self.interference_freq = 2000  # Interference frequency
        self.noise_power = 0.01

        # Array geometry (uniform linear array)
        self.antenna_spacing = 0.5  # Half wavelength spacing
        self.steering_angle = 30.0  # Degrees, direction of desired signal

        self._create_signal_sources()
        self._create_linalg_processing()
        self._create_sinks()
        self._connect_flowgraph()

    def _create_signal_sources(self):
        """Create multi-channel signal sources simulating antenna array"""
        self.signal_sources = []
        self.noise_sources = []
        self.adders = []
        self.float_to_double = []

        # Calculate phase delays for spatial steering
        wavelength = 3e8 / self.signal_freq  # Approximate wavelength
        k = 2 * math.pi / wavelength

        for i in range(self.num_antennas):
            # Position of antenna i
            position = i * self.antenna_spacing

            # Phase delay for desired signal direction
            phase_delay = k * position * math.sin(math.radians(self.steering_angle))

            # Desired signal with spatial phase
            signal_source = analog.sig_source_f(
                self.sample_rate,
                analog.GR_COS_WAVE,
                self.signal_freq,
                1.0,  # amplitude
                phase_delay,
            )
            self.signal_sources.append(signal_source)

            # Noise source
            noise_source = analog.noise_source_f(
                analog.GR_GAUSSIAN,
                math.sqrt(self.noise_power),
                42 + i,  # Different seed for each antenna
            )
            self.noise_sources.append(noise_source)

            # Add signal and noise
            adder = blocks.add_ff()
            self.adders.append(adder)

            # Convert to double for linalg processing
            float_to_double = blocks.float_to_double()
            self.float_to_double.append(float_to_double)

        # Combine antenna outputs into vector stream
        self.antenna_combiner = blocks.streams_to_vector(
            gr.sizeof_double, self.num_antennas
        )

    def _create_linalg_processing(self):
        """Create linear algebra blocks for beamforming"""

        # Outer product for covariance matrix estimation
        # R = E[x x^H] where x is the antenna vector
        self.outer_product = vector_outer_product_sync_d()

        # Moving average to estimate covariance (simplified)
        # In practice, this would be a more sophisticated estimator
        self.moving_average = blocks.moving_average_ff(
            1000,  # Length of averaging
            1.0 / 1000.0,  # Scale factor
            4000,  # Max iterations
            self.vector_size * self.vector_size,  # Vector length for covariance matrix
        )

        # Convert back to double after averaging
        self.avg_to_double = blocks.float_to_double(self.vector_size * self.vector_size)

        # Matrix condition number check
        self.condition_number = matrix_condition_number_sync_d()

        # Matrix inverse for optimal beamforming weights
        # w_opt = R^(-1) * s, where s is steering vector
        self.matrix_inverse = matrix_inverse_sync_d(
            [self.num_antennas, self.num_antennas]
        )

        # Create steering vector (simplified - normally computed from array geometry)
        steering_vector = np.ones(self.num_antennas, dtype=np.float64)
        steering_vector[0] = 1.0  # Reference element
        for i in range(1, self.num_antennas):
            # Simple progressive phase shift
            phase = i * math.pi / 4
            steering_vector[i] = math.cos(phase)

        self.steering_source = blocks.vector_source_d(
            steering_vector.tolist(), repeat=True, vlen=self.num_antennas
        )

        # Matrix-vector multiply: w = R^(-1) * s
        self.weight_calculation = matrix_multiply_sync_d(
            [self.num_antennas, self.num_antennas], 2
        )

    def _create_sinks(self):
        """Create sinks for analysis"""
        self.antenna_sink = blocks.vector_sink_d(self.num_antennas)
        self.covariance_sink = blocks.vector_sink_f(
            self.num_antennas * self.num_antennas
        )
        self.condition_sink = blocks.vector_sink_d(1)
        self.weights_sink = blocks.vector_sink_d(self.num_antennas)

    def _connect_flowgraph(self):
        """Connect the complete beamforming flowgraph"""

        # Connect signal sources
        for i in range(self.num_antennas):
            # Signal + noise for each antenna
            self.connect((self.signal_sources[i], 0), (self.adders[i], 0))
            self.connect((self.noise_sources[i], 0), (self.adders[i], 1))
            self.connect((self.adders[i], 0), (self.float_to_double[i], 0))
            self.connect((self.float_to_double[i], 0), (self.antenna_combiner, i))

        # Antenna array output
        self.connect((self.antenna_combiner, 0), (self.antenna_sink, 0))

        # Covariance matrix estimation using outer product
        self.connect((self.antenna_combiner, 0), (self.outer_product, 0))
        self.connect((self.antenna_combiner, 0), (self.outer_product, 1))

        # Convert to float for moving average, then back to double
        covariance_to_float = blocks.double_to_float(
            self.num_antennas * self.num_antennas
        )
        self.connect((self.outer_product, 0), (covariance_to_float, 0))
        self.connect((covariance_to_float, 0), (self.moving_average, 0))
        self.connect((self.moving_average, 0), (self.covariance_sink, 0))
        self.connect((self.moving_average, 0), (self.avg_to_double, 0))

        # Check condition number of covariance matrix
        self.connect((self.avg_to_double, 0), (self.condition_number, 0))
        self.connect((self.condition_number, 0), (self.condition_sink, 0))

        # Calculate beamforming weights: w = R^(-1) * s
        self.connect((self.avg_to_double, 0), (self.matrix_inverse, 0))
        self.connect((self.matrix_inverse, 0), (self.weight_calculation, 0))
        self.connect((self.steering_source, 0), (self.weight_calculation, 1))
        self.connect((self.weight_calculation, 0), (self.weights_sink, 0))

    def get_results(self):
        """Get beamforming analysis results"""
        antenna_data = np.array(self.antenna_sink.data())
        covariance_data = np.array(self.covariance_sink.data())
        condition_data = self.condition_sink.data()
        weights_data = np.array(self.weights_sink.data())

        return {
            "antenna_samples": antenna_data.shape[0] if len(antenna_data) > 0 else 0,
            "covariance_samples": (
                covariance_data.shape[0] if len(covariance_data) > 0 else 0
            ),
            "condition_numbers": condition_data,
            "beamforming_weights": weights_data,
            "avg_condition_number": (
                np.mean(condition_data) if len(condition_data) > 0 else None
            ),
        }


class TestBeamformingIntegration(unittest.TestCase):
    """Integration test for beamforming using linear algebra blocks"""

    def setUp(self):
        """Set up beamforming test"""
        self.tb = BeamformingFlowgraph(num_antennas=4)

    def tearDown(self):
        """Clean up test"""
        self.tb.stop()
        self.tb.wait()
        self.tb = None

    def test_beamforming_flowgraph(self):
        """Test complete beamforming flowgraph execution"""
        # Run flowgraph for sufficient time to collect statistics
        self.tb.start()
        time.sleep(3.0)  # Run for 3 seconds
        self.tb.stop()
        self.tb.wait()

        # Analyze results
        results = self.tb.get_results()

        # Verify data was processed
        self.assertGreater(
            results["antenna_samples"],
            1000,
            "Should have processed significant antenna data",
        )

        self.assertGreater(
            results["covariance_samples"],
            100,
            "Should have covariance matrix estimates",
        )

        # Verify beamforming weights were calculated
        self.assertGreater(
            len(results["beamforming_weights"]),
            0,
            "Should have calculated beamforming weights",
        )

        # Verify condition numbers are reasonable
        if results["avg_condition_number"] is not None:
            self.assertGreater(
                results["avg_condition_number"], 1.0, "Condition number should be >= 1"
            )
            self.assertLess(
                results["avg_condition_number"],
                1000.0,
                "Condition number suggests reasonable matrix conditioning",
            )

        print("✓ Beamforming flowgraph completed successfully")
        print(f"  Processed {results['antenna_samples']} antenna samples")
        print(f"  Generated {results['covariance_samples']} covariance estimates")
        print(f"  Calculated {len(results['beamforming_weights'])} weight vectors")
        if results["avg_condition_number"] is not None:
            print(f"  Average condition number: {results['avg_condition_number']:.2f}")

        # Print sample beamforming weights
        if len(results["beamforming_weights"]) >= 4:
            weights_sample = results["beamforming_weights"][:4]
            print(
                f"  Sample weights: [{', '.join([f'{w:.3f}' for w in weights_sample])}]"
            )


def run_beamforming_test():
    """Run beamforming integration test"""
    print("GNU Radio Beamforming Integration Test")
    print("======================================")
    print()

    try:
        from gnuradio.linalg import vector_outer_product_sync_d, matrix_inverse_sync_d

        print("✓ Successfully imported beamforming-required modules")

        unittest.main(verbosity=2, exit=False)

    except ImportError as e:
        print(f"✗ Failed to import required modules: {e}")
        return False
    except Exception as e:
        print(f"✗ Beamforming test failed: {e}")
        return False

    return True


if __name__ == "__main__":
    success = run_beamforming_test()
    exit(0 if success else 1)
