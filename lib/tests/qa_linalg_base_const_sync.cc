/* -*- c++ -*- */
/*
 * Copyright 2025 Wylie Standage-Beier.
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 */

#include "gnuradio/linalg/linalg_base.h"
#include <Eigen/Dense>
#include <boost/test/tools/old/interface.hpp>
#include <boost/test/unit_test.hpp>
#include <gnuradio/attributes.h>
#include <gnuradio/blocks/vector_sink.h>
#include <gnuradio/blocks/vector_source.h>
#include <gnuradio/linalg/linalg_base_const_sync.h>
#include <gnuradio/linalg/types.h>
#include <gnuradio/linalg/utils.h>
#include <gnuradio/top_block.h>
#include <memory>
#include <string>
#include <vector>

#include <gnuradio/io_signature.h>

namespace gr {
namespace linalg {

// Test class for matrix addition with constant matrix
class ConstSyncMatrixAdd : public linalg_base_const_sync<float> {
public:
  typedef std::shared_ptr<ConstSyncMatrixAdd> sptr;

  ConstSyncMatrixAdd(const std::string &name = "const_sync_matrix_add_test")
      : linalg_base<float>(name, {{2, 2}}, {"streaming_input"}, {{2, 2}},
                           {"output"}, array_broadcast_type::NONE,
                           error_tag_t::NONE, error_pdu_p::NONE),
        linalg_base_sync<float>(name, {{2, 2}}, {"streaming_input"}, {{2, 2}},
                                {"output"}, array_broadcast_type::NONE,
                                error_tag_t::NONE, error_pdu_p::NONE,
                                gr::block::TPP_ALL_TO_ALL),
        linalg_base_const_sync<float>(
            name, 1,                    // num_const_data
            {{1.0f, 2.0f, 3.0f, 4.0f}}, // const_data
            {{2, 2}, {2, 2}},           // Input shapes (streaming + constant)
            {"streaming_input", "constant_input"}, // Input names
            {{2, 2}},                              // Output shapes
            {"output"},                            // Output names
            const_tag_t::NONE,                     // tag_const
            const_pdu_t::NONE,                     // pdu_const
            array_broadcast_type::NONE,            // broadcast_type
            error_tag_t::NONE,                     // tag_errors
            error_pdu_p::NONE,                     // pdu_errors
            gr::block::TPP_ALL_TO_ALL              // tag_propagation_policy
        ) {}

  // Override work function to ensure linalg_base_const_sync::work() is called
  int work(int noutput_items, gr_vector_const_void_star &input_items,
           gr_vector_void_star &output_items) override {
    return linalg_base_const_sync<float>::work(noutput_items, input_items,
                                               output_items);
  }

  OperationReturn
  operation(types::vector_const_matrix_map<float> &input_matrices,
            types::vector_matrix_map<float> &output_matrices) override {

    if (input_matrices.size() < 2 || output_matrices.size() < 1) {
      return OperationReturn::INVALID_SHAPE;
    }

    // First matrix is streaming input, second is constant
    auto A = *input_matrices[0];  // Streaming input matrix
    auto B = *input_matrices[1];  // Constant matrix
    auto C = *output_matrices[0]; // Output matrix

    // Add streaming matrix and constant matrix
    C = A + B;
    return OperationReturn::SUCCESS;
  }

  static sptr make(const std::string &name = "const_sync_matrix_add_test") {
    return gnuradio::make_block_sptr<ConstSyncMatrixAdd>(name);
  }

  static types::vector_shapes
  compute_output_shapes(const types::vector_shapes &input_shapes) {
    if (input_shapes.empty()) {
      return {};
    }
    return {input_shapes[0]}; // Output shape same as first input
  }
};

// Test class with error conditions for tags/PDUs
class ConstSyncWithErrors : public linalg_base_const_sync<float> {
public:
  typedef std::shared_ptr<ConstSyncWithErrors> sptr;

  ConstSyncWithErrors(const std::string &name = "const_sync_error_test")
      : linalg_base<float>(name, {{2, 2}}, {"In"}, {{2, 2}}, {"Out"},
                           array_broadcast_type::NONE, error_tag_t::TAGS,
                           error_pdu_p::PDU),
        linalg_base_sync<float>(name, {{2, 2}}, {"In"}, {{2, 2}}, {"Out"},
                                array_broadcast_type::NONE, error_tag_t::TAGS,
                                error_pdu_p::PDU, gr::block::TPP_ALL_TO_ALL),
        linalg_base_const_sync<float>(
            name, 1, {{5.0f, 6.0f, 7.0f, 8.0f}}, // Constant data
            {{2, 2}, {2, 2}},  // Input shapes (streaming + constant)
            {"In", "const"},   // Input names
            {{2, 2}}, {"Out"}, // Output shapes and names
            const_tag_t::TAGS, const_pdu_t::NONE, array_broadcast_type::NONE,
            error_tag_t::TAGS, error_pdu_p::PDU, gr::block::TPP_ALL_TO_ALL) {}

  // Override work function to ensure linalg_base_const_sync::work() is called
  int work(int noutput_items, gr_vector_const_void_star &input_items,
           gr_vector_void_star &output_items) override {
    return linalg_base_const_sync<float>::work(noutput_items, input_items,
                                               output_items);
  }

  OperationReturn
  operation(types::vector_const_matrix_map<float> &input_matrices,
            types::vector_matrix_map<float> &output_matrices) override {

    if (input_matrices.size() < 2 || output_matrices.size() < 1) {
      return OperationReturn::INVALID_SHAPE;
    }

    auto A = *input_matrices[0];  // Streaming input
    auto B = *input_matrices[1];  // Constant matrix
    auto C = *output_matrices[0]; // Output

    float trigger = A(0, 0); // Use first element as error trigger

    if (trigger == -1.0f) {
      return OperationReturn::FAILURE;
    } else if (trigger == -2.0f) {
      return OperationReturn::NOT_IMPLEMENTED;
    } else if (trigger == -3.0f) {
      return OperationReturn::INVALID_SHAPE;
    } else if (trigger == -4.0f) {
      return OperationReturn::OUT_OF_RANGE;
    }

    // Normal operation: multiply by constant
    C = A.cwiseProduct(B); // Element-wise multiplication
    return OperationReturn::SUCCESS;
  }

  static sptr make(const std::string &name = "const_sync_error_test") {
    return gnuradio::make_block_sptr<ConstSyncWithErrors>(name);
  }

  static types::vector_shapes
  compute_output_shapes(const types::vector_shapes &input_shapes) {
    if (input_shapes.empty()) {
      return {};
    }
    return {input_shapes[0]};
  }
};

BOOST_AUTO_TEST_CASE(test_linalg_base_const_sync_constructor) {
  // Test that we can construct the base class
  std::vector<std::vector<float>> const_data = {{1.0f, 2.0f, 3.0f, 4.0f}};
  types::vector_shapes input_shapes = {{2, 2}, {2, 2}};
  types::vector_shapes output_shapes = {{2, 2}};

  auto block = ConstSyncMatrixAdd::make();
  BOOST_REQUIRE(block);

  // Verify block properties
  BOOST_CHECK_EQUAL(block->input_signature()->max_streams(),
                    1); // Only streaming input
  BOOST_CHECK_EQUAL(block->output_signature()->max_streams(), 1);
}

BOOST_AUTO_TEST_CASE(test_linalg_base_const_sync_operation) {
  auto block = ConstSyncMatrixAdd::make();
  BOOST_REQUIRE(block);

  // Create test matrices
  Eigen::MatrixXf streaming_input(2, 2);
  streaming_input << 10.0f, 20.0f, 30.0f, 40.0f;

  Eigen::MatrixXf constant_matrix(2, 2);
  constant_matrix << 1.0f, 2.0f, 3.0f, 4.0f; // Should match constructor

  Eigen::MatrixXf output_matrix(2, 2);

  // Create Eigen Maps
  Eigen::Map<const Eigen::MatrixXf> streaming_map(streaming_input.data(), 2, 2);
  Eigen::Map<const Eigen::MatrixXf> constant_map(constant_matrix.data(), 2, 2);
  Eigen::Map<Eigen::MatrixXf> output_map(output_matrix.data(), 2, 2);

  // Create input/output vectors
  types::vector_const_matrix_map<float> input_matrices;
  types::vector_matrix_map<float> output_matrices;

  input_matrices.push_back(&streaming_map);
  input_matrices.push_back(&constant_map);
  output_matrices.push_back(&output_map);

  // Call operation method
  auto result = block->operation(input_matrices, output_matrices);
  BOOST_CHECK(result == OperationReturn::SUCCESS);

  // Verify result is sum of streaming input and constant matrix
  Eigen::MatrixXf expected_result = streaming_input + constant_matrix;
  BOOST_CHECK(output_matrix.isApprox(expected_result, 1e-6));
}

BOOST_AUTO_TEST_CASE(test_linalg_base_const_sync_set_const_data) {
  auto block = ConstSyncMatrixAdd::make();
  BOOST_REQUIRE(block);

  // Test updating constant data
  std::vector<float> new_const_data = {100.0f, 200.0f, 300.0f, 400.0f};

  // Set new constant data for index 0
  block->set_const_data(0, new_const_data);

  // Test bulk update
  std::vector<std::vector<float>> bulk_const_data = {{5.0f, 6.0f, 7.0f, 8.0f}};
  block->set_const_data(bulk_const_data);

  BOOST_CHECK(true); // If we get here without exceptions, the test passes
}

BOOST_AUTO_TEST_CASE(test_linalg_base_const_sync_error_conditions) {
  auto block = ConstSyncWithErrors::make();
  BOOST_REQUIRE(block);

  // Test invalid index for set_const_data
  std::vector<float> test_data = {1.0f, 2.0f, 3.0f, 4.0f};
  BOOST_CHECK_THROW(block->set_const_data(10, test_data), std::out_of_range);

  // Test wrong size constant data
  std::vector<float> wrong_size_data = {1.0f, 2.0f}; // Too small for 2x2 matrix
  BOOST_CHECK_THROW(block->set_const_data(0, wrong_size_data),
                    std::runtime_error);

  // Test wrong count of constant data vectors
  std::vector<std::vector<float>> wrong_count = {
      {1.0f, 2.0f, 3.0f, 4.0f}, {5.0f, 6.0f, 7.0f, 8.0f}}; // Too many
  BOOST_CHECK_THROW(block->set_const_data(wrong_count), std::runtime_error);
}

BOOST_AUTO_TEST_CASE(test_linalg_base_const_sync_output_shapes) {
  types::vector_shapes input_shapes = {{2, 2}};
  auto output_shapes = ConstSyncMatrixAdd::compute_output_shapes(input_shapes);

  BOOST_CHECK_EQUAL(output_shapes.size(), 1);
  BOOST_CHECK_EQUAL(output_shapes[0][0], 2);
  BOOST_CHECK_EQUAL(output_shapes[0][1], 2);
}

BOOST_AUTO_TEST_CASE(test_linalg_base_const_sync_flowgraph) {
  // Test integration with GNU Radio flowgraph
  auto block = ConstSyncMatrixAdd::make();
  BOOST_REQUIRE(block);

  // Create test data - streaming input matrices
  std::vector<float> input_data = {
      10.0f, 20.0f,  30.0f,  40.0f, // First 2x2 matrix
      50.0f, 60.0f,  70.0f,  80.0f, // Second 2x2 matrix
      90.0f, 100.0f, 110.0f, 120.0f // Third 2x2 matrix
  };

  auto tb = gr::make_top_block("test_const_sync_flowgraph");
  auto src = gr::blocks::vector_source_f::make(input_data, false,
                                               4); // 4 elements per matrix
  auto sink = gr::blocks::vector_sink_f::make(4);

  tb->connect(src, 0, block, 0);
  tb->connect(block, 0, sink, 0);
  tb->run();

  // Verify output
  const auto &output_data = sink->data();
  BOOST_CHECK_EQUAL(output_data.size(), input_data.size());

  // Expected output: each input matrix + constant matrix [1,2,3,4]
  std::vector<float> expected_output = {
      11.0f, 22.0f,  33.0f,  44.0f, // [10,20,30,40] + [1,2,3,4]
      51.0f, 62.0f,  73.0f,  84.0f, // [50,60,70,80] + [1,2,3,4]
      91.0f, 102.0f, 113.0f, 124.0f // [90,100,110,120] + [1,2,3,4]
  };

  for (size_t i = 0; i < expected_output.size(); ++i) {
    BOOST_CHECK_CLOSE(output_data[i], expected_output[i], 1e-6);
  }
}

BOOST_AUTO_TEST_CASE(test_linalg_base_const_sync_error_tags) {
  // Test error tag generation
  auto block = ConstSyncWithErrors::make();
  BOOST_REQUIRE(block);

  std::vector<float> input_data = {
      1.0f,  2.0f, 3.0f, 4.0f, // Normal operation
      -1.0f, 2.0f, 3.0f, 4.0f, // FAILURE condition
      -2.0f, 2.0f, 3.0f, 4.0f, // NOT_IMPLEMENTED
      -3.0f, 2.0f, 3.0f, 4.0f, // INVALID_SHAPE
      -4.0f, 2.0f, 3.0f, 4.0f, // OUT_OF_RANGE
      5.0f,  6.0f, 7.0f, 8.0f  // Normal operation
  };

  auto tb = gr::make_top_block("test_const_sync_error_tags");
  auto src = gr::blocks::vector_source_f::make(input_data, false, 4);
  auto sink = gr::blocks::vector_sink_f::make(4);

  tb->connect(src, 0, block, 0);
  tb->connect(block, 0, sink, 0);
  tb->run();

  // Check that error tags were generated
  const auto &tag_list = sink->tags();
  BOOST_CHECK_EQUAL(tag_list.size(), 4); // 4 error conditions

  // Verify tags contain error information
  for (const auto &tag : tag_list) {
    BOOST_CHECK_EQUAL(tag.key, pmt::intern("Error"));
    BOOST_CHECK(!pmt::is_null(tag.value));
  }
}

} /* namespace linalg */
} /* namespace gr */
