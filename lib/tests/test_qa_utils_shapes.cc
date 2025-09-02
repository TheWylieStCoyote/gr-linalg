// lib/tests/test_qa_utils_shapes.cc
// NOTE: Skeleton test file for qa_utils_shapes utilities.
// Update these tests once the actual APIs in qa_utils_shapes are confirmed.
// If qa_utils_shapes.h does not yet exist, all tests will be skipped.

#include <boost/test/unit_test.hpp>

namespace gr {
namespace linalg {

BOOST_AUTO_TEST_CASE(test_qa_utils_shapes_product) {
  // Test product function with a simple vector
  std::vector<int> vec = {1, 2, 3, 4};
  int result = utils::product(vec);
  BOOST_CHECK_EQUAL(result, 24); // 1 * 2 * 3 * 4 = 24
}

} /* namespace linalg */
} /* namespace gr */
