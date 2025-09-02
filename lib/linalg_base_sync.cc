#include <algorithm>
#include <boost/chrono.hpp>
#include <complex>
#include <gnuradio/block.h>
#include <gnuradio/io_signature.h>
#include <gnuradio/linalg/linalg_base_sync.h>
#include <gnuradio/linalg/types.h>
#include <gnuradio/linalg/utils.h>
#include <gnuradio/sync_block.h>
#include <gnuradio/tags.h>
#include <iostream>
#include <pmt/pmt.h>
#include <stdexcept>
#include <vector>

namespace gr {
namespace linalg {

// Helper to compute byte sizes vector<int> from vector_shapes without using
// `this`
namespace {
template <class Scalar>
std::vector<int>
make_itemsize_vec_from_shapes(const types::vector_shapes &shapes) {
  std::vector<int> v;
  v.reserve(shapes.size());
  for (const auto &s : shapes) {
    v.push_back(static_cast<int>(types::compute_size<Scalar>(s)));
  }
  return v;
}
} // namespace

template <class Scalar>
linalg_base_sync<Scalar>::linalg_base_sync(
    const std::string &name, const types::vector_shapes &shape_inputs,
    const std::vector<std::string> &input_names,
    const types::vector_shapes &shape_outputs,
    const std::vector<std::string> &output_names,
    const array_broadcast_type &broadcast_type,
    const ERROR_TAG &tag_errors, // Error tag type
    const ERROR_PDU &pdu_errors, // Error PDU type
    const gr::block::tag_propagation_policy_t tag_propagation_policy)
    : linalg_base<Scalar>(name, shape_inputs, input_names, shape_outputs,
                          output_names, broadcast_type, tag_errors, pdu_errors),
      gr::sync_block(
          name, // The name of the block
          shape_inputs.empty()
              ? io_signature::make(0, 0, 0)
              : io_signature::makev(
                    static_cast<int>(shape_inputs.size()),
                    static_cast<int>(shape_inputs.size()),
                    make_itemsize_vec_from_shapes<Scalar>(shape_inputs)),
          shape_outputs.empty()
              ? io_signature::make(0, 0, 0)
              : io_signature::makev(
                    static_cast<int>(shape_outputs.size()),
                    static_cast<int>(shape_outputs.size()),
                    make_itemsize_vec_from_shapes<Scalar>(shape_outputs))) {
  set_tag_propagation_policy(tag_propagation_policy);

  if (shape_inputs.empty() && shape_outputs.empty()) {
    throw std::invalid_argument("linalg_base_sync: both shape_inputs and "
                                "shape_outputs cannot be empty");
  }
  // check if any of the input or output shapes are empty
  if (std::any_of(shape_inputs.begin(), shape_inputs.end(),
                  [](const types::shape &s) { return s.empty(); })) {
    throw std::invalid_argument(
        "linalg_base_sync: shape_inputs must not contain empty shapes");
  }
  if (std::any_of(shape_outputs.begin(), shape_outputs.end(),
                  [](const types::shape &s) { return s.empty(); })) {
    throw std::invalid_argument(
        "linalg_base_sync: shape_outputs must not contain empty shapes");
  }
  if (std::any_of(shape_inputs.begin(), shape_inputs.end(),
                  [](const types::shape &s) { return s.size() > 2; })) {
    // Print warning that array broadcasting is not tested
    std::cerr << "[linalg_base_sync] Warning: shape_inputs contains shapes "
                 "with more than 2 dimensions, array broadcasting is not "
                 "tested and may not work as expected."
              << std::endl;
  }

  if (pdu_errors == ERROR_PDU::PDU) {
    // Register PDU out for error handling
    const pmt::pmt_t pdu_error_port = pmt::intern("Error");
    this->message_port_register_out(pdu_error_port);
  }
}

template <class Scalar>
linalg_base_sync<Scalar>::linalg_base_sync(
    const std::string &name, const types::vector_shapes &shape_inputs,
    const types::vector_shapes &shape_outputs)
    : linalg_base_sync(name, shape_inputs, std::vector<std::string>{},
                       shape_outputs, std::vector<std::string>{},
                       array_broadcast_type::NONE, error_tag_t::NONE,
                       error_pdu_p::NONE, gr::block::TPP_ALL_TO_ALL) {}

// Helper to get rows/cols from a shape
static inline std::pair<int, int> _rows_cols_from_shape(const types::shape &s) {
  if (s.empty())
    return {1, 0};
  if (s.size() == 1)
    return {1, s[0]};
  int rows = s[0];
  int cols = 1;
  for (size_t i = 1; i < s.size(); ++i)
    cols *= s[i];
  return {rows, cols};
}

template <class Scalar>
int linalg_base_sync<Scalar>::work(int noutput_items,
                                   gr_vector_const_void_star &input_items,
                                   gr_vector_void_star &output_items) {
  using types::const_matrix_map_dynamic;
  using types::matrix_map_dynamic;
  using types::vector_const_matrix_map;
  using types::vector_matrix_map;
  // TODO FIXME: This does not work and needs to be reworked
  const auto &in_shapes = this->shape_inputs;
  const auto &out_shapes = this->shape_outputs;

  // Number of ports
  const int nin = static_cast<int>(in_shapes.size());
  const int nout = static_cast<int>(out_shapes.size());

  // Precompute element counts per item
  std::vector<int> in_elems(nin, 0), out_elems(nout, 0);
  for (int p = 0; p < nin; ++p) {
    auto rc = _rows_cols_from_shape(in_shapes[p]);
    in_elems[p] = rc.first * rc.second;
  }
  for (int p = 0; p < nout; ++p) {
    auto rc = _rows_cols_from_shape(out_shapes[p]);
    out_elems[p] = rc.first * rc.second;
  }

  for (int i = 0; i < noutput_items; ++i) {
    // Build maps for this item across ports
    std::vector<const_matrix_map_dynamic<Scalar>> in_maps_storage;
    std::vector<matrix_map_dynamic<Scalar>> out_maps_storage;
    in_maps_storage.reserve(nin);
    out_maps_storage.reserve(nout);

    vector_const_matrix_map<Scalar> in_maps;
    vector_matrix_map<Scalar> out_maps;
    in_maps.reserve(nin);
    out_maps.reserve(nout);

    // Inputs
    for (int p = 0; p < nin; ++p) {
      auto rc = _rows_cols_from_shape(in_shapes[p]);
      auto ptr = static_cast<const Scalar *>(input_items[p]);
      const Scalar *offset_ptr = ptr + static_cast<size_t>(i) * in_elems[p];
      in_maps_storage.emplace_back(offset_ptr, rc.first, rc.second);
      in_maps.push_back(&in_maps_storage.back());
    }

    // Outputs
    for (int p = 0; p < nout; ++p) {
      auto rc = _rows_cols_from_shape(out_shapes[p]);
      auto ptr = static_cast<Scalar *>(output_items[p]);
      Scalar *offset_ptr = ptr + static_cast<size_t>(i) * out_elems[p];
      out_maps_storage.emplace_back(offset_ptr, rc.first, rc.second);
      out_maps.push_back(&out_maps_storage.back());
    }

    auto ret = this->operation(in_maps, out_maps);
    if (ret != OperationReturn::SUCCESS) {
      // Handle errors based on the error tag and PDU settings
      // TODO: This is a placeholder for error handling
      // Add Real messages and
      const auto msg = "linalg_base_sync::operation failed";
      const auto error_msg = pmt::intern(msg);
      auto error_dict = pmt::make_dict();
      // Add more information to the error dict if needed
      error_dict = pmt::dict_add(error_dict, pmt::intern("error"), error_msg);
      // Add index and operation type to the error dict, FIXME this should be
      // the running index not i but for now we use i as a placeholder to
      // indicate the operation that failed
      error_dict =
          pmt::dict_add(error_dict, pmt::intern("index"), pmt::from_long(i));
      // Add operation type to the error dict
      error_dict = pmt::dict_add(error_dict, pmt::intern("operation"),
                                 pmt::intern("linalg_base_sync"));
      // Add time of the error with fractional seconds
      // const boost::chrono::system_clock::time_point now =
      //     boost::chrono::system_clock::now();
      // const double seconds =
      //     boost::chrono::duration_cast<boost::chrono::duration<double>>(
      //         now.time_since_epoch())
      //         .count();
      // error_dict = pmt::dict_add(error_dict, pmt::intern("time"),
      //                            pmt::from_double(seconds));
      // Add tag errors if configured
      if (this->tag_errors == error_tag_t::TAGS) {
        // Create an error tag
        pmt::pmt_t error_tag =
            pmt::cons(pmt::intern("error"),
                      pmt::cons(pmt::intern("linalg_base_sync"), error_dict));
        // Add the tag to the output port
        // FIXME: Sign comparison warning - comparing unsigned int with const
        // int Problem: nout is const int but p is unsigned int, causing
        // -Wsign-compare warning Solution: Change p to int or cast nout to
        // unsigned int: static_cast<unsigned int>(nout)
        for (int p = 0; p < nout; ++p) {
          // Add the tag at the absolute output sample index for item i
          const uint64_t abs_offset = this->nitems_written(p) + i;
          this->add_item_tag(p, abs_offset, pmt::intern("Error"), error_tag);
        }
        // this->add_item_tag(0, error_tag, pmt::intern("Error"));
      }
      // Send a PDU if configured to do so
      if (this->pdu_errors == error_pdu_p::PDU) {
        // Create an error PDU
        pmt::pmt_t error_pdu =
            pmt::cons(pmt::intern("Error"),
                      pmt::cons(pmt::intern("linalg_base_sync"), error_dict));
        // Send the error PDU
        this->message_port_pub(pmt::intern("Error"), error_pdu);
        // this->pdu_errors For now, fail-fast on invalid operation
        // throw std::runtime_error("linalg_base_sync::operation failed");
      }
    }
  }
  return noutput_items;
}

// Explicit template instantiation for common types
template class linalg_base_sync<float>;
template class linalg_base_sync<double>;
template class linalg_base_sync<std::complex<float>>;
template class linalg_base_sync<std::complex<double>>;

} // namespace linalg
} // namespace gr