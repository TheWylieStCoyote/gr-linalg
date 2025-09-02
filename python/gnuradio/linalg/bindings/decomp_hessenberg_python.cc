/*
 * Copyright 2025 Free Software Foundation, Inc.
 *
 * This file is part of GNU Radio
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 *
 */

#include <pybind11/complex.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;

#include <gnuradio/linalg/decomp_hessenberg.h>
// Note: pydoc documentation disabled to avoid missing symbol errors

void bind_decomp_hessenberg(py::module &m) {

  using decomp_hessenberg_sync_f = gr::linalg::decomp_hessenberg_sync_f;
  using decomp_hessenberg_sync_d = gr::linalg::decomp_hessenberg_sync_d;
  using decomp_hessenberg_sync_cf = gr::linalg::decomp_hessenberg_sync_cf;
  using decomp_hessenberg_sync_cd = gr::linalg::decomp_hessenberg_sync_cd;

  using decomp_hessenberg_pdu_f = gr::linalg::decomp_hessenberg_pdu_f;
  using decomp_hessenberg_pdu_d = gr::linalg::decomp_hessenberg_pdu_d;
  using decomp_hessenberg_pdu_cf = gr::linalg::decomp_hessenberg_pdu_cf;
  using decomp_hessenberg_pdu_cd = gr::linalg::decomp_hessenberg_pdu_cd;

  // Sync blocks
  py::class_<decomp_hessenberg_sync_f, gr::sync_block, gr::block,
             gr::basic_block, std::shared_ptr<decomp_hessenberg_sync_f>>(
      m, "decomp_hessenberg_sync_f")
      .def(py::init([](const std::vector<int> &shape, bool compute_q) {
             return decomp_hessenberg_sync_f::make(
                 shape, compute_q, nullptr, gr::linalg::hessenberg_options{});
           }),
           py::arg("shape"), py::arg("compute_q") = true)
      .def(py::init(&decomp_hessenberg_sync_f::make), py::arg("shape"),
           py::arg("compute_q"), py::arg("algorithm"), py::arg("options"))
      .def_static("make", &decomp_hessenberg_sync_f::make, py::arg("shape"),
                  py::arg("compute_q") = true, py::arg("algorithm") = nullptr,
                  py::arg("options") = gr::linalg::hessenberg_options{});

  py::class_<decomp_hessenberg_sync_d, gr::sync_block, gr::block,
             gr::basic_block, std::shared_ptr<decomp_hessenberg_sync_d>>(
      m, "decomp_hessenberg_sync_d")
      .def(py::init([](const std::vector<int> &shape, bool compute_q) {
             return decomp_hessenberg_sync_d::make(
                 shape, compute_q, nullptr, gr::linalg::hessenberg_options{});
           }),
           py::arg("shape"), py::arg("compute_q") = true)
      .def(py::init(&decomp_hessenberg_sync_d::make), py::arg("shape"),
           py::arg("compute_q"), py::arg("algorithm"), py::arg("options"))
      .def_static("make", &decomp_hessenberg_sync_d::make, py::arg("shape"),
                  py::arg("compute_q") = true, py::arg("algorithm") = nullptr,
                  py::arg("options") = gr::linalg::hessenberg_options{});

  py::class_<decomp_hessenberg_sync_cf, gr::sync_block, gr::block,
             gr::basic_block, std::shared_ptr<decomp_hessenberg_sync_cf>>(
      m, "decomp_hessenberg_sync_cf")
      .def(py::init([](const std::vector<int> &shape, bool compute_q) {
             return decomp_hessenberg_sync_cf::make(
                 shape, compute_q, nullptr, gr::linalg::hessenberg_options{});
           }),
           py::arg("shape"), py::arg("compute_q") = true)
      .def(py::init(&decomp_hessenberg_sync_cf::make), py::arg("shape"),
           py::arg("compute_q"), py::arg("algorithm"), py::arg("options"))
      .def_static("make", &decomp_hessenberg_sync_cf::make, py::arg("shape"),
                  py::arg("compute_q") = true, py::arg("algorithm") = nullptr,
                  py::arg("options") = gr::linalg::hessenberg_options{});

  py::class_<decomp_hessenberg_sync_cd, gr::sync_block, gr::block,
             gr::basic_block, std::shared_ptr<decomp_hessenberg_sync_cd>>(
      m, "decomp_hessenberg_sync_cd")
      .def(py::init([](const std::vector<int> &shape, bool compute_q) {
             return decomp_hessenberg_sync_cd::make(
                 shape, compute_q, nullptr, gr::linalg::hessenberg_options{});
           }),
           py::arg("shape"), py::arg("compute_q") = true)
      .def(py::init(&decomp_hessenberg_sync_cd::make), py::arg("shape"),
           py::arg("compute_q"), py::arg("algorithm"), py::arg("options"))
      .def_static("make", &decomp_hessenberg_sync_cd::make, py::arg("shape"),
                  py::arg("compute_q") = true, py::arg("algorithm") = nullptr,
                  py::arg("options") = gr::linalg::hessenberg_options{});

  // PDU blocks
  py::class_<decomp_hessenberg_pdu_f, gr::block, gr::basic_block,
             std::shared_ptr<decomp_hessenberg_pdu_f>>(
      m, "decomp_hessenberg_pdu_f")
      .def_static("make", &decomp_hessenberg_pdu_f::make, py::arg("shape"),
                  py::arg("compute_q") = true, py::arg("algorithm") = nullptr,
                  py::arg("options") = gr::linalg::hessenberg_options{});

  py::class_<decomp_hessenberg_pdu_d, gr::block, gr::basic_block,
             std::shared_ptr<decomp_hessenberg_pdu_d>>(
      m, "decomp_hessenberg_pdu_d")
      .def_static("make", &decomp_hessenberg_pdu_d::make, py::arg("shape"),
                  py::arg("compute_q") = true, py::arg("algorithm") = nullptr,
                  py::arg("options") = gr::linalg::hessenberg_options{});

  py::class_<decomp_hessenberg_pdu_cf, gr::block, gr::basic_block,
             std::shared_ptr<decomp_hessenberg_pdu_cf>>(
      m, "decomp_hessenberg_pdu_cf")
      .def_static("make", &decomp_hessenberg_pdu_cf::make, py::arg("shape"),
                  py::arg("compute_q") = true, py::arg("algorithm") = nullptr,
                  py::arg("options") = gr::linalg::hessenberg_options{});

  py::class_<decomp_hessenberg_pdu_cd, gr::block, gr::basic_block,
             std::shared_ptr<decomp_hessenberg_pdu_cd>>(
      m, "decomp_hessenberg_pdu_cd")
      .def_static("make", &decomp_hessenberg_pdu_cd::make, py::arg("shape"),
                  py::arg("compute_q") = true, py::arg("algorithm") = nullptr,
                  py::arg("options") = gr::linalg::hessenberg_options{});
}