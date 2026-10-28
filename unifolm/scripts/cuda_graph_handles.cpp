// Diagnostic access to the graph handles in the installed, pinned Torch 2.3.1.
// A protected pointer-to-member is formed in a derived class and applied to
// the base object. No object-layout casts or changes to Torch are involved.
#include <pybind11/pybind11.h>
#include <ATen/cuda/CUDAGraph.h>
#include <cstdint>

struct GraphHandles : at::cuda::CUDAGraph {
  static pybind11::tuple get(const at::cuda::CUDAGraph& graph) {
    const auto raw = graph.*(&GraphHandles::graph_);
    const auto executable = graph.*(&GraphHandles::graph_exec_);
    const bool raw_alive = graph.*(&GraphHandles::has_graph_);
    const bool executable_alive = graph.*(&GraphHandles::has_graph_exec_);
    if (!raw_alive || !executable_alive || !raw || !executable) {
      throw std::runtime_error("Enable debug mode before capture; keep graph alive");
    }
    return pybind11::make_tuple(reinterpret_cast<std::uintptr_t>(raw),
                                reinterpret_cast<std::uintptr_t>(executable));
  }
};

PYBIND11_MODULE(TORCH_EXTENSION_NAME, module) {
  module.def("get_handles", &GraphHandles::get);
}
