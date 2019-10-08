#pragma once

#include <dnnl.hpp>

using namespace dnnl;

namespace at { namespace native {

// CpuEngine singleton
struct CpuEngine {
  static CpuEngine& Instance() {
    static CpuEngine myInstance;
    return myInstance;
  }
  engine& get_engine() {
    return _cpu_engine;
  }
  CpuEngine(CpuEngine const&) = delete;
  CpuEngine& operator=(CpuEngine const&) = delete;

protected:
  CpuEngine():_cpu_engine(dnnl::engine::kind::cpu, 0) {}
  ~CpuEngine() {}

private:
  engine _cpu_engine;
};

// Stream singleton
struct Stream {
  static Stream& Instance() {
    static thread_local Stream myInstance;
    return myInstance;
  };
  stream& get_stream() {
    return _cpu_stream;
  }
  Stream(Stream const&) = delete;
  Stream& operator=(Stream const&) = delete;

protected:
  // XPZ: correct ?
  Stream():_cpu_stream(CpuEngine::Instance().get_engine()) {}
  ~Stream() {}

private:
  stream _cpu_stream;
};

}}  // namespace at::native
