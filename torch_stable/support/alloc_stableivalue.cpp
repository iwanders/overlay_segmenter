#include <cstdint>

// This is here to avoid valgrind from complaining and making debugging harder.
extern "C" {
uint64_t* iw_stable_torch_alloc_stableivalue() {
  return new uint64_t(0);
}
}
