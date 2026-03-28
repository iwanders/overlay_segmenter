#include <cstdint>

// This is here to avoid valgrind from complaining and making debugging harder.
// https://github.com/pytorch/pytorch/blob/v2.11.0/torch/csrc/stable/stableivalue_conversions.h#L224-L266
// So for nullpointers, this is a nullpointer.
// For non nullpointers a 'new StableIValue' heap allocated u64 is created, and that pointer goes into the outer
// StableIValue, we hit a snag here... we're allocating with rust, but the c++ side is clearing it up.
// Okay, so this causes mismatched free() / delete warnings in valgrind, but it may not actually be an issue?
extern "C" {
uint64_t* iw_stable_torch_alloc_stableivalue() {
  return new uint64_t(0);
}
}
