// https://docs.pytorch.org/docs/stable/notes/libtorch_stable_abi.html
//
// Docs aren't great, the the stable c++ api is well documented:
// https://github.com/pytorch/pytorch/tree/main/torch/csrc/stable
//
// https://github.com/pytorch/pytorch/blob/f2b47323ac2c438722c2db58aa31d9222676509d/torch/csrc/inductor/aoti_torch/c/shim.h
// https://github.com/pytorch/pytorch/blob/f2b47323ac2c438722c2db58aa31d9222676509d/torch/csrc/inductor/aoti_torch/generated/c_shim_aten.h#L21

// https://github.com/pytorch/pytorch/blob/f2b47323ac2c438722c2db58aa31d9222676509d/torch/csrc/inductor/aoti_torch/c/macros.h#L46

// Hmm, the stable c++ interface actually goes through the call inteface; https://github.com/pytorch/pytorch/blob/e3d4dcc4adddff454ceeef71b3d735acc5e23344/torch/csrc/stable/ops.h#L625
//
//
// Bunch of tests here: https://github.com/pytorch/pytorch/tree/1c0fd99bc15e998eab3cee79588544a033f0e4df/test/cpp_extensions/libtorch_agn_2_10_extension/csrc
// empty is here: https://github.com/pytorch/pytorch/blob/1c0fd99bc15e998eab3cee79588544a033f0e4df/test/cpp_extensions/libtorch_agn_2_10_extension/csrc/my_empty.cpp#L10-L18
// Through; https://github.com/pytorch/pytorch/blob/1c0fd99bc15e998eab3cee79588544a033f0e4df/test/cpp_extensions/libtorch_agn_2_10_extension/csrc/my_empty.cpp#L10-L18

// The file structure follows the upstream torch repository.
// The root of this crate is https://github.com/pytorch/pytorch/tree/f2b47323ac2c438722c2db58aa31d9222676509d/torch/csrc

// ooh https://github.com/pytorch/extension-cpp/tree/1c325b202ae5e11de3cefb9a65be28f47949edd4
// they just pass a pytorch Tensor to a torch::stable::Tensor!? O_O

pub mod aoti_torch;
pub mod contrib;
pub mod headeronly;
pub mod stable;
mod support;
mod util;

use aoti_torch::*;
// Input param: AtenTensorHandle
// Output param: &mut AtenTensorHandle

pub use util::StableTorchResult;
pub(crate) use util::{unsafe_call_bail, unsafe_call_panic};

#[cfg(test)]
pub(crate) const RUN_SPAMMY_TESTS: bool = false;

pub const TORCH_ABI_VERSION: u64 = 0x20b000000000000; // as retrieved by _torch_abi_version, through test_aoti_torch_abi_version_print
