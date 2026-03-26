// https://github.com/pytorch/pytorch/tree/f2b47323ac2c438722c2db58aa31d9222676509d/torch/csrc/inductor/aoti_torch/c
mod macros;
mod shim;

pub use macros::*;
pub use shim::*;
