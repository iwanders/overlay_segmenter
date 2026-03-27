// https://github.com/pytorch/pytorch/blob/f2b47323ac2c438722c2db58aa31d9222676509d/torch/csrc/stable/tensor_struct.h
// https://github.com/pytorch/pytorch/blob/f2b47323ac2c438722c2db58aa31d9222676509d/torch/csrc/stable/tensor_inl.h

use super::device::DeviceIndex;
use crate::headeronly::core::Layout;

use crate::{StableTorchResult, unsafe_call_bail};
use anyhow::{anyhow, bail};
struct Tensor {}
