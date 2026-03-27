// https://github.com/pytorch/pytorch/blob/f2b47323ac2c438722c2db58aa31d9222676509d/torch/csrc/stable/tensor_struct.h
// https://github.com/pytorch/pytorch/blob/f2b47323ac2c438722c2db58aa31d9222676509d/torch/csrc/stable/tensor_inl.h

use super::device::DeviceIndex;
use crate::aoti_torch::{AtenTensorHandle, aoti_torch_delete_tensor_object};
use crate::headeronly::core::Layout;

use crate::aoti_torch::*;

use std::sync::Arc;

use crate::{StableTorchResult, unsafe_call_bail};
use anyhow::{anyhow, bail};

struct Tensordropper(AtenTensorHandle);
impl Drop for Tensordropper {
    fn drop(&mut self) {
        // We can't do anything with the return value here, so we quietly ignore it :/
        unsafe { aoti_torch_delete_tensor_object(self.0) };
    }
}

#[derive(Clone)]
pub struct Tensor {
    ath: Arc<Tensordropper>,
}
impl Tensor {
    /// Creates a new uninitialised tensor
    /// https://github.com/pytorch/pytorch/blob/f2b47323ac2c438722c2db58aa31d9222676509d/torch/csrc/stable/tensor_struct.h#L55
    pub fn new() -> StableTorchResult<Self> {
        let mut handle_res: AtenTensorHandle = std::ptr::null_mut();
        unsafe_call_bail!(aoti_torch_new_uninitialized_tensor(&mut handle_res));
        Ok(Self {
            ath: Arc::new(Tensordropper(handle_res)),
        })
    }
}

#[cfg(test)]
mod test {
    use super::*;
    #[test]
    fn test_tensor_uninitialised() {
        // Mostly to check if valgrind is clean with this test, to see if the dropping mechanism works, and it does.
        let t = Tensor::new().expect("should succeed");
    }
}
