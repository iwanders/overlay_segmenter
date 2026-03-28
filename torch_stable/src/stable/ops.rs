use super::tensor::Tensor;
use crate::aoti_torch::*;
use crate::{StableTorchResult, unsafe_call_bail};

impl Tensor {
    // https://github.com/pytorch/pytorch/blob/f2b47323ac2c438722c2db58aa31d9222676509d/torch/csrc/stable/ops.h#L1037
    pub fn subtract(&self, other: &Tensor) -> StableTorchResult<Tensor> {
        let mut handle_res: AtenTensorHandle = std::ptr::null_mut();
        unsafe_call_bail!(aoti_torch_aten_subtract_Tensor(
            self.get(),
            other.get(),
            1.0,
            &mut handle_res
        ));
        Ok(Self::from_handle(handle_res))
    }
    // For some reason, addition is NOT a default op? >_<
}

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn test_tensor_ops_subtract() {
        use crate::contrib::{FromScalar, ToScalar};
        let a = Tensor::from_f32(5.0).unwrap();
        let b = Tensor::from_f32(3.0).unwrap();
        let c = a.subtract(&b).unwrap();
        assert_eq!(c.to_f32().unwrap(), 2.0);
    }
}
