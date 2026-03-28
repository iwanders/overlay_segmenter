use super::device::Device;
use super::tensor::Tensor;
use crate::aoti_torch::*;
use crate::headeronly::core::{Layout, MemoryFormat, ScalarType};
use crate::{StableTorchResult, unsafe_call_bail};

#[derive(Copy, Clone, Debug, Default)]
struct ToOptions {
    pub dtype: Option<ScalarType>,
    pub layout: Option<Layout>,
    pub device: Option<Device>,
    pub pin_memory: Option<bool>,
    pub memory_format: Option<MemoryFormat>,
    pub non_blocking: bool,
    pub copy: bool,
}

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

    // https://github.com/pytorch/pytorch/blob/f2b47323ac2c438722c2db58aa31d9222676509d/torch/csrc/stable/ops.h#L824
    pub fn to(&self, options: &ToOptions) -> StableTorchResult<Tensor> {
        todo!("needs this call dispatch thing")
    }
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
    #[test]
    fn test_tensor_ops_to() -> StableTorchResult<()> {
        use crate::contrib::{FromScalar, ToScalar};
        let a = Tensor::from_f32(5.0).unwrap();
        a.to(&ToOptions {
            device: Some(Device::from_str("cuda")?),
            ..Default::default()
        })
        .unwrap();

        Ok(())
    }
}
