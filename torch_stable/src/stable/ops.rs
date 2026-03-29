use super::device::Device;
use super::stableivalue_conversions::StableIValue;
use super::tensor::Tensor;
use crate::aoti_torch::*;
use crate::headeronly::core::{Layout, MemoryFormat, ScalarType};
use crate::stable::c::*;
use crate::{StableTorchResult, unsafe_call_bail, unsafe_call_dispatch_bail};

// Dispatch through?
// https://github.com/pytorch/pytorch/blob/v2.11.0/aten/src/ATen/native/native_functions.yaml
//

#[derive(Copy, Clone, Debug, Default)]
pub struct ToOptions {
    pub dtype: Option<ScalarType>,
    pub layout: Option<Layout>,
    pub device: Option<Device>,
    pub pin_memory: Option<bool>,
    pub memory_format: Option<MemoryFormat>,
    pub non_blocking: bool,
    pub copy: bool,
}

#[derive(Copy, Clone, Debug, Default)]
pub struct EmtpyOptions {
    pub dtype: Option<ScalarType>,
    pub layout: Option<Layout>,
    pub device: Option<Device>,
    pub pin_memory: Option<bool>,
    pub memory_format: Option<MemoryFormat>,
}
impl Tensor {
    // https://github.com/pytorch/pytorch/blob/v2.11.0/torch/csrc/stable/ops.h#L442
    pub fn unsqueeze(&self, dim: usize) -> StableTorchResult<Tensor> {
        let mut stack: [StableIValue; 2] = [self.into(), dim.into()];
        unsafe_call_dispatch_bail!("aten::unsqueeze", "", stack.as_mut_slice());
        stack[0].try_into()
    }

    // Lets try a simpler operation first.
    // https://github.com/pytorch/pytorch/blob/v2.11.0/torch/csrc/stable/ops.h#L531
    pub fn matmul(&self, other: &Tensor) -> StableTorchResult<Tensor> {
        let mut stack: [StableIValue; 2] = [self.into(), other.into()];
        unsafe_call_dispatch_bail!("aten::matmul", "", stack.as_mut_slice());
        stack[0].try_into()
    }

    // https://github.com/pytorch/pytorch/blob/v2.11.0/torch/csrc/stable/ops.h#L594-L628
    pub fn empty(dimensions: &[usize], options: &EmtpyOptions) -> StableTorchResult<Tensor> {
        let mut stack: [StableIValue; 6] = [
            (dimensions).into(),
            (&options.dtype).into(),
            (&options.layout).into(),
            (&options.device).into(),
            (&options.pin_memory).into(),
            (&options.memory_format).into(),
        ];
        unsafe_call_dispatch_bail!("aten::empty", "memory_format", stack.as_mut_slice());
        stack[0].try_into()
    }

    // https://github.com/pytorch/pytorch/blob/v2.11.0/torch/csrc/stable/ops.h#L824
    pub fn to(&self, options: &ToOptions) -> StableTorchResult<Tensor> {
        let mut stack: [StableIValue; 8] = [
            self.into(),
            (&options.dtype).into(),
            (&options.layout).into(),
            (&options.device).into(),
            (&options.pin_memory).into(),
            options.non_blocking.into(),
            options.copy.into(),
            (&options.memory_format).into(),
        ];
        unsafe_call_dispatch_bail!("aten::to", "dtype_layout", stack.as_mut_slice());
        stack[0].try_into()
    }

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
    #[test]
    fn test_tensor_ops_to() -> StableTorchResult<()> {
        use crate::contrib::{FromScalar, ToScalar};
        let a = Tensor::from_f32(5.0).unwrap();
        let a = a.unsqueeze(0)?;
        let b = a.to(&ToOptions {
            device: Some(Device::from_str("cpu")?),
            // device: Some(Device::from_str("cuda:0")?),
            copy: false,
            ..Default::default()
        })?;

        assert_eq!(
            b.device().device_type(),
            crate::stable::device::DeviceType::CPU
        );
        let res = b.to_f32();
        assert_eq!(res.unwrap(), 5.0);

        Ok(())
    }
    #[test]
    fn test_tensor_ops_matmul() -> StableTorchResult<()> {
        use crate::contrib::{FromScalar, ToScalar};
        let a = Tensor::from_f32(5.0)?;
        let b = Tensor::from_f32(3.0)?;
        let a = a.unsqueeze(0)?; // need a dimension to perform a matmul.
        let b = b.unsqueeze(0)?;
        assert_eq!(b.layout(), Layout::Strided);
        assert_eq!(b.layout(), Layout::Strided);
        let c = a.matmul(&b)?;
        let res = c.to_f32();
        assert_eq!(res.unwrap(), 15.0);
        Ok(())
    }
    #[test]
    fn test_tensor_ops_unsqueeze() -> StableTorchResult<()> {
        use crate::contrib::FromScalar;
        let a = Tensor::from_f32(5.0)?;
        let b = a.unsqueeze(0)?;
        assert_eq!(a.sizes(), &[]); // scalar is dimensionless.
        assert_eq!(b.sizes(), &[1]);

        Ok(())
    }

    #[test]
    fn test_tensor_ops_empty() -> StableTorchResult<()> {
        let b = Tensor::empty(&[3, 3], &Default::default())?;
        assert_eq!(b.dim(), 2);
        assert_eq!(b.sizes(), &[3, 3]);

        Ok(())
    }
}
