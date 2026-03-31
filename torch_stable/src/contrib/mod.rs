// This holds things that aren't actually part of the upstream stable API, but make sense.
// They're implemented as a trait, to make it clear it is opt in.

// Random notes:
// https://github.com/pytorch/pytorch/issues/34646
// https://github.com/pytorch/pytorch/issues/107112
// https://docs.pytorch.org/docs/stable/dlpack.html#torch.utils.dlpack.from_dlpack
//
// How do we dispatch?
// https://github.com/pytorch/pytorch/blob/main/aten/src/ATen/native/README.md
// https://github.com/pytorch/pytorch/wiki/PyTorch-dispatcher-walkthrough
// https://github.com/pytorch/pytorch/wiki/Codegen-and-Structured-Kernels
// https://blog.ezyang.com/2019/05/pytorch-internals/
//
// let mut stack: [StableIValue; 2] = [self.into(), other.into()];
// unsafe_call_dispatch_bail!("aten::add", "Tensor", stack.as_mut_slice()); // does something, mostly complain about scalars
// stack[0].try_into()
//

use anyhow::bail;

use crate::aoti_torch::AtenTensorHandle;
use crate::stable::scalar::Scalar;
use crate::stable::tensor::Tensor;
use crate::{StableTorchResult, unsafe_call_bail};
use crate::{aoti_torch::*, unsafe_call_dispatch_bail};

pub trait FromScalar {
    fn from_f32(value: f32) -> StableTorchResult<Tensor>;
    fn from_f64(value: f64) -> StableTorchResult<Tensor>;
}

impl FromScalar for Tensor {
    fn from_f32(value: f32) -> StableTorchResult<Self> {
        let mut handle_res: AtenTensorHandle = std::ptr::null_mut();
        unsafe_call_bail!(aoti_torch_scalar_to_tensor_float32(value, &mut handle_res));
        Ok(Tensor::from_handle(handle_res))
    }
    fn from_f64(value: f64) -> StableTorchResult<Self> {
        let mut handle_res: AtenTensorHandle = std::ptr::null_mut();
        unsafe_call_bail!(aoti_torch_scalar_to_tensor_float64(value, &mut handle_res));
        Ok(Tensor::from_handle(handle_res))
    }
}

pub trait ToScalar {
    fn to_f32(&self) -> StableTorchResult<f32>;
    fn to_f64(&self) -> StableTorchResult<f64>;
}
impl ToScalar for Tensor {
    fn to_f32(&self) -> StableTorchResult<f32> {
        let mut sum_result: f32 = 0.0;
        unsafe_call_bail!(aoti_torch_item_float32(self.get(), &mut sum_result));
        Ok(sum_result)
    }

    fn to_f64(&self) -> StableTorchResult<f64> {
        let mut sum_result: f64 = 0.0;
        unsafe_call_bail!(aoti_torch_item_float64(self.get(), &mut sum_result));
        Ok(sum_result)
    }
}

pub trait Math {
    fn add(&self, other: &Tensor) -> StableTorchResult<Tensor>;
    fn sub2(&self, other: &Tensor) -> StableTorchResult<Tensor>;
}
impl Math for Tensor {
    fn add(&self, other: &Tensor) -> StableTorchResult<Tensor> {
        if cfg!(feature = "use_torch_devel") {
            let mut stack: [StableIValue; 3] =
                [self.into(), other.into(), Scalar::from_f64(1.0).into()];
            unsafe_call_dispatch_bail!("aten::add", "Tensor", stack.as_mut_slice());
            stack[0].try_into()
        } else {
            let mut handle_res: AtenTensorHandle = std::ptr::null_mut();
            // How do we call a native function like https://github.com/pytorch/pytorch/blob/v2.11.0/aten/src/ATen/native/native_functions.yaml#L577C9-L577C16 ?
            // Yes, this is a subtract with self - alpha * other with alpha = -1.0.
            unsafe_call_bail!(aoti_torch_aten_subtract_Tensor(
                self.get(),
                other.get(),
                -1.0,
                &mut handle_res
            ));
            Ok(Self::from_handle(handle_res))
        }
    }
    fn sub2(&self, other: &Tensor) -> StableTorchResult<Tensor> {
        let mut stack: [StableIValue; 2] = [self.into(), other.into()];
        unsafe_call_dispatch_bail!("aten::subtract", "Tensor", stack.as_mut_slice()); // does something, mostly complain about scalars

        stack[0].try_into()
    }
}

pub trait Manipulation {
    fn data_ref(&self) -> StableTorchResult<&[u8]>;
    fn data_mut(&self) -> StableTorchResult<&mut [u8]>;
}

impl Manipulation for Tensor {
    fn data_ref(&self) -> StableTorchResult<&[u8]> {
        let element_size = self.element_size();
        let elements = self.numel();
        let data_ptr = self.const_data_ptr();
        if self.is_cpu() {
            Ok(unsafe { std::slice::from_raw_parts(data_ptr, elements * element_size) })
        } else {
            bail!("tensor must be on cpu to access slice")
        }
    }

    fn data_mut(&self) -> StableTorchResult<&mut [u8]> {
        let element_size = self.element_size();
        let elements = self.numel();
        let data_ptr = self.mutable_data_ptr();
        if self.is_cpu() {
            Ok(unsafe { std::slice::from_raw_parts_mut(data_ptr, elements * element_size) })
        } else {
            bail!("tensor must be on cpu to access slice")
        }
    }
}

#[cfg(test)]
mod test {
    use super::*;
    use crate::stable::ops::ToOptions;

    use crate::headeronly::core::ScalarType;

    use crate::stable::device::{Device, DeviceIndex};
    #[test]
    fn test_tensor_conbtrib_from_to_scalar() {
        use crate::contrib::FromScalar;
        use crate::contrib::ToScalar;
        let t = Tensor::from_f32(std::f32::consts::PI).unwrap();
        assert_eq!(t.dim(), 0);
        assert_eq!(t.numel(), 1);
        assert_eq!(t.sizes(), &[]);
        assert_eq!(t.strides(), &[]);
        assert_eq!(t.is_contiguous(), true);
        // assert_eq!(t.stride(0), 0); // no dimensions
        assert_eq!(t.get_device_index(), DeviceIndex(-1));
        assert_eq!(t.defined(), true);
        assert_eq!(t.scalar_type(), ScalarType::Float);
        assert_eq!(t.device(), Device::from_str("cpu").unwrap());
        assert_eq!(t.to_f32().unwrap(), std::f32::consts::PI);

        // and double

        let t = Tensor::from_f64(std::f64::consts::PI).unwrap();
        assert_eq!(t.scalar_type(), ScalarType::Double);
        assert_eq!(t.to_f64().unwrap(), std::f64::consts::PI);
    }

    #[test]
    fn test_tensor_contrib_addition() -> StableTorchResult<()> {
        use crate::contrib::{FromScalar, ToScalar};
        let a = Tensor::from_f32(5.0)?;
        let b = Tensor::from_f32(3.0)?;
        let c = a.add(&b)?;
        assert_eq!(c.to_f32().unwrap(), 8.0);
        Ok(())
    }

    #[test]
    fn test_tensor_contrib_sub2() -> StableTorchResult<()> {
        if false {
            use crate::contrib::{FromScalar, ToScalar};
            let a = Tensor::from_f32(5.0)?.unsqueeze(0)?;
            let b = Tensor::from_f32(3.0)?.unsqueeze(0)?;
            let c = a.sub2(&b).unwrap();
            assert_eq!(c.to_f32().unwrap(), 2.0);
        }
        Ok(())
    }

    #[test]
    fn test_tensor_contrib_data() -> StableTorchResult<()> {
        let a = Tensor::from_f32(5.0)?.unsqueeze(0)?;
        println!("a.element_size(): {:?}", a.element_size());
        println!("a.scalar_type(): {:?}", a.scalar_type());
        println!("a.data_ptr: {:?}", a.data_ptr());
        println!("a.const_data_ptr: {:?}", a.const_data_ptr());
        println!("a.mutable_data_ptr: {:?}", a.mutable_data_ptr());
        println!("a.data_mut(): {:?}", a.data_mut()?);
        a.data_mut()?[0] = 3;
        println!("a.data_ref(): {:?}", a.data_ref()?);
        assert_eq!(a.data_ref()?, &[3, 0, 160, 64]);

        #[cfg(not(feature = "use_torch_devel"))]
        {
            let b = a.to(&ToOptions {
                // device: Some(Device::from_str("cpu")?),
                device: Some(Device::from_str("cuda:0")?),
                copy: false,
                ..Default::default()
            })?;
            println!("b.data_ptr: {:?}", b.data_ptr());
            println!("b.const_data_ptr: {:?}", b.const_data_ptr());
            println!("b.mutable_data_ptr: {:?}", b.mutable_data_ptr());
            assert_eq!(b.data_mut().is_err(), true);
        }
        Ok(())
    }
}
