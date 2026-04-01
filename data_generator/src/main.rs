// struct Foo;

// trait FooF32 {
//     fn s(&self, v: f32);
// }
// impl FooF32 for Foo {
//     fn s(&self, v: f32) {
//         todo!()
//     }
// }
// trait FooF64 {
//     fn s(&self, v: f64);
// }
// impl FooF64 for Foo {
//     fn s(&self, v: f64) {
//         todo!()
//     }
// }

use torch_stable::{
    StableTorchResult,
    contrib::{DataManipulation, FromScalar},
    stable::{device::Device, ops::ToOptions, tensor::Tensor},
};
pub fn main() -> StableTorchResult<()> {
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
    // https://docs.rs/tch/latest/tch/struct.Tensor.html#method.data_ptr
    /*
    use tch::{Device, Kind, Layout, Tensor};
    let t = Tensor::from_slice(&[3u8; 100000]);
    println!("{:?}", t.data_ptr());
    // Send the tensor to the gpu
    let t = t.to_device_(Device::Cuda(0), Kind::Uint8, false, true);
    println!("cuda {:?}", t.data_ptr());

    println!("cuda {:?}", t.device());
    let tcpu = t.to_device_(Device::Cpu, Kind::Uint8, false, true);
    println!("back {:?}", tcpu.data_ptr());
    println!("back {:?}", tcpu.device());
    */

    // let f = Foo;
    // f.s(3.3f32);
    /*
     * 0x55ef651e4e00
     cuda 0x7f3a53a00000
     cuda Cuda(0)
     back 0x55ef65d4b2c0
     back Cpu

    */
}
