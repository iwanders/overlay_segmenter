pub fn main() {
    // https://docs.rs/tch/latest/tch/struct.Tensor.html#method.data_ptr
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

    /*
     * 0x55ef651e4e00
     cuda 0x7f3a53a00000
     cuda Cuda(0)
     back 0x55ef65d4b2c0
     back Cpu

    */
}
