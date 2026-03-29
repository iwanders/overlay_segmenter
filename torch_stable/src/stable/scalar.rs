use crate::aoti_torch::StableIValue;

use super::super::headeronly::core::ScalarType;
#[derive(Copy, Clone, Eq, PartialEq)]
pub struct Scalar {
    scalar_type: ScalarType,
    value: StableIValue,
}
impl Scalar {
    pub fn from_f64(value: f64) -> Self {
        Self {
            scalar_type: ScalarType::Double,
            value: StableIValue::from(value),
        }
    }
    pub fn scalar_type(&self) -> ScalarType {
        self.scalar_type
    }
    pub fn value(&self) -> StableIValue {
        self.value
    }
}
