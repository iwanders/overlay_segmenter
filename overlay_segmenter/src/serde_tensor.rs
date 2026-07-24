use flash_powder::Tensor;
use flash_powder::prelude::*;
use flash_powder_safetensors::prelude::*;
use serde::{
    Deserialize, Deserializer, Serialize, Serializer, de::Error as _, ser::Error as _,
    ser::SerializeSeq,
};
pub mod tensor {
    use super::*;

    // Serialize each element in the Vec<T>
    pub fn serialize<S>(data: &Tensor, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        let mut d = flash_powder::nn::StateDict::default();
        // How does this result in such a poor error?? Because postcard hides it.
        // https://github.com/jamesmunns/postcard/issues/38
        // let data = data.clone();
        let data = data.clone().cpu().map_err(|a| S::Error::custom(a))?;
        let value = flash_powder::nn::Data::Parameter(data);
        d.add_data("tensor", value)
            .map_err(|a| S::Error::custom(a))?;
        let e = d.serialize_safetensors();
        let t = e.map_err(|a| S::Error::custom(a))?;

        serde_bytes::serialize(&t, serializer)
    }

    // Deserialize sequence back into a Vec<T>
    pub fn deserialize<'de, D>(deserializer: D) -> Result<Tensor, D::Error>
    where
        D: Deserializer<'de>,
    {
        let bytes: Vec<u8> = serde_bytes::deserialize(deserializer)?;
        let s = flash_powder::nn::StateDict::deserialize_safetensors(&bytes, &Default::default())
            .map_err(|a| D::Error::custom(a))?;

        let t = s
            .as_map()
            .get("tensor")
            .ok_or(D::Error::custom("could not find tensor in safetensors"))?;

        Ok(t.as_tensor().map_err(|a| D::Error::custom(a))?.clone())
    }
}
pub mod vec_tensor {
    use super::*;

    #[derive(Deserialize, Serialize)]
    struct TensorWrapper(#[serde(with = "crate::serde_tensor::tensor")] Tensor);

    pub fn serialize<S>(data: &[Tensor], serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        let mut seq = serializer.serialize_seq(Some(data.len()))?;
        for item in data {
            seq.serialize_element(&TensorWrapper(item.clone()))?;
        }
        seq.end()
    }

    pub fn deserialize<'de, D>(deserializer: D) -> Result<Vec<Tensor>, D::Error>
    where
        D: Deserializer<'de>,
    {
        struct VecVisitor();

        impl<'de> serde::de::Visitor<'de> for VecVisitor {
            type Value = Vec<Tensor>;

            fn expecting(&self, formatter: &mut std::fmt::Formatter) -> std::fmt::Result {
                formatter.write_str("a sequence of items")
            }

            fn visit_seq<A>(self, mut seq: A) -> Result<Self::Value, A::Error>
            where
                A: serde::de::SeqAccess<'de>,
            {
                let mut values = Vec::new();
                while let Some(item) = seq.next_element::<TensorWrapper>()? {
                    values.push(item.0);
                }
                Ok(values)
            }
        }

        deserializer.deserialize_seq(VecVisitor())
    }
}

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn test_serde_tensor() -> Result<(), anyhow::Error> {
        let a_in: Tensor = [-3.0f32, -2.0, -1.0, 1.0, 2.0, 3.0].try_into()?;

        #[derive(Debug, Clone, Deserialize, Serialize)]
        struct Foo {
            #[serde(with = "crate::serde_tensor::tensor")]
            t: Tensor,
        }

        let foo = Foo { t: a_in };

        let v: String = serde_json::to_string(&foo)?;
        println!("String: {v:?}");

        let back: Foo = serde_json::de::from_str(&v)?;

        assert!(foo.t.is_equal(&back.t)?);

        let a_in: Tensor = [-3.5f32, -2.0, -1.3, 1.0, 5.0, 3.0].try_into()?;
        let b_in: Tensor = [-3.0f32, -3.0, -1.0, 1.0, 2.0, 3.0].try_into()?;
        #[derive(Debug, Clone, Deserialize, Serialize)]
        struct Bar {
            #[serde(with = "crate::serde_tensor::vec_tensor")]
            t: Vec<Tensor>,
        }

        let bar = Bar {
            t: vec![a_in, b_in],
        };
        let v: String = serde_json::to_string(&bar)?;
        println!("String: {v:?}");

        let back: Bar = serde_json::de::from_str(&v)?;
        assert_eq!(bar.t.len(), back.t.len());
        assert!(bar.t[0].is_equal(&back.t[0])?);
        assert!(bar.t[1].is_equal(&back.t[1])?);

        Ok(())
    }
}
