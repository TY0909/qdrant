use segment::json_path::JsonPath;
use serde::Serialize;

#[derive(Clone, Debug, Eq, PartialEq, Hash, Serialize)]
pub enum PayloadQueryInternal {
    Text(TextQueryInternal),
}

#[derive(Clone, Debug, Eq, PartialEq, Hash, Serialize)]
pub struct TextQueryInternal {
    pub key: JsonPath,
    pub query_str: String,
}
