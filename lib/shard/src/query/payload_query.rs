use common::types::ScoreType;
use segment::json_path::JsonPath;
use segment::types::Filter;
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

/// Payload query request, used as a part of query request.
#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct QueryPayloadRequestInternal {
    /// Payload query definition.
    pub payload_query: PayloadQueryInternal,

    /// Look only for points which satisfy these conditions. If not provided - all points.
    pub filter: Option<Filter>,

    /// Keep only points with better score than this threshold.
    pub score_threshold: Option<ScoreType>,

    /// Max number of results.
    pub limit: usize,
}
