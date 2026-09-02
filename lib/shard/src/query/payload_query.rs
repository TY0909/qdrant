use ordered_float::OrderedFloat;
use segment::common::operation_error::{OperationError, OperationResult};
use segment::data_types::query_context::PayloadTextIndexStats;
use segment::json_path::JsonPath;
use segment::types::{
    Filter, PayloadFieldSchema, PayloadSchemaParams, QueryTokenWeight, QueryTokenWeightSet,
};
use serde::Serialize;

pub type TextQueryTokenWeights = Vec<(String, OrderedFloat<f32>)>;

/// Collection-wide statistics required to execute a payload text query.
///
/// The token weights may be empty when all query tokens are removed by text
/// preprocessing. Keeping them inside this resolved wrapper distinguishes that
/// valid state from a query whose statistics have not been resolved yet.
#[derive(Clone, Debug, Eq, PartialEq, Hash, Serialize)]
pub struct ResolvedTextQuery {
    pub token_weights: TextQueryTokenWeights,
    pub average_document_length: Option<OrderedFloat<f64>>,
}

#[derive(Clone, Debug, Eq, PartialEq, Hash, Serialize)]
pub struct TextQueryInternal {
    pub key: JsonPath,
    pub query_str: String,
    /// Collection-wide query statistics supplied by the coordinating node.
    pub resolved: Option<ResolvedTextQuery>,
}

impl TextQueryInternal {
    pub fn resolved_query(&self) -> OperationResult<QueryTokenWeightSet> {
        let Some(resolved) = &self.resolved else {
            return Err(OperationError::service_error(
                "text query token weights were not resolved before segment search",
            ));
        };
        let tokens = resolved
            .token_weights
            .iter()
            .map(|(token, weight)| QueryTokenWeight::new(token.clone(), weight.into_inner()))
            .collect();
        let query = QueryTokenWeightSet::new(tokens);
        Ok(match resolved.average_document_length {
            Some(average) => query.with_average_document_length(average.into_inner()),
            None => query,
        })
    }
}

#[derive(Clone, Debug, Eq, PartialEq, Hash, Serialize)]
pub struct TextQueryStatsRequest {
    pub key: JsonPath,
    pub query_str: String,
    pub corpus: Option<Filter>,
}

#[derive(Clone, Debug, Default, Eq, PartialEq, Hash)]
pub struct TextQueryStats {
    pub doc_count: usize,
    pub sum_document_length: u64,
    pub tokens: Vec<String>,
    pub doc_frequencies: Vec<usize>,
}

impl From<PayloadTextIndexStats> for TextQueryStats {
    fn from(stats: PayloadTextIndexStats) -> Self {
        Self {
            doc_count: stats.document_count,
            sum_document_length: stats.sum_document_length,
            tokens: stats.tokens,
            doc_frequencies: stats.document_frequencies,
        }
    }
}

impl TextQueryStats {
    pub fn merge(&mut self, other: Self) -> OperationResult<()> {
        if !other.tokens.is_empty() {
            if self.tokens.is_empty() {
                self.tokens = other.tokens;
                self.doc_frequencies = other.doc_frequencies;
            } else if self.tokens == other.tokens {
                for (total, value) in self.doc_frequencies.iter_mut().zip(other.doc_frequencies) {
                    *total += value;
                }
            } else {
                return Err(OperationError::service_error(
                    "text tokenizer produced inconsistent tokens",
                ));
            }
        }
        self.doc_count += other.doc_count;
        self.sum_document_length += other.sum_document_length;
        Ok(())
    }

    pub fn into_query_parts(self) -> (TextQueryTokenWeights, Option<OrderedFloat<f64>>) {
        let weights = self
            .tokens
            .into_iter()
            .zip(self.doc_frequencies)
            .map(|(token, frequency)| (token, OrderedFloat(bm25_idf(self.doc_count, frequency))))
            .collect();
        let average_document_length = (self.doc_count != 0)
            .then(|| OrderedFloat(self.sum_document_length as f64 / self.doc_count as f64));
        (weights, average_document_length)
    }
}

pub fn bm25_idf(doc_count: usize, doc_frequency: usize) -> f32 {
    let doc_count = doc_count as f64;
    let doc_frequency = doc_frequency as f64;
    ((doc_count - doc_frequency + 0.5) / (doc_frequency + 0.5)).ln_1p() as f32
}

pub fn validate_text_query_schema(
    key: &JsonPath,
    schema: Option<&PayloadFieldSchema>,
) -> OperationResult<()> {
    let Some(schema) = schema else {
        return Err(OperationError::validation_error(format!(
            "payload field `{key}` is not indexed",
        )));
    };
    let expanded = schema.expand();
    let PayloadSchemaParams::Text(params) = expanded.as_ref() else {
        return Err(OperationError::validation_error(format!(
            "payload field `{key}` is not a full-text index",
        )));
    };
    if !params
        .bm25_config
        .as_ref()
        .is_some_and(|config| config.is_enabled())
    {
        return Err(OperationError::validation_error(format!(
            "BM25 scoring is not enabled for payload field `{key}`",
        )));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use segment::data_types::index::{TextIndexBm25Config, TextIndexParams};
    use segment::types::{PayloadSchemaParams, PayloadSchemaType};

    use super::*;

    #[test]
    fn text_query_schema_requires_bm25_full_text_index() {
        let key = JsonPath::new("text");

        assert!(validate_text_query_schema(&key, None).is_err());
        assert!(
            validate_text_query_schema(
                &key,
                Some(&PayloadFieldSchema::FieldType(PayloadSchemaType::Keyword)),
            )
            .is_err()
        );
        assert!(
            validate_text_query_schema(
                &key,
                Some(&PayloadFieldSchema::FieldType(PayloadSchemaType::Text)),
            )
            .is_err()
        );

        let schema = PayloadFieldSchema::FieldParams(PayloadSchemaParams::Text(TextIndexParams {
            bm25_config: Some(TextIndexBm25Config {
                enable: Some(true),
                k1: None,
                b: None,
            }),
            ..Default::default()
        }));
        validate_text_query_schema(&key, Some(&schema)).unwrap();
    }

    #[test]
    fn bm25_idf_stays_positive_for_large_ubiquitous_terms() {
        let doc_count = 1 << 23;
        assert!(bm25_idf(doc_count, doc_count) > 0.0);
    }

    #[test]
    fn unresolved_text_query_cannot_be_searched() {
        let query = TextQueryInternal {
            key: JsonPath::new("text"),
            query_str: "pending".to_string(),
            resolved: None,
        };

        assert!(query.resolved_query().is_err());
    }

    #[test]
    fn resolved_empty_text_query_is_valid() {
        let query = TextQueryInternal {
            key: JsonPath::new("text"),
            query_str: "v. w".to_string(),
            resolved: Some(ResolvedTextQuery {
                token_weights: Vec::new(),
                average_document_length: Some(OrderedFloat(3.5)),
            }),
        };

        let resolved = query.resolved_query().unwrap();
        assert!(resolved.query_tokens().is_empty());
        assert_eq!(resolved.average_document_length(), Some(3.5));
    }
}
