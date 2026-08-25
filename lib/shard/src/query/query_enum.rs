use common::types::ScoreType;
use segment::data_types::vectors::*;
use segment::json_path::JsonPath;
use segment::types::{Distance, Order, VectorName};
use segment::vector_storage::query::*;
use serde::Serialize;
use sparse::common::sparse_vector::SparseVector;

use super::payload_query::TextQueryInternal;

/// Every scoring query that can be performed on segment level.
#[derive(Clone, Debug, PartialEq, Hash, Serialize)]
pub enum QueryEnum {
    Nearest(NamedQuery<VectorInternal>),
    RecommendBestScore(NamedQuery<RecoQuery<VectorInternal>>),
    RecommendSumScores(NamedQuery<RecoQuery<VectorInternal>>),
    Discover(NamedQuery<DiscoverQuery<VectorInternal>>),
    Context(NamedQuery<ContextQuery<VectorInternal>>),
    FeedbackNaive(NamedQuery<NaiveFeedbackQuery<VectorInternal>>),
    Text(TextQueryInternal),
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum QueryTarget<'a> {
    Vector(&'a VectorName),
    PayloadField(&'a JsonPath),
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ScoreSemantics<'a> {
    Distance(&'a VectorName),
    LargerBetter,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct QueryCapabilities<'a> {
    pub target: QueryTarget<'a>,
    pub score: ScoreSemantics<'a>,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum ResolvedScoreSemantics {
    Distance(Distance),
    LargerBetter,
}

impl ScoreSemantics<'_> {
    pub fn resolve<E>(
        self,
        get_distance: impl FnOnce(&VectorName) -> Result<Distance, E>,
    ) -> Result<ResolvedScoreSemantics, E> {
        match self {
            Self::Distance(vector_name) => {
                get_distance(vector_name).map(ResolvedScoreSemantics::Distance)
            }
            Self::LargerBetter => Ok(ResolvedScoreSemantics::LargerBetter),
        }
    }
}

impl ResolvedScoreSemantics {
    pub fn order(self) -> Order {
        match self {
            Self::Distance(distance) => distance.distance_order(),
            Self::LargerBetter => Order::LargeBetter,
        }
    }

    pub fn postprocess(self, score: ScoreType) -> ScoreType {
        match self {
            Self::Distance(distance) => distance.postprocess_score(score),
            Self::LargerBetter => score,
        }
    }

    pub fn passes_threshold(self, score: ScoreType, threshold: ScoreType) -> bool {
        match self {
            Self::Distance(distance) => distance.check_threshold(score, threshold),
            Self::LargerBetter => score >= threshold,
        }
    }

    pub fn is_ordered(self, left: ScoreType, right: ScoreType) -> bool {
        match self.order() {
            Order::LargeBetter => left >= right,
            Order::SmallBetter => left <= right,
        }
    }
}

impl QueryEnum {
    pub fn capabilities(&self) -> QueryCapabilities<'_> {
        let target = match self {
            QueryEnum::Nearest(query) => QueryTarget::Vector(query.get_name()),
            QueryEnum::RecommendBestScore(query) | QueryEnum::RecommendSumScores(query) => {
                QueryTarget::Vector(query.get_name())
            }
            QueryEnum::Discover(query) => QueryTarget::Vector(query.get_name()),
            QueryEnum::Context(query) => QueryTarget::Vector(query.get_name()),
            QueryEnum::FeedbackNaive(query) => QueryTarget::Vector(query.get_name()),
            QueryEnum::Text(query) => QueryTarget::PayloadField(&query.key),
        };
        let score = match self {
            QueryEnum::Nearest(query) => ScoreSemantics::Distance(query.get_name()),
            QueryEnum::RecommendBestScore(_)
            | QueryEnum::RecommendSumScores(_)
            | QueryEnum::Discover(_)
            | QueryEnum::Context(_)
            | QueryEnum::FeedbackNaive(_)
            | QueryEnum::Text(_) => ScoreSemantics::LargerBetter,
        };
        QueryCapabilities { target, score }
    }

    pub fn iterate_sparse(&self, mut f: impl FnMut(&VectorName, &SparseVector)) {
        match self {
            QueryEnum::Nearest(named) => match &named.query {
                VectorInternal::Sparse(sparse_vector) => f(named.get_name(), sparse_vector),
                VectorInternal::Dense(_) | VectorInternal::MultiDense(_) => {}
            },
            QueryEnum::RecommendBestScore(reco_query)
            | QueryEnum::RecommendSumScores(reco_query) => {
                let name = reco_query.get_name();
                for vector in reco_query.query.flat_iter() {
                    match vector {
                        VectorInternal::Sparse(sparse_vector) => f(name, sparse_vector),
                        VectorInternal::Dense(_) | VectorInternal::MultiDense(_) => {}
                    }
                }
            }
            QueryEnum::Discover(discover_query) => {
                let name = discover_query.get_name();
                for vector in discover_query.query.flat_iter() {
                    match vector {
                        VectorInternal::Sparse(sparse_vector) => f(name, sparse_vector),
                        VectorInternal::Dense(_) | VectorInternal::MultiDense(_) => {}
                    }
                }
            }
            QueryEnum::Context(context_query) => {
                let name = context_query.get_name();
                for vector in context_query.query.flat_iter() {
                    match vector {
                        VectorInternal::Sparse(sparse_vector) => f(name, sparse_vector),
                        VectorInternal::Dense(_) | VectorInternal::MultiDense(_) => {}
                    }
                }
            }
            QueryEnum::FeedbackNaive(feedback_query) => {
                let name = feedback_query.get_name();
                for vector in feedback_query.query.flat_iter() {
                    match vector {
                        VectorInternal::Sparse(sparse_vector) => f(name, sparse_vector),
                        VectorInternal::Dense(_) | VectorInternal::MultiDense(_) => {}
                    }
                }
            }
            QueryEnum::Text(_) => {}
        }
    }

    /// Returns the estimated cost of using this query in terms of number of vectors.
    /// The cost approximates how many similarity comparisons this query will make against one point.
    pub fn estimated_cost(&self) -> usize {
        match self {
            QueryEnum::Nearest(named_query) => search_cost([&named_query.query]),
            QueryEnum::RecommendBestScore(named_query) => {
                search_cost(named_query.query.flat_iter())
            }
            QueryEnum::RecommendSumScores(named_query) => {
                search_cost(named_query.query.flat_iter())
            }
            QueryEnum::Discover(named_query) => search_cost(named_query.query.flat_iter()),
            QueryEnum::Context(named_query) => search_cost(named_query.query.flat_iter()),
            QueryEnum::FeedbackNaive(named_query) => search_cost(named_query.query.flat_iter()),
            QueryEnum::Text(_) => 1,
        }
    }
}

fn search_cost<'a>(vectors: impl IntoIterator<Item = &'a VectorInternal>) -> usize {
    vectors
        .into_iter()
        .map(VectorInternal::similarity_cost)
        .sum()
}

impl AsRef<QueryEnum> for QueryEnum {
    fn as_ref(&self) -> &QueryEnum {
        self
    }
}

impl From<DenseVector> for QueryEnum {
    fn from(vector: DenseVector) -> Self {
        QueryEnum::Nearest(NamedQuery {
            query: VectorInternal::Dense(vector),
            using: None,
        })
    }
}

impl From<NamedQuery<DiscoverQuery<VectorInternal>>> for QueryEnum {
    fn from(query: NamedQuery<DiscoverQuery<VectorInternal>>) -> Self {
        QueryEnum::Discover(query)
    }
}

impl QueryEnum {
    pub fn into_query_vector(self) -> Option<QueryVector> {
        Some(match self {
            QueryEnum::Nearest(named) => QueryVector::Nearest(named.query),
            QueryEnum::RecommendBestScore(named) => QueryVector::RecommendBestScore(named.query),
            QueryEnum::RecommendSumScores(named) => QueryVector::RecommendSumScores(named.query),
            QueryEnum::Discover(named) => QueryVector::Discover(named.query),
            QueryEnum::Context(named) => QueryVector::Context(named.query),
            QueryEnum::FeedbackNaive(named) => QueryVector::FeedbackNaive(named.query),
            QueryEnum::Text(_) => return None,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn nearest_query_exposes_distance_capabilities() {
        let query = QueryEnum::from(vec![1.0, 2.0]);
        let capabilities = query.capabilities();

        assert!(matches!(capabilities.target, QueryTarget::Vector(_)));
        assert!(matches!(capabilities.score, ScoreSemantics::Distance(_)));
        assert!(query.estimated_cost() > 0);
    }

    #[test]
    fn resolved_score_semantics_own_score_behavior() {
        let distance = ResolvedScoreSemantics::Distance(Distance::Euclid);
        assert_eq!(distance.order(), Distance::Euclid.distance_order());
        assert_eq!(
            distance.postprocess(2.0),
            Distance::Euclid.postprocess_score(2.0),
        );
        assert_eq!(
            distance.passes_threshold(2.0, 3.0),
            Distance::Euclid.check_threshold(2.0, 3.0),
        );

        let larger_better = ResolvedScoreSemantics::LargerBetter;
        assert_eq!(larger_better.order(), Order::LargeBetter);
        assert_eq!(larger_better.postprocess(2.0), 2.0);
        assert!(larger_better.passes_threshold(3.0, 2.0));
        assert!(!larger_better.passes_threshold(1.0, 2.0));
    }
}
