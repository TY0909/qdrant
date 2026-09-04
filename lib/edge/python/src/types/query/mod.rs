pub mod with_payload;
pub mod with_vector;

use std::fmt;

use bytemuck::{TransparentWrapper, TransparentWrapperAlloc as _};
use derive_more::Into;
use ordered_float::OrderedFloat;
use pyo3::prelude::*;
use segment::data_types::vectors::{NamedQuery, VectorInternal};
use segment::vector_storage::query::*;
use shard::query::payload_query::TextQueryInternal;
use shard::query::query_enum::QueryEnum;

pub use self::with_payload::*;
pub use self::with_vector::*;
use crate::repr::*;
use crate::types::*;

#[derive(Clone, Debug, Into, TransparentWrapper)]
#[repr(transparent)]
pub struct PyQuery(pub QueryEnum);

impl FromPyObject<'_, '_> for PyQuery {
    type Error = PyErr;

    fn extract(query: Borrowed<'_, '_, PyAny>) -> PyResult<Self> {
        Ok(Self::from(query.extract::<PyQueryInterface>()?))
    }
}

impl From<PyQueryInterface> for PyQuery {
    fn from(query: PyQueryInterface) -> Self {
        Self(match query {
            PyQueryInterface::Nearest { query, using } => QueryEnum::Nearest(NamedQuery {
                query: VectorInternal::from(query),
                using,
            }),

            PyQueryInterface::RecommendBestScore { query, using } => {
                QueryEnum::RecommendBestScore(NamedQuery {
                    query: RecoQuery::from(query),
                    using,
                })
            }

            PyQueryInterface::RecommendSumScores { query, using } => {
                QueryEnum::RecommendSumScores(NamedQuery {
                    query: RecoQuery::from(query),
                    using,
                })
            }

            PyQueryInterface::Discover { query, using } => QueryEnum::Discover(NamedQuery {
                query: DiscoverQuery::from(query),
                using,
            }),

            PyQueryInterface::Context { query, using } => QueryEnum::Context(NamedQuery {
                query: ContextQuery::from(query),
                using,
            }),

            PyQueryInterface::FeedbackNaive { query, using } => {
                QueryEnum::FeedbackNaive(NamedQuery {
                    query: NaiveFeedbackQuery::from(query),
                    using,
                })
            }

            PyQueryInterface::Text { key, query_str } => QueryEnum::Text(TextQueryInternal {
                key: key.0,
                query_str,
                resolved: None,
            }),
        })
    }
}

impl<'py> IntoPyObject<'py> for PyQuery {
    type Target = PyQueryInterface;
    type Output = Bound<'py, Self::Target>;
    type Error = PyErr; // Infallible?

    fn into_pyobject(self, py: Python<'py>) -> PyResult<Self::Output> {
        let query = match self.0 {
            QueryEnum::Nearest(NamedQuery { query, using }) => PyQueryInterface::Nearest {
                query: PyNamedVectorInternal(query),
                using,
            },

            QueryEnum::RecommendBestScore(NamedQuery { query, using }) => {
                PyQueryInterface::RecommendBestScore {
                    query: PyRecommendQuery(query),
                    using,
                }
            }

            QueryEnum::RecommendSumScores(NamedQuery { query, using }) => {
                PyQueryInterface::RecommendSumScores {
                    query: PyRecommendQuery(query),
                    using,
                }
            }

            QueryEnum::Discover(NamedQuery { query, using }) => PyQueryInterface::Discover {
                query: PyDiscoverQuery(query),
                using,
            },

            QueryEnum::Context(NamedQuery { query, using }) => PyQueryInterface::Context {
                query: PyContextQuery(query),
                using,
            },

            QueryEnum::FeedbackNaive(NamedQuery { query, using }) => {
                PyQueryInterface::FeedbackNaive {
                    query: PyFeedbackNaiveQuery(query),
                    using,
                }
            }

            QueryEnum::Text(TextQueryInternal {
                key,
                query_str,
                resolved: _,
            }) => PyQueryInterface::Text {
                key: PyJsonPath(key),
                query_str,
            },
        };

        Bound::new(py, query)
    }
}

impl<'py> IntoPyObject<'py> for &PyQuery {
    type Target = PyQueryInterface;
    type Output = Bound<'py, Self::Target>;
    type Error = PyErr; // Infallible

    fn into_pyobject(self, py: Python<'py>) -> PyResult<Self::Output> {
        IntoPyObject::into_pyobject(self.clone(), py)
    }
}

fn fmt_query_with_using(
    f: &mut Formatter<'_>,
    variant: &str,
    query: &dyn Repr,
    using: &Option<String>,
) -> fmt::Result {
    f.complex_enum::<PyQueryInterface>(variant, &[("query", query), ("using", using)])
}

impl Repr for PyQuery {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        match &self.0 {
            QueryEnum::Nearest(NamedQuery { query, using }) => {
                fmt_query_with_using(f, "Nearest", PyNamedVectorInternal::wrap_ref(query), using)
            }
            QueryEnum::RecommendBestScore(NamedQuery { query, using }) => fmt_query_with_using(
                f,
                "RecommendBestScore",
                PyRecommendQuery::wrap_ref(query),
                using,
            ),
            QueryEnum::RecommendSumScores(NamedQuery { query, using }) => fmt_query_with_using(
                f,
                "RecommendSumScores",
                PyRecommendQuery::wrap_ref(query),
                using,
            ),
            QueryEnum::Discover(NamedQuery { query, using }) => {
                fmt_query_with_using(f, "Discover", PyDiscoverQuery::wrap_ref(query), using)
            }
            QueryEnum::Context(NamedQuery { query, using }) => {
                fmt_query_with_using(f, "Context", PyContextQuery::wrap_ref(query), using)
            }
            QueryEnum::FeedbackNaive(NamedQuery { query, using }) => fmt_query_with_using(
                f,
                "FeedbackNaive",
                PyFeedbackNaiveQuery::wrap_ref(query),
                using,
            ),
            QueryEnum::Text(TextQueryInternal {
                key,
                query_str,
                resolved: _,
            }) => f.complex_enum::<PyQueryInterface>(
                "Text",
                &[("key", PyJsonPath::wrap_ref(key)), ("query_str", query_str)],
            ),
        }
    }
}

#[pyclass(name = "Query", from_py_object)]
#[derive(Clone, Debug)]
pub enum PyQueryInterface {
    #[pyo3(constructor = (query, using = None))]
    Nearest {
        query: PyNamedVectorInternal,
        using: Option<String>,
    },

    #[pyo3(constructor = (query, using = None))]
    RecommendBestScore {
        query: PyRecommendQuery,
        using: Option<String>,
    },

    #[pyo3(constructor = (query, using = None))]
    RecommendSumScores {
        query: PyRecommendQuery,
        using: Option<String>,
    },

    #[pyo3(constructor = (query, using = None))]
    Discover {
        query: PyDiscoverQuery,
        using: Option<String>,
    },

    #[pyo3(constructor = (query, using = None))]
    Context {
        query: PyContextQuery,
        using: Option<String>,
    },

    #[pyo3(constructor = (query, using = None))]
    FeedbackNaive {
        query: PyFeedbackNaiveQuery,
        using: Option<String>,
    },

    #[pyo3(constructor = (key, query_str))]
    Text { key: PyJsonPath, query_str: String },
}

#[pymethods]
impl PyQueryInterface {
    pub fn __repr__(&self) -> String {
        self.repr()
    }
}

impl Repr for PyQueryInterface {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        match self {
            PyQueryInterface::Nearest { query, using } => {
                fmt_query_with_using(f, "Nearest", query, using)
            }
            PyQueryInterface::RecommendBestScore { query, using } => {
                fmt_query_with_using(f, "RecommendBestScore", query, using)
            }
            PyQueryInterface::RecommendSumScores { query, using } => {
                fmt_query_with_using(f, "RecommendSumScores", query, using)
            }
            PyQueryInterface::Discover { query, using } => {
                fmt_query_with_using(f, "Discover", query, using)
            }
            PyQueryInterface::Context { query, using } => {
                fmt_query_with_using(f, "Context", query, using)
            }
            PyQueryInterface::FeedbackNaive { query, using } => {
                fmt_query_with_using(f, "FeedbackNaive", query, using)
            }
            PyQueryInterface::Text { key, query_str } => {
                f.complex_enum::<Self>("Text", &[("key", key), ("query_str", query_str)])
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use segment::json_path::JsonPath;

    use super::*;

    #[test]
    fn text_query_converts_to_unresolved_internal_query() {
        let key: JsonPath = "description".parse().expect("valid JSON path");
        let PyQuery(query) = PyQuery::from(PyQueryInterface::Text {
            key: PyJsonPath(key.clone()),
            query_str: "rust search".to_string(),
        });

        let QueryEnum::Text(query) = query else {
            panic!("expected text query");
        };
        assert_eq!(query.key, key);
        assert_eq!(query.query_str, "rust search");
        assert_eq!(query.resolved, None);
    }
}

#[pyclass(name = "RecommendQuery", from_py_object)]
#[derive(Clone, Debug, Into, TransparentWrapper)]
#[repr(transparent)]
pub struct PyRecommendQuery(RecoQuery<VectorInternal>);

#[pyclass_repr]
#[pymethods]
impl PyRecommendQuery {
    #[new]
    pub fn new(
        positives: Vec<PyNamedVectorInternal>,
        negatives: Vec<PyNamedVectorInternal>,
    ) -> Self {
        Self(RecoQuery {
            positives: PyNamedVectorInternal::peel_vec(positives),
            negatives: PyNamedVectorInternal::peel_vec(negatives),
        })
    }

    #[getter]
    pub fn positives(&self) -> &[PyNamedVectorInternal] {
        PyNamedVectorInternal::wrap_slice(&self.0.positives)
    }

    #[getter]
    pub fn negatives(&self) -> &[PyNamedVectorInternal] {
        PyNamedVectorInternal::wrap_slice(&self.0.negatives)
    }

    pub fn __repr__(&self) -> String {
        self.repr()
    }
}

impl PyRecommendQuery {
    fn _getters(self) {
        // Every field should have a getter method
        let RecoQuery {
            positives: _,
            negatives: _,
        } = self.0;
    }
}

#[pyclass(name = "DiscoverQuery", from_py_object)]
#[derive(Clone, Debug, Into, TransparentWrapper)]
#[repr(transparent)]
pub struct PyDiscoverQuery(DiscoverQuery<VectorInternal>);

#[pyclass_repr]
#[pymethods]
impl PyDiscoverQuery {
    #[new]
    pub fn new(target: PyNamedVectorInternal, pairs: Vec<PyContextPair>) -> Self {
        Self(DiscoverQuery {
            target: VectorInternal::from(target),
            pairs: PyContextPair::peel_vec(pairs),
        })
    }

    #[getter]
    pub fn target(&self) -> &PyNamedVectorInternal {
        PyNamedVectorInternal::wrap_ref(&self.0.target)
    }

    #[getter]
    pub fn pairs(&self) -> &[PyContextPair] {
        PyContextPair::wrap_slice(&self.0.pairs)
    }

    pub fn __repr__(&self) -> String {
        self.repr()
    }
}

impl PyDiscoverQuery {
    fn _getters(self) {
        // Every field should have a getter method
        let DiscoverQuery {
            target: _,
            pairs: _,
        } = self.0;
    }
}

#[pyclass(name = "ContextQuery", from_py_object)]
#[derive(Clone, Debug, Into, TransparentWrapper)]
#[repr(transparent)]
pub struct PyContextQuery(ContextQuery<VectorInternal>);

#[pyclass_repr]
#[pymethods]
impl PyContextQuery {
    #[new]
    pub fn new(pairs: Vec<PyContextPair>) -> Self {
        Self(ContextQuery {
            pairs: PyContextPair::peel_vec(pairs),
        })
    }

    #[getter]
    pub fn pairs(&self) -> &[PyContextPair] {
        PyContextPair::wrap_slice(&self.0.pairs)
    }

    pub fn __repr__(&self) -> String {
        self.repr()
    }
}

impl PyContextQuery {
    fn _getters(self) {
        // Every field should have a getter method
        let ContextQuery { pairs: _ } = self.0;
    }
}

#[pyclass(name = "ContextPair", from_py_object)]
#[derive(Clone, Debug, Into, TransparentWrapper)]
#[repr(transparent)]
pub struct PyContextPair(ContextPair<VectorInternal>);

#[pyclass_repr]
#[pymethods]
impl PyContextPair {
    #[new]
    pub fn new(positive: PyNamedVectorInternal, negative: PyNamedVectorInternal) -> Self {
        Self(ContextPair {
            positive: VectorInternal::from(positive),
            negative: VectorInternal::from(negative),
        })
    }

    #[getter]
    pub fn positive(&self) -> &PyNamedVectorInternal {
        PyNamedVectorInternal::wrap_ref(&self.0.positive)
    }

    #[getter]
    pub fn negative(&self) -> &PyNamedVectorInternal {
        PyNamedVectorInternal::wrap_ref(&self.0.negative)
    }

    pub fn __repr__(&self) -> String {
        self.repr()
    }
}

impl PyContextPair {
    fn _getters(self) {
        // Every field should have a getter method
        let ContextPair {
            positive: _,
            negative: _,
        } = self.0;
    }
}

impl<'py> IntoPyObject<'py> for &PyContextPair {
    type Target = PyContextPair;
    type Output = Bound<'py, Self::Target>;
    type Error = PyErr; // Infallible

    fn into_pyobject(self, py: Python<'py>) -> PyResult<Self::Output> {
        IntoPyObject::into_pyobject(self.clone(), py)
    }
}

#[pyclass(name = "FeedbackNaiveQuery", from_py_object)]
#[derive(Clone, Debug, Into, TransparentWrapper)]
#[repr(transparent)]
pub struct PyFeedbackNaiveQuery(NaiveFeedbackQuery<VectorInternal>);

#[pyclass_repr]
#[pymethods]
impl PyFeedbackNaiveQuery {
    #[new]
    pub fn new(
        target: PyNamedVectorInternal,
        feedback: Vec<PyFeedbackItem>,
        strategy: PyNaiveFeedbackCoefficients,
    ) -> Self {
        Self(NaiveFeedbackQuery {
            target: VectorInternal::from(target),
            feedback: PyFeedbackItem::peel_vec(feedback),
            coefficients: NaiveFeedbackCoefficients::from(strategy),
        })
    }

    #[getter]
    pub fn target(&self) -> &PyNamedVectorInternal {
        PyNamedVectorInternal::wrap_ref(&self.0.target)
    }

    #[getter]
    pub fn feedback(&self) -> &[PyFeedbackItem] {
        PyFeedbackItem::wrap_slice(&self.0.feedback)
    }

    #[getter]
    pub fn coefficients(&self) -> PyNaiveFeedbackCoefficients {
        PyNaiveFeedbackCoefficients(self.0.coefficients)
    }

    pub fn __repr__(&self) -> String {
        self.repr()
    }
}

impl PyFeedbackNaiveQuery {
    fn _getters(self) {
        // Every field should have a getter method
        let NaiveFeedbackQuery {
            target: _,
            feedback: _,
            coefficients: _,
        } = self.0;
    }
}

#[pyclass(name = "FeedbackItem", from_py_object)]
#[derive(Clone, Debug, Into, TransparentWrapper)]
#[repr(transparent)]
pub struct PyFeedbackItem(FeedbackItem<VectorInternal>);

#[pyclass_repr]
#[pymethods]
impl PyFeedbackItem {
    #[new]
    pub fn new(vector: PyNamedVectorInternal, score: f32) -> Self {
        Self(FeedbackItem {
            vector: VectorInternal::from(vector),
            score: OrderedFloat(score),
        })
    }

    #[getter]
    pub fn vector(&self) -> &PyNamedVectorInternal {
        PyNamedVectorInternal::wrap_ref(&self.0.vector)
    }

    #[getter]
    pub fn score(&self) -> f32 {
        self.0.score.into_inner()
    }

    pub fn __repr__(&self) -> String {
        self.repr()
    }
}

impl PyFeedbackItem {
    fn _getters(self) {
        // Every field should have a getter method
        let FeedbackItem {
            vector: _,
            score: _,
        } = self.0;
    }
}

impl<'py> IntoPyObject<'py> for &PyFeedbackItem {
    type Target = PyFeedbackItem;
    type Output = Bound<'py, Self::Target>;
    type Error = PyErr; // Infallible

    fn into_pyobject(self, py: Python<'py>) -> PyResult<Self::Output> {
        IntoPyObject::into_pyobject(self.clone(), py)
    }
}

#[pyclass(name = "NaiveFeedbackStrategy", from_py_object)]
#[derive(Copy, Clone, Debug, Into)]
pub struct PyNaiveFeedbackCoefficients(NaiveFeedbackCoefficients);

#[pyclass_repr]
#[pymethods]
impl PyNaiveFeedbackCoefficients {
    #[new]
    pub fn new(a: f32, b: f32, c: f32) -> Self {
        Self(NaiveFeedbackCoefficients {
            a: OrderedFloat(a),
            b: OrderedFloat(b),
            c: OrderedFloat(c),
        })
    }

    #[getter]
    pub fn a(&self) -> f32 {
        self.0.a.into_inner()
    }

    #[getter]
    pub fn b(&self) -> f32 {
        self.0.b.into_inner()
    }

    #[getter]
    pub fn c(&self) -> f32 {
        self.0.c.into_inner()
    }

    pub fn __repr__(&self) -> String {
        self.repr()
    }
}

impl PyNaiveFeedbackCoefficients {
    fn _getters(self) {
        // Every field should have a getter method
        let NaiveFeedbackCoefficients { a: _, b: _, c: _ } = self.0;
    }
}
