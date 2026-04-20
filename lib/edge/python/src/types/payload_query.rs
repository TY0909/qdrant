use std::fmt;

use bytemuck::TransparentWrapper;
use derive_more::Into;
use pyo3::prelude::*;
use shard::query::payload_query::{PayloadQueryInternal, TextQueryInternal};

use crate::repr::*;
use crate::types::*;

#[derive(Clone, Debug, Into, TransparentWrapper)]
#[repr(transparent)]
pub struct PyPayloadQuery(pub PayloadQueryInternal);

impl FromPyObject<'_, '_> for PyPayloadQuery {
    type Error = PyErr;

    fn extract(query: Borrowed<'_, '_, PyAny>) -> PyResult<Self> {
        let query = match query.extract()? {
            PyPayloadQueryInterface::Text { query } => {
                PayloadQueryInternal::Text(TextQueryInternal::from(query))
            }
        };

        Ok(Self(query))
    }
}

impl<'py> IntoPyObject<'py> for PyPayloadQuery {
    type Target = PyPayloadQueryInterface;
    type Output = Bound<'py, Self::Target>;
    type Error = PyErr;

    fn into_pyobject(self, py: Python<'py>) -> PyResult<Self::Output> {
        let query = match self.0 {
            PayloadQueryInternal::Text(query) => PyPayloadQueryInterface::Text {
                query: PyTextQuery(query),
            },
        };

        Bound::new(py, query)
    }
}

impl<'py> IntoPyObject<'py> for &PyPayloadQuery {
    type Target = PyPayloadQueryInterface;
    type Output = Bound<'py, Self::Target>;
    type Error = PyErr;

    fn into_pyobject(self, py: Python<'py>) -> PyResult<Self::Output> {
        IntoPyObject::into_pyobject(self.clone(), py)
    }
}

impl Repr for PyPayloadQuery {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        let (repr, query): (_, &dyn Repr) = match &self.0 {
            PayloadQueryInternal::Text(query) => ("Text", PyTextQuery::wrap_ref(query)),
        };

        f.complex_enum::<PyPayloadQueryInterface>(repr, &[("query", query)])
    }
}

#[pyclass(name = "PayloadQuery", from_py_object)]
#[derive(Clone, Debug)]
pub enum PyPayloadQueryInterface {
    #[pyo3(constructor = (query))]
    Text { query: PyTextQuery },
}

#[pymethods]
impl PyPayloadQueryInterface {
    pub fn __repr__(&self) -> String {
        self.repr()
    }
}

impl Repr for PyPayloadQueryInterface {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        let (repr, query): (_, &dyn Repr) = match self {
            PyPayloadQueryInterface::Text { query } => ("Text", query),
        };

        f.complex_enum::<Self>(repr, &[("query", query)])
    }
}

#[pyclass(name = "TextQuery", from_py_object)]
#[derive(Clone, Debug, Into, TransparentWrapper)]
#[repr(transparent)]
pub struct PyTextQuery(TextQueryInternal);

#[pyclass_repr]
#[pymethods]
impl PyTextQuery {
    #[new]
    pub fn new(key: PyJsonPath, query_str: String) -> Self {
        Self(TextQueryInternal {
            key: key.into(),
            query_str,
        })
    }

    #[getter]
    pub fn key(&self) -> &PyJsonPath {
        PyJsonPath::wrap_ref(&self.0.key)
    }

    #[getter]
    pub fn query_str(&self) -> &str {
        &self.0.query_str
    }

    pub fn __repr__(&self) -> String {
        self.repr()
    }
}

impl PyTextQuery {
    fn _getters(self) {
        let TextQueryInternal {
            key: _,
            query_str: _,
        } = self.0;
    }
}
