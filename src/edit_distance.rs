use std::cmp;
use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};
use std::mem;

use pyo3::class::basic::CompareOp;
use pyo3::exceptions::PyValueError;
use pyo3::ffi::PyTypeObject;
use pyo3::once_cell::GILOnceCell;
use pyo3::prelude::{
    pyclass, pyfunction, pymethods, IntoPy, Py, PyAny, PyModule, PyObject, PyRef, PyResult, Python,
};
use pyo3::types::PyType;
use pyo3::AsPyPointer;

#[repr(transparent)]
pub struct Tensor(pyo3::PyAny);

impl Tensor {
    fn type_object_raw(py: Python) -> *mut PyTypeObject {
        static TYPE_OBJECT: GILOnceCell<Py<PyType>> = GILOnceCell::new();

        TYPE_OBJECT
            .get_or_init(py, || {
                py.import("torch")
                    .expect("Can not import torch module")
                    .getattr("Tensor")
                    .expect("Can not load Tensor class")
                    .extract()
                    .expect("Imported Tensor should be a type object")
            })
            .as_ptr() as *mut _
    }

    fn get_type(py: Python) -> &pyo3::types::PyType {
        unsafe { pyo3::types::PyType::from_type_ptr(py, Self::type_object_raw(py)) }
    }
}

#[pyclass]
#[derive(Clone, Copy, PartialEq, Eq, Hash)]
pub enum Action {
    #[pyo3(name = "INSERTION")]
    Insertion,
    #[pyo3(name = "DELETION")]
    Deletion,
    #[pyo3(name = "SUBSTITUTION")]
    Substitution,
}

#[pymethods]
impl Action {
    #[staticmethod]
    pub fn from_int(integer: usize) -> PyResult<Self> {
        Ok(match integer {
            0 => Self::Insertion,
            1 => Self::Deletion,
            2 => Self::Substitution,
            other => return Err(PyValueError::new_err(format!("Invalid enum value {other}"))),
        })
    }

    fn __hash__(&self) -> PyResult<u64> {
        let mut hasher = DefaultHasher::new();
        self.hash(&mut hasher);
        Ok(hasher.finish())
    }
}

#[pyfunction]
pub fn levensthein(
    py: Python,
    string_a: Vec<PyObject>,
    string_b: Vec<PyObject>,
) -> PyResult<usize> {
    let m = string_a.len();
    let n = string_b.len();
    let mut previous_row: Vec<usize> = (0..=n).collect();
    let mut current_row = vec![0; n + 1];

    for i in 0..m {
        current_row[0] = i + 1;

        for j in 0..n {
            let deletion = previous_row[j + 1] + 1;
            let insertion = current_row[j] + 1;
            let substitution =
                previous_row[j] + (string_a[i].as_ref(py).ne(&string_b[j].as_ref(py))? as usize);
            current_row[j + 1] = cmp::min(cmp::min(deletion, insertion), substitution);
        }

        mem::swap(&mut previous_row, &mut current_row);
    }

    Ok(previous_row[n])
}

type Operations = Vec<(Action, usize, usize)>;

#[pyfunction]
pub fn to_substitutions<'a>(
    string_a: Vec<&'a str>,
    string_b: Vec<&'a str>,
    operations: Operations,
) -> Vec<(Action, &'a str, &'a str)> {
    operations
        .into_iter()
        .map(|(operation, a_index, b_index)| match operation {
            Action::Deletion => (operation, string_a[a_index], ""),
            Action::Insertion => (operation, "", string_b[b_index]),
            Action::Substitution => (operation, string_a[a_index], string_b[b_index]),
        })
        .collect()
}

#[inline]
pub fn levensthein_operations_general(
    py: Python,
    string_a: Vec<PyObject>,
    string_b: Vec<PyObject>,
    deletion_cost: f32,
    mut weight_function: impl FnMut(
        Python,
        f32,
        f32,
        f32,
        &PyObject,
        &PyObject,
    ) -> PyResult<(f32, f32, f32)>,
) -> PyResult<(Operations, f32)> {
    let m = string_a.len();
    let n = string_b.len();
    // Compute full matrix - otherwise equivalent to optimized score solution
    let mut matrix: Vec<Vec<f32>> = vec![(0..=n).map(|x| x as f32).collect()];

    for i in 0..m {
        let previous_row = &mut matrix[i];
        let mut current_row = previous_row.clone();
        current_row[0] += deletion_cost;

        for j in 0..n {
            let (insertion, deletion, substitution) = weight_function(
                py,
                previous_row[j + 1],
                current_row[j],
                previous_row[j],
                &string_a[i],
                &string_b[j],
            )?;
            current_row[j + 1] = insertion.min(deletion).min(substitution);
        }

        matrix.push(current_row);
    }

    // First best path
    let mut best_path = Vec::new();
    let final_cost = matrix[m][n];
    let mut current_cost = final_cost;
    let mut current_coord = (m, n);

    while current_cost != 0. {
        let (i, j) = current_coord;
        let (operation, cost) = if i == 0 {
            if j == 0 {
                break;
            }
            (Some(Action::Insertion), matrix[i][j - 1])
        } else if j == 0 {
            (Some(Action::Deletion), matrix[i - 1][j])
        } else {
            let deletion = matrix[i - 1][j];
            let insertion = matrix[i][j - 1];
            let substitution = matrix[i - 1][j - 1];

            let (mut operation, mut cost) = if deletion < insertion {
                (Some(Action::Deletion), deletion)
            } else {
                (Some(Action::Insertion), insertion)
            };

            if substitution <= cost {
                if substitution == current_cost {
                    operation = None;
                } else {
                    operation = Some(Action::Substitution);
                }
                cost = substitution;
            }

            (operation, cost)
        };

        current_cost = cost;
        match operation {
            None => {
                current_coord = (current_coord.0 - 1, current_coord.1 - 1);
            }
            Some(Action::Deletion) => {
                current_coord.0 -= 1;
            }
            Some(Action::Insertion) => {
                current_coord.1 -= 1;
            }
            Some(Action::Substitution) => {
                current_coord = (current_coord.0 - 1, current_coord.1 - 1);
            }
        }

        if let Some(action) = operation {
            best_path.push((action, current_coord.0, current_coord.1));
        }
    }

    best_path.reverse();
    let final_cost = final_cost as f32;
    Ok((best_path, final_cost))
}

#[inline]
pub fn levensthein_matrix_general(
    py: Python,
    string_a: Vec<PyObject>,
    string_b: Vec<PyObject>,
    deletion_cost: f32,
    mut weight_function: impl FnMut(
        Python,
        f32,
        f32,
        f32,
        &PyObject,
        &PyObject,
    ) -> PyResult<(f32, f32, f32)>,
) -> PyResult<&PyAny> {
    let m = string_a.len();
    let n = string_b.len();
    // Compute full matrix - otherwise equivalent to optimized score solution
    let mut matrix: Vec<Vec<f32>> = vec![(0..=n).map(|x| x as f32).collect()];

    for i in 0..m {
        let previous_row = &mut matrix[i];
        let mut current_row = previous_row.clone();
        current_row[0] += deletion_cost;

        for j in 0..n {
            let (insertion, deletion, substitution) = weight_function(
                py,
                previous_row[j + 1],
                current_row[j],
                previous_row[j],
                &string_a[i],
                &string_b[j],
            )?;
            current_row[j + 1] = insertion.min(deletion).min(substitution);
        }

        matrix.push(current_row);
    }

    Tensor::get_type(py).call1((matrix,))
}

#[pyfunction]
pub fn levensthein_matrix(
    py: Python,
    string_a: Vec<PyObject>,
    string_b: Vec<PyObject>,
) -> PyResult<&PyAny> {
    levensthein_matrix_general(py, string_a, string_b, 1., uniform_costs)
}

#[pyfunction]
pub fn levensthein_operations(
    py: Python,
    string_a: Vec<PyObject>,
    string_b: Vec<PyObject>,
) -> PyResult<(Operations, f32)> {
    levensthein_operations_general(py, string_a, string_b, 1., uniform_costs)
}

#[pyclass]
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct EditStatistics {
    #[pyo3(get)]
    insertions: u64,
    #[pyo3(get)]
    deletions: u64,
    #[pyo3(get)]
    substitutions: u64,
    #[pyo3(get)]
    correct: u64,
}

#[pymethods]
impl EditStatistics {
    #[new]
    pub fn new(insertions: u64, deletions: u64, substitutions: u64, correct: u64) -> Self {
        Self {
            insertions,
            deletions,
            substitutions,
            correct,
        }
    }

    #[classmethod]
    pub fn zeros(cls: &PyType) -> PyResult<&PyAny> {
        cls.call1((0, 0, 0, 0))
    }

    pub fn word_error_rate(&self) -> f32 {
        let substituted_or_deleted = (self.substitutions + self.deletions) as f32;
        // (S + D + I) / N
        // = (S + D + I) / (S + D + C)
        (substituted_or_deleted + (self.insertions as f32))
            / (substituted_or_deleted + (self.correct as f32))
    }

    fn expected_count(&self) -> f32 {
        (self.substitutions + self.deletions + self.correct) as f32
    }

    pub fn substitution_rate(&self) -> f32 {
        (self.substitutions as f32) / self.expected_count()
    }

    pub fn insertion_rate(&self) -> f32 {
        (self.insertions as f32) / self.expected_count()
    }

    pub fn deletion_rate(&self) -> f32 {
        (self.deletions as f32) / self.expected_count()
    }

    fn __richcmp__(&self, other: PyRef<EditStatistics>, op: CompareOp) -> PyObject {
        match op {
            CompareOp::Eq => self.eq(&*other).into_py(other.py()),
            _ => other.py().NotImplemented(),
        }
    }

    fn __hash__(&self) -> u64 {
        let mut hasher = DefaultHasher::new();
        self.hash(&mut hasher);
        hasher.finish()
    }

    fn __repr__(&self) -> PyResult<String> {
        Ok(format!(
            "EditStatistics(insertions={}, deletions={}, substitutions={}, correct={})",
            self.insertions, self.deletions, self.substitutions, self.correct,
        ))
    }

    fn __add__(&self, rhs: EditStatistics) -> PyResult<EditStatistics> {
        Ok(EditStatistics::new(
            self.insertions + rhs.insertions,
            self.deletions + rhs.deletions,
            self.substitutions + rhs.substitutions,
            self.correct + rhs.correct,
        ))
    }

    fn __iadd__(&mut self, other: EditStatistics) {
        self.insertions += other.insertions;
        self.deletions += other.deletions;
        self.substitutions += other.substitutions;
        self.correct += other.correct;
    }
}

fn levensthein_statistics_general(
    py: Python,
    string_a: Vec<PyObject>,
    string_b: Vec<PyObject>,
    deletion_cost: f32,
    mut weight_function: impl FnMut(
        Python,
        f32,
        f32,
        f32,
        &PyObject,
        &PyObject,
    ) -> PyResult<(f32, f32, f32)>,
) -> PyResult<EditStatistics> {
    let m = string_a.len();
    let n = string_b.len();
    // Compute full matrix - otherwise equivalent to optimized score solution
    let mut matrix: Vec<Vec<f32>> = vec![(0..=n).map(|i| i as f32).collect()];

    for i in 0..m {
        let previous_row = &mut matrix[i];
        let mut current_row = previous_row.clone();
        current_row[0] += deletion_cost;

        for j in 0..n {
            let (insertion, deletion, substitution) = weight_function(
                py,
                previous_row[j + 1],
                current_row[j],
                previous_row[j],
                &string_a[i],
                &string_b[j],
            )?;
            current_row[j + 1] = insertion.min(deletion).min(substitution);
        }

        matrix.push(current_row);
    }

    // First best path
    let final_cost = matrix[m][n];
    let mut current_cost = final_cost;
    let mut current_coord = (m, n);
    let mut insertions = 0;
    let mut deletions = 0;
    let mut substitutions = 0;
    let mut correct = 0;

    while current_cost != 0. {
        let (i, j) = current_coord;
        let (operation, cost) = if i == 0 {
            if j == 0 {
                break;
            }
            (Some(Action::Insertion), matrix[i][j - 1])
        } else if j == 0 {
            (Some(Action::Deletion), matrix[i - 1][j])
        } else {
            let deletion = matrix[i - 1][j];
            let insertion = matrix[i][j - 1];
            let substitution = matrix[i - 1][j - 1];

            let (mut operation, mut cost) = if deletion < insertion {
                (Some(Action::Deletion), deletion)
            } else {
                (Some(Action::Insertion), insertion)
            };

            if substitution <= cost {
                if substitution == current_cost {
                    operation = None;
                } else {
                    operation = Some(Action::Substitution);
                }
                cost = substitution;
            }

            (operation, cost)
        };

        current_cost = cost;
        match operation {
            None => {
                current_coord = (current_coord.0 - 1, current_coord.1 - 1);
                correct += 1;
            }
            Some(Action::Deletion) => {
                current_coord.0 -= 1;
                deletions += 1;
            }
            Some(Action::Insertion) => {
                current_coord.1 -= 1;
                insertions += 1;
            }
            Some(Action::Substitution) => {
                current_coord = (current_coord.0 - 1, current_coord.1 - 1);
                substitutions += 1;
            }
        }
    }

    // Add remaining characters of the source string as correct if the cost is 0
    correct += current_coord.0 as u64;
    Ok(EditStatistics::new(
        insertions,
        deletions,
        substitutions,
        correct,
    ))
}

fn uniform_costs(
    py: Python,
    above_cost: f32,
    left_cost: f32,
    upper_left_cost: f32,
    current: &PyObject,
    expected: &PyObject,
) -> PyResult<(f32, f32, f32)> {
    let deletion = above_cost + 1.;
    let insertion = left_cost + 1.;
    let substitution =
        upper_left_cost + (current.as_ref(py).ne(expected.as_ref(py))? as u32 as f32);
    Ok((insertion, deletion, substitution))
}

#[pyclass]
#[derive(Clone, Debug)]
pub struct PropertyWeighting {
    insertion_cost: f32,
    deletion_cost: f32,
    table_indexer: PyObject,
}

#[pymethods]
impl PropertyWeighting {
    #[new]
    pub fn new(
        py: Python,
        insertion_cost: f32,
        deletion_cost: f32,
        property_table: PyObject,
    ) -> PyResult<Self> {
        Ok(Self {
            insertion_cost,
            deletion_cost,
            table_indexer: property_table.getattr(py, "__getitem__")?,
        })
    }

    pub fn levensthein_matrix<'a>(
        &self,
        py: Python<'a>,
        string_a: Vec<PyObject>,
        string_b: Vec<PyObject>,
    ) -> PyResult<&'a PyAny> {
        // let other_costs = property_table.call_method1(py, "size", (1,))?.extract::<f32>(py)? / 3.;
        levensthein_matrix_general(
            py,
            string_a,
            string_b,
            self.deletion_cost,
            self.cost_function(py),
        )
    }

    pub fn levensthein_operations(
        &self,
        py: Python,
        string_a: Vec<PyObject>,
        string_b: Vec<PyObject>,
    ) -> PyResult<(Operations, f32)> {
        // let other_costs = property_table.call_method1(py, "size", (1,))?.extract::<f32>(py)? / 3.;
        levensthein_operations_general(
            py,
            string_a,
            string_b,
            self.deletion_cost,
            self.cost_function(py),
        )
    }

    pub fn levensthein_statistics(
        &self,
        py: Python,
        string_a: Vec<PyObject>,
        string_b: Vec<PyObject>,
    ) -> PyResult<EditStatistics> {
        // let other_costs = property_table.call_method1(py, "size", (1,))?.extract::<f32>(py)? / 3.;
        levensthein_statistics_general(
            py,
            string_a,
            string_b,
            self.deletion_cost,
            self.cost_function(py),
        )
    }
}

trait Captures<'a> {}
impl<'a, T: ?Sized> Captures<'a> for T {}

impl PropertyWeighting {
    fn cost_function<'a, 'b: 'a>(
        &'a self,
        py: Python<'b>,
    ) -> impl Fn(Python, f32, f32, f32, &PyObject, &PyObject) -> PyResult<(f32, f32, f32)>
           + 'a
           + Captures<'b> {
        move |_: Python,
              above_cost: f32,
              left_cost: f32,
              upper_left_cost: f32,
              current: &PyObject,
              expected: &PyObject| {
            let insertion = left_cost + self.insertion_cost;
            let deletion = above_cost + self.deletion_cost;
            let substitution = upper_left_cost
                + self
                    .table_indexer
                    .call1(py, (current,))?
                    .call_method1(py, "ne", (self.table_indexer.call1(py, (expected,))?,))?
                    .call_method0(py, "sum")?
                    .extract::<f32>(py)?;
            Ok((insertion, deletion, substitution))
        }
    }
}

#[pyfunction]
pub fn levensthein_statistics(
    py: Python,
    string_a: Vec<PyObject>,
    string_b: Vec<PyObject>,
) -> PyResult<EditStatistics> {
    levensthein_statistics_general(py, string_a, string_b, 1., uniform_costs)
}

pub fn add_edit_functions(module: &PyModule) -> PyResult<()> {
    module.add_function(pyo3::wrap_pyfunction!(levensthein, module)?)?;
    module.add_function(pyo3::wrap_pyfunction!(levensthein_operations, module)?)?;
    module.add_function(pyo3::wrap_pyfunction!(levensthein_statistics, module)?)?;
    module.add_function(pyo3::wrap_pyfunction!(levensthein_matrix, module)?)?;
    module.add_function(pyo3::wrap_pyfunction!(levensthein, module)?)?;
    module.add_function(pyo3::wrap_pyfunction!(to_substitutions, module)?)?;
    Ok(())
}
