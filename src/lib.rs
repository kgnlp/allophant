mod edit_distance;
mod ipa_segmenter;

use pyo3::prelude::{pymodule, PyModule, PyResult, Python};

use edit_distance::{Action, EditStatistics, PropertyWeighting};
use ipa_segmenter::{IpaSegmenter, MissingSegmentError};

#[pymodule]
fn phonemes(py: Python, module: &PyModule) -> PyResult<()> {
    module.add_class::<IpaSegmenter>()?;
    module.add_class::<EditStatistics>()?;
    module.add_class::<PropertyWeighting>()?;
    module.add_class::<Action>()?;
    edit_distance::add_edit_functions(module)?;
    module.add("MissingSegmentError", py.get_type::<MissingSegmentError>())?;
    Ok(())
}
