use std::cell::Cell;
use std::iter;
use std::mem;
use std::rc::Rc;

use aho_corasick::{AhoCorasick, AhoCorasickBuilder, MatchKind};
use pyo3::exceptions::PyValueError;
use pyo3::pymethods;
use pyo3::{prelude::pyclass, PyErr, PyResult};

pyo3::create_exception!(phonemes, MissingSegmentError, PyValueError);

fn vocabulary_error(element: &str, word: &str) -> PyErr {
    MissingSegmentError::new_err(format!(
        "Segment {:?} is missing from the vocabulary. Found in: {:?}",
        element, word
    ))
}

#[pyclass]
pub struct IpaSegmenter {
    automaton: AhoCorasick,
    #[pyo3(get)]
    ipa_segments: Vec<String>,
}

impl IpaSegmenter {
    fn segment_word<'b: 'a, 'a>(&'a self, word: &'b str) -> impl Iterator<Item = &'b str> + 'a {
        self.automaton
            .find_iter(word)
            .map(move |segment| &word[segment.start()..segment.end()])
    }

    fn segment_word_with_missing<'b: 'a, 'a>(
        &'a self,
        word: &'b str,
    ) -> impl Iterator<Item = &'b str> + 'a {
        let mut last_end = 0;
        let mut next_end = None;
        let mut found_segments = self.automaton.find_iter(word);
        let length = word.len();

        iter::from_fn(move || {
            if let Some(end) = next_end.take() {
                return Some(&word[mem::replace(&mut last_end, end)..end]);
            }
            match found_segments.next() {
                Some(segment) => {
                    let start = segment.start();
                    let end = segment.end();
                    if start != last_end {
                        next_end = Some(end);
                        Some(&word[mem::replace(&mut last_end, start)..start])
                    } else {
                        last_end = end;
                        Some(&word[start..end])
                    }
                }
                None if last_end != length => Some(&word[mem::replace(&mut last_end, length)..]),
                None => None,
            }
        })
    }

    fn segment_word_checked<'b: 'a, 'a>(
        &'a self,
        word: &'b str,
    ) -> impl Iterator<Item = PyResult<&'b str>> + 'a {
        let last_end = Rc::new(Cell::new(0));
        let last_end_final = last_end.clone();

        self.automaton
            .find_iter(word)
            .map(move |segment| {
                let start = segment.start();
                if start == last_end.get() {
                    let end = segment.end();
                    last_end.set(end);
                    Ok(&word[start..end])
                } else {
                    Err(vocabulary_error(&word[last_end.get()..start], word))
                }
            })
            .chain(iter::from_fn(move || {
                if last_end_final.get() != word.len() {
                    Some(Err(vocabulary_error(&word[last_end_final.get()..], word)))
                } else {
                    None
                }
            }))
    }
}

#[pymethods]
impl IpaSegmenter {
    #[new]
    pub fn new(ipa_segments: Vec<String>) -> Self {
        Self {
            automaton: AhoCorasickBuilder::new()
                .match_kind(MatchKind::LeftmostLongest)
                .build(&ipa_segments),
            ipa_segments,
        }
    }

    #[args(include_missing = "false")]
    pub fn segment<'a>(&self, transcription: &'a str, include_missing: bool) -> Vec<&'a str> {
        if include_missing {
            self.segment_word_with_missing(transcription).collect()
        } else {
            self.segment_word(transcription).collect()
        }
    }

    pub fn segment_checked<'a>(&self, transcription: &'a str) -> PyResult<Vec<&'a str>> {
        self.segment_word_checked(transcription).collect()
    }

    #[args(include_missing = "false")]
    pub fn segment_words<'a>(
        &self,
        transcription: Vec<&'a str>,
        include_missing: bool,
    ) -> Vec<&'a str> {
        let iterator = transcription.iter();
        if include_missing {
            iterator
                .flat_map(|word| self.segment_word_with_missing(word))
                .collect()
        } else {
            iterator.flat_map(|word| self.segment_word(word)).collect()
        }
    }

    pub fn segment_words_checked<'a>(&self, transcription: Vec<&'a str>) -> PyResult<Vec<&'a str>> {
        transcription
            .iter()
            .flat_map(|word| self.segment_word_checked(word))
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::IpaSegmenter;

    fn make_segmenter() -> IpaSegmenter {
        IpaSegmenter::new(vec!["test".to_string(), "te".to_string(), "tool".to_string()])
    }

    #[test]
    fn test_segment() {
        let segmenter = make_segmenter();
        // Segments
        assert_eq!(vec!["tool", "test", "te"], segmenter.segment("atoolbtestattet", false));
        // Empty string
        assert_eq!(Vec::<&str>::new(), segmenter.segment("", false));
        // No matches
        assert_eq!(Vec::<&str>::new(), segmenter.segment("Segments don't match here", false));
        // Checked segmentation
        assert!(segmenter.segment_checked("atoolbtestatte").is_err());
        assert!(segmenter.segment_checked("testtoolte").is_ok());
    }
}
