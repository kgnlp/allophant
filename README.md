# Allophant

Allophant is a multilingual phoneme recognizer trained on spoken sentences in 34 languages, capable of generalizing zero-shot to unseen phoneme inventories.

This implementation was utilized in our INTERSPEECH 2023 paper ["Allophant: Cross-lingual Phoneme Recognition with Articulatory Attributes"](https://www.isca-archive.org/interspeech_2023/glocker23_interspeech.html) ([Citation](#citation))

## Checkpoints

Pre-trained checkpoints for all evaluated models can be found on [Hugging Face](https://huggingface.co/collections/kgnlp/allophant-67141a8deba78564d9dcfdad):

| Model Name       | UCLA Phonetic Corpus (PER) | UCLA Phonetic Corpus (AER) | Common Voice (PER) | Common Voice (AER) |
| ---------------- | -------------------------: | -------------------------: | -----------------: | -----------------: |
| [Multitask](https://huggingface.co/kgnlp/allophant)        | **45.62%** | 19.44% | **34.34%** | **8.36%** |
| [Hierarchical](https://huggingface.co/kgnlp/allophant-hierarchical)     | 46.09% | **19.18%** | 34.35% | 8.56% |
| [Multitask Shared](https://huggingface.co/kgnlp/allophant-shared) | 46.05% | 19.52% | 41.20% | 8.88% |
| [Baseline Shared](https://huggingface.co/kgnlp/allophant-baseline-shared)  | 48.25% |   -    | 45.35% |  -    |
| [Baseline](https://huggingface.co/kgnlp/allophant-baseline)         | 57.01% |   -    | 46.95% |  -    |

Note that our baseline models were trained without phonetic feature classifiers and therefore only support phoneme recognition.

## Result Files

JSON files containing detailed error rates and statistics for all languages can be found in the [`interspeech_results`](/interspeech_results/) directory. Results on the [UCLA Phonetic Corpus](https://github.com/xinjli/ucla-phonetic-corpus) are stored in files ending in "ucla", while files containing results on the training subset of languages from [Mozilla Common Voice](https://commonvoice.mozilla.org) end in "commonvoice". See [Error Rates](#error-rates) for more information.

## Installation

### System Dependencies

For most Linux and macOS systems, pre-built binaries are available via pip. For installation on other platforms or when building from source, a Rust compiler is required for building the native `pyo3` extension. [Rustup](https://rustup.rs/) is recommended for managing Rust installations.

#### Optional

Torchaudio mp3 support requires ffmpeg to be installed on the system. E.g. for Debian-based Linux distributions:

```bash
sudo apt update && sudo apt install ffmpeg
```
For transcribing training and evaluation data with eSpeak NG G2P, The [espeak-ng](https://github.com/espeak-ng/espeak-ng) package is required:

```bash
sudo apt install espeak-ng
```

We transcribed Common Voice using version 1.51 for our paper.

### Allophant Package

Allophant can be installed via pip:

```bash
pip install allophant
```

Note that the package currently requires Python >= 3.10 and was tested on 3.12. For use on GPU, torch and torchaudio may need to be manually installed for your required CUDA or ROCm version. ([PyTorch installation](https://pytorch.org/get-started/locally/))

For development, an editable package can be installed as follows:

```bash
git clone https://github.com/kgnlp/allophant
cd allophant
pip install -e allophant
```

## Usage

### Inference With Pre-trained Models

A pre-trained model can be loaded with the `allophant` package from a huggingface checkpoint or local file:

```python
from allophant.estimator import Estimator

device = "cpu"
model, attribute_indexer = Estimator.restore("kgnlp/allophant", device=device)
supported_features = attribute_indexer.feature_names
# The phonetic feature categories supported by the model, including "phonemes"
print(supported_features)
```
Allophant supports decoding custom phoneme inventories, which can be constructed in multiple ways:

```python
# 1. For a single language:
inventory = attribute_indexer.phoneme_inventory("es")
# 2. For multiple languages, e.g. in code-switching scenarios
inventory = attribute_indexer.phoneme_inventory(["es", "it"])
# 3. Any custom selection of phones for which features are available in the Allophoible database
inventory = ['a', 'ai̯', 'au̯', 'b', 'e', 'eu̯', 'f', 'ɡ', 'l', 'ʎ', 'm', 'ɲ', 'o', 'p', 'ɾ', 's', 't̠ʃ']
````

Audio files can then be loaded, resampled and transcribed using the given
inventory by first computing the log probabilities for each classifier:

```python
import torch
import torchaudio
from allophant.dataset_processing import Batch

# Load an audio file and resample the first channel to the sample rate used by the model
audio, sample_rate = torchaudio.load("utterance.wav")
audio = torchaudio.functional.resample(audio[:1], sample_rate, model.sample_rate)

# Construct a batch of 0-padded single channel audio, lengths and language IDs
# Language ID can be 0 for inference
batch = Batch(audio, torch.tensor([audio.shape[1]]), torch.zeros(1))
model_outputs = model.predict(
  batch.to(device),
  attribute_indexer.composition_feature_matrix(inventory).to(device)
)
```

Finally, the log probabilities can be decoded into the recognized phonemes or phonetic features:

```python
from allophant import predictions

# Create a feature mapping for your inventory and CTC decoders for the desired feature set
inventory_indexer = attribute_indexer.attributes.subset(inventory)
ctc_decoders = predictions.feature_decoders(inventory_indexer, feature_names=supported_features)

for feature_name, decoder in ctc_decoders.items():
    decoded = decoder(model_outputs.outputs[feature_name].transpose(1, 0), model_outputs.lengths)
    # Print the feature name and values for each utterance in the batch
    for [hypothesis] in decoded:
        # NOTE: token indices are offset by one due to the <BLANK> token used during decoding
        recognized = inventory_indexer.feature_values(feature_name, hypothesis.tokens - 1)
        print(feature_name, recognized)
```

### Configuration

To specify options for preprocessing, training, and the model architecture, a configuration file in [TOML format](https://toml.io) can be passed to most commands.
For automation purposes, JSON configuration files can be used instead with the `--config-json-data/-j` flag.
To start, a default configuration file with comments can be generated as follows:

```bash
allophant generate-config [path/to/config]
```

### Preprocessing

The `allophant-data` command contains all functionality for corpus processing and management available in `allophant`.
For training, corpora without phoneme-level transcriptions have to be transcribed beforehand with a grapheme-to-phoneme model.

#### Transcription

Phoneme transcriptions for a supported corpus format can be generated with `transcribe`.
For instance, for transcribing the German and English subsets of a corpus with eSpeak NG and [PHOIBLE features](https://github.com/phoible/dev/tree/master/raw-data/FEATURES) from [Allophoible](https://github.com/Aariciah/allophoible) using a batch size of 512 and at most 15,000 utterances per language:

```bash
allophant-data transcribe -p -e espeak-ng -b 512 -l de,en -t 15000 -f phoible path/to/corpus -o transcribed_data
```

Note that no audio data is moved or copied in this process.
All commands that load corpora also accept a path to the `*.bin` transcription file directly instead of a directory. This allows loading only specific splits, such as loading only the `test` split for [evaluation](#evaluation).

#### Utterance Lengths

As an optional step, utterance lengths can be extracted from a transcribed corpus for more memory efficient batching. If a subset of the corpus was transcribed, lengths will only be stored for the transcribed utterances.

```bash
allophant-data save-lengths [-c /path/to/config.toml] path/to/transcribed_corpus path/to/output
```

### Training

During training, the best checkpoint is saved after each evaluation step to the path provided via the `--save-path/-s` flag. To save every checkpoint instead, a directory needs to be passed to `--save-path/-s` and the `--save-all/-a` flag included. The number of worker threads is auto-detected from the number of available CPU threads but can be set manually with `-w number`. To train only on the CPU instead of using CUDA, the `--cpu` flag can be used. Finally, any progress logging to stderr can be disabled with `--no-progress`.

```bash
allophant train [-c /path/to/config.toml] [-w number] [--cpu] [--no-progress] [--save-all]
  [-s /path/to/checkpoint.pt] [-l /path/to/lengths] path/to/transcribed_corpus
```

Note that at least the `--lengths/-l` flag with a path to previously computed utterance lengths has to be specified when the "frames" batching mode is enabled.

### Evaluation

#### Test Data Inference

For evaluation, test data can be transcribed with the `predict` sub-command. The resulting file contains metadata, transcriptions for phonemes and features, and gold standard labels from the test data.

```bash
allophant predict [--cpu] [-w number] [-t {ucla-phonetic,common-voice}] [-f phonemes,feature1,feature2]
  [--fix-unicode] [--training-languages {include,exclude,only}] [-m {frames,utterances}] [-s number]
  [--language-phonemes] [--no-progress] [-c] [-o /path/to/prediction_file.jsonl] /path/to/dataset huggingface/model_id or /path/to/checkpoint
```

Use `--dataset-type/-t` to select the data set type. Note that only Common Voice and the UCLA Phonetic Corpus are currently supported. Predictions will either be printed to stdout or saved to a file given by `--output/-o`. Gzip compression is either inferred from a ".jsonl.gz" extension or can be forced with the `--compress/-c` flag. The `--training-languages` argument allows filtering utterances based on the languages that also occur in the training data, and should be set to "exclude" for zero-shot evaluation.

Using `--feature-subset/-f`, a comma separated list of features or "phoneme" such as `syllabic,round,phoneme` can be provided to predict only the given subset of classes. With the `--fix-unicode` option, `predict` attempts to resolve issues of phonemes from the test data missing from the database due to differences in their unicode binary representation.

The batch sizes defined in the model configuration for training can be overwritten with the `--batch-size/-s` and `--batch-mode/-m`. Note that if the batch mode is set to "utterance" either in the model configuration or by setting the `--batch-mode/-m` flag, [utterance lengths](#utterance-lengths) have to be provided via the `--lengths/-l` argument. A beam size can be specified for CTC decoding with beam search (`--ctc-beam/-b`). We used a beam size of 1 for greedy decoding in our paper.

#### Error Rates

The `evaluate` sub-command computes edit statistics and phoneme and attribute error rates for each language of a given corpus or split.

```bash
allophant evaluate [--fix-unicode] [--no-remap] [--split-complex] [-j] [--no-progress]
  [-o path/to/results.json] [-d] path/to/predictions.jsonl
```

Without `--no-remap`, transcriptions are mapped to language inventories using the same mapping scheme used during training. In our paper, all results were computed without this mapping, meaning that the transcriptions were directly compared to labels without an additional mapping step. If `--fix-unicode` was used during prediction, it should also be used in `evaluate`. Evaluation supports splitting any complex phoneme segments before computing error statistics with the `--split-complex/-s` flag.

For further analysis of evaluation results, JSON output should be enabled via the `--json/-j` flag. The JSON file can then be read using `allophant.evaluation.EvaluationResults`. For quick inspection of human-readable (average) error rates from evaluation results saved in JSON format, use `allophant-error-rates`:

```bash
allophant-error-rates path/to/results_file
```

### Allophoible Inventories

Inventories and feature sets preprocessed from (a subset of) [Allophoible](https://github.com/Aariciah/allophoible) for training can be extracted with the `allophant-features` command.

```bash
allophant-features [-p /path/to/allophoible.csv] [--remove-zero] [--prefer-allophant-dialects]
  [-o /path/to/output.csv] [en,fr,ko,ar,...]
```

# Citation

When using our work, please cite our paper as follows:

```bibtex
@inproceedings{glocker2023allophant,
    title={Allophant: Cross-lingual Phoneme Recognition with Articulatory Attributes},
    author={Glocker, Kevin and Herygers, Aaricia and Georges, Munir},
    year={2023},
    booktitle={{Proc. Interspeech 2023}},
    month={8}}
```
