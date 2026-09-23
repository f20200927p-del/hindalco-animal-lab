# Animal Classifier Specification

## Goal

Predict which animal appears in a photo and return the predicted animal together
with a confidence score and scores for every known animal.

## Model

The classifier uses transfer learning with `torchvision` MobileNetV2 and its
ImageNet weights by default. `model.features` is frozen, and
`model.classifier[1]` is replaced with a new `torch.nn.Linear` layer whose output
size is the number of discovered animal classes. During training,
`model.features` remains in evaluation mode. The `--from_scratch` training
experiment disables ImageNet weights and makes every layer trainable.

## Images and classes

Images are stored under one directory per animal; the directory name is the
label. The implementation must discover class names from directory names and
must not hard-code either the number or names of animals. Adding another animal
directory requires no code change. Images are resized to 224x224 and normalized
with ImageNet mean `(0.485, 0.456, 0.406)` and standard deviation
`(0.229, 0.224, 0.225)`. Exactly the same transform is used for training,
evaluation, and prediction.

## Data split

`prep.py` validates every image and makes an 80% train / 20% test split within
each animal directory. The split is deterministic with seed 42 by default (or
the supplied `--seed`) and honors the supplied `--test_ratio`. At least two
animal directories and at least 10 valid images per animal are required by
default. A single wrapper directory around the animal directories is accepted.

## Persistence

The saved model is a directory containing:

- `model.pt`: the model `state_dict`.
- `classes.json`: a JSON list of animal names in model-output order.

`load_model` searches the requested directory and its subdirectories for this
pair and reconstructs the model from the saved class list.

## Quality gate and metrics

`evaluate.py` reports overall accuracy, per-animal accuracy, and a confusion
matrix whose rows are real classes and columns are predicted classes. It writes
these values to `metrics.json`. It exits with code 1 when test accuracy is below
`--min_accuracy` (0.70 by default), and exits successfully otherwise.

## Command-line interfaces

- `prep.py --raw_data --train_out --test_out [--test_ratio 0.2] [--min_images 10] [--seed 42]`
- `train.py --train_data --model_dir [--epochs 10] [--lr 0.001] [--batch_size 16] [--from_scratch]`
- `evaluate.py --model_dir --test_data --metrics_out [--min_accuracy 0.70]`
- `predict.py --model_dir --image`
- `score.py` exposes Azure ML `init()` and `run(raw_data)`.

## Acceptance criteria

**AC1.** The specification exists before implementation and defines the goal,
model, preprocessing, dynamic classes, split, persistence, and quality gate.

**AC2.** `prep.py` accepts the exact required arguments, validates that every
image opens, supports a wrapper directory, discovers classes dynamically, and
prints a split table.

**AC3.** Preparation rejects fewer than two animals or fewer than the minimum
images in any animal, and produces per-animal train and test folders.

**AC4.** The split is deterministic for a fixed seed and defaults to 80/20.

**AC5.** `model_utils.py` implements the shared ImageNet transform, MobileNetV2
construction/freezing rules, save/load behavior, prediction scores, and the
Azure-only MLflow metric rule.

**AC6.** `train.py` accepts the exact pipeline arguments, reports loss and
training accuracy per epoch, and saves the required model artifacts.

**AC7.** `evaluate.py` writes the required metrics and confusion matrix, prints
the required reports, and enforces the 0.70 default quality gate with exit code
1 on failure.

**AC8.** `predict.py` prints `Prediction: <animal>  (confidence NN%)` and uses
the same preprocessing as training.

**AC9.** `score.py` loads from `AZUREML_MODEL_DIR`, accepts base64-image JSON,
returns the specified prediction schema, logs one line per prediction, and
returns `{"error": "<message>"}` for errors.