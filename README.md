# Plant Health Capstone — RGB vs HSV Preprocessing

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/AhnafArib/plant-health-capstone/blob/main/plant_health_capstone.ipynb)

An independent project testing whether HSV colour-space preprocessing improves convolutional neural network accuracy on plant disease classification, compared to standard RGB input.

**Status:** In progress. Environment, dataset preparation and background learning are complete; model training and the RGB/HSV comparison are the next stage.

---

## The question

Plant disease is usually diagnosed visually, from discolouration, lesions and texture changes on leaves. Most image classifiers feed RGB pixels straight into the network. RGB entangles colour with brightness — the same leaf photographed in shade and in sun produces very different RGB values.

HSV separates hue and saturation from brightness (value). In principle this should make a model less sensitive to lighting variation, which matters for disease symptoms that are essentially colour changes. In practice it is not obvious that it helps, since CNNs can learn lighting invariance on their own given enough data.

This project tests it directly: same architecture, same data, same training procedure, with only the colour space changed.

## Dataset

The PlantVillage dataset, via the [emmarex/plantdisease](https://www.kaggle.com/datasets/emmarex/plantdisease) Kaggle mirror.

Scope was narrowed to three crops — **bell pepper, potato and tomato** — giving 15 classes covering both diseased and healthy leaves. Mint and cucumber were in the original plan but are not present in this dataset, so they were dropped rather than substituted.

## Repository contents

| File | What it is |
|------|------------|
| `plant_health_capstone.ipynb` | Working notebook. Part 1 follows the official TensorFlow image classification tutorial as a learning exercise; Part 2 is the capstone dataset preparation. |

Outputs are cleared from the committed notebook to keep it renderable on GitHub. Results will be saved as separate image files once training begins.

## Running this yourself

Click the Colab badge above to open the notebook. Part 1 runs as-is. Part 2 downloads the dataset from Kaggle, so it requires your own Kaggle API token (`kaggle.json`, available from your Kaggle account settings) and a Google Drive mount for persistent storage.

## Progress

**Done**

- Colab environment configured, TensorFlow 2.20 running on GPU
- Worked through the [TensorFlow image classification tutorial](https://www.tensorflow.org/tutorials/images/classification) end to end, including diagnosing an overfitting model and correcting it with data augmentation and dropout
- Dataset downloaded, inspected and stored persistently on Google Drive
- Class scope finalised at 15 classes across three crops

**Next**

- Build the training pipeline on the PlantVillage data
- Establish an RGB baseline model
- Add an HSV conversion step and train an otherwise identical model
- Compare accuracy, per-class performance and training behaviour
- Test robustness under varied lighting, which is where any HSV advantage should show up most clearly

## Method notes

The comparison only means something if everything except colour space is held constant — same architecture, same train/validation split and seed, same augmentation, same number of epochs. Conversion happens as a preprocessing step so the two runs differ in exactly one respect.

A single run of each is not enough to conclude anything, since neural network training is stochastic. Multiple runs per condition are planned so that any difference can be distinguished from random variation.

A known limitation of PlantVillage is that most images are single leaves photographed against a plain background under controlled lighting. Accuracy on this dataset does not transfer directly to field conditions, and any lighting-robustness result will need to be read with that in mind.

## Attribution

Part 1 of the notebook is adapted from the official [TensorFlow image classification tutorial](https://www.tensorflow.org/tutorials/images/classification), used to learn the Keras workflow before applying it to my own problem. It is included because it documents how I got up to speed, not as original work.

The PlantVillage dataset was published by Hughes and Salathé (2015), *An open access repository of images on plant health to enable the development of mobile disease diagnostics*, arXiv:1511.08060.

## About

Independent capstone project, begun 2026. I'm a Computer Science applicant interested in machine learning applied to agriculture and climate adaptation. I maintain a related observation log of plant phenology in Dhaka in a [separate repository](https://github.com/AhnafArib/dhaka-orchid-phenology).
