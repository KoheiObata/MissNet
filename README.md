# MissNet



## Requirements

We run all the experiments in `python 3.8`, see `requirements.txt` for the list of `pip` dependencies.

To install packages

```
pip install -r requirements.txt
```

or

```
pip install numpy
pip install pandas
pip install matplotlib
pip install seaborn
```


## Datasets preparation

### Synthetic datasets
Generate synthetic datasets with the code below.

```
cd  data
python generate.py
```

---

### MotionCapture datasets
MotionCapture datasets are stored in `dynammo.zip`.

The original data can be downloaded from
[link](https://github.com/lileicc/dynammo/tree/master/data/c3d).
We changed the filename for convenience.

This is the original website
[link](http://mocap.cs.cmu.edu).

---

### Motes dataset
Motes dataset is stored in `motes.zip`.

The original data can be downloaded from
[link](https://db.csail.mit.edu/labdata/labdata.html).


## Missing block generation

We generate missing blocks for the experiments.

```
cd  data
python conversion.py
```

Change `missing_rate_test` in `conversion.py` for full experiments (if necessary).

The data containing missing blocks will be stored in `./data/experiment`.


## Experiments

After the preparation of datasets and missing blocks, run below.

```
python Experiment.py --datasets dynammo
python Experiment.py --datasets motes
python Experiment.py --datasets synthetic/pattern
```
