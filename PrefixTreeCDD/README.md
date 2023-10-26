# PrefixTreeCDD
Repository of the unsupervised-learning online concept drift detection technique over event streams.

Application has a built in CLI tool. All tests were done on Windows machine.

Here is a list of the available options:

* **f, file**           - Path to the XES log file.
* **c, config**         - (True/False) Configuration for generating sub-logs from the drifts identified.
* **t, tree_size**      - Maximum size of the trees (pruningSteps).
* **w, window_size**    - Maximum size of the window of detection (maxWindowSize)
* **n, noise**          - Noise filter applied to relations and activities which have a frequency below certain threshold.
* **d, decay_lambda**   - Decaying lambda for the older traces when performing a pruning step.

## Getting Command Help

You can get help from each command by running `-h` or `--help`

Eg.

```text
python BPIC_CDD.py --help
```

This will output to your shell the following documentation.

```text
Usage: BPIC_CDD.py [OPTIONS]

  This is the Prefix-Tree Concept Drift Detection algorithm.

Options:
  -l, --decay_lambda FLOAT   Decaying lambda for the older traces when
                             performing a pruning step.  [default: 0.25]
  -n, --noise INTEGER        Noise filter applied to relations and activities
                             which have a frequency below certain threshold.
                             [default: 1]
  -w, --window_size INTEGER  Maximum size of the window of detection
                             (maxWindowSize)  [default: 10]
  -t, --tree_size INTEGER    Maximum size of the trees (pruningSteps)
                             [default: 1000]
  -c, --config BOOLEAN       (True/False) Configuration for generating sub-
                             logs from the drifts identified  [default: False]
  -f, --file TEXT            Path to the XES log file.
  --help                     Show this message and exit.
```

## Usage
All libraries must be installed from either the requirements.txt or requirements.yml file on your environment.

```text
python BPIC_CDD.py -f path-to-log-file
```

### Options

* `-c` defaults to False
* `-w` defaults to 10
* `-t` defaults to 1000
* `-n` defaults to 1
* `-l` defaults to 0.25

