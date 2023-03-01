# PyTorch Project Template

## Setup paths 
Before you can start using the template, you need to first setup the paths.yaml, to do so, please run:
```shell=bash
bash setup_paths.sh
```
The script will generate `paths.yaml` which contains the reference to the current directory.

## Setup environment
Install all necessary packages to run the codes
```shell=bash
pip install -r requirements.txt
```

## Running the training code
To do this, go to the `src/engine` folder, and run the `main.py` by passing the config file placed under `configs` directory.

```shell=bash
cd src/engine
python main.py --config_file=base.yaml
python main.py --config_file=base2.yaml
```

The initial step of setting up the paths help us locate the `base.yaml` under the `configs` directory.

## Visualizing Runs
In this template, we are using mlflow to track the progress of our training. Which means we can visualize the previous runs using MLFlow UI. To start the server, simply run the script:
```shell=bash
bash start_mlflow.sh
```

## Using the Template
1. Add more dataset under `src/engine/dataloader.py`; essentially this would be the data wrapper that your model is interfaced with.
2. Add more architectures to `src/engine/archs` and define the wrapper in `src/engine/model.py`
    - In the current example, we have:
        - `src/engine/archs/DummyNet.py: DummyNet` as the network architecture, and 
        - `src/engine/model.py: DummyModel` as the wrapper.
3. Implement the `train_step` and `eval_step` of the model wrapper.
4. Route all parameters from `configs/{config_file}.yaml`.
5. Run your training/inferencing/evaluation routine `python main.py --config_file={config_name}.yaml`

----
Feel free to contact me for any inquiries on email: jonathanhans31@gmail.com

