#  Copyright (c) Microsoft Corporation.
#  Licensed under the MIT License.
import logging
import os
os.environ["MLFLOW_ALLOW_FILE_STORE"] = "true"

from pathlib import Path
import sys
import tempfile

import fire
from jinja2 import Template, meta
from ruamel.yaml import YAML

import qlib
from qlib.config import C
from qlib.log import get_module_logger
from qlib.model.trainer import task_train
from qlib.utils import set_log_with_config
from qlib.utils.data import update_config

set_log_with_config(C.logging_config)
logger = get_module_logger("qrun", logging.INFO)


def get_path_list(path):
    if isinstance(path, str):
        return [path]
    else:
        return list(path)


def sys_config(config, config_path):
    """
    Configure the `sys` section

    Parameters
    ----------
    config : dict
        configuration of the workflow.
    config_path : str
        path of the configuration
    """
    sys_config = config.get("sys", {})

    # abspath
    for p in get_path_list(sys_config.get("path", [])):
        sys.path.append(p)

    # relative path to config path
    for p in get_path_list(sys_config.get("rel_path", [])):
        sys.path.append(str(Path(config_path).parent.resolve().absolute() / p))


def render_template(config_path: str, config_content: str = None) -> str:
    """
    render the template based on the environment

    Parameters
    ----------
    config_path : str
        configuration path

    Returns
    -------
    str
        the rendered content
    """
    if config_content is None:
        with open(config_path, "r") as f:
            config = f.read()
    else:
        config = config_content
    # Set up the Jinja2 environment
    template = Template(config)

    # Parse the template to find undeclared variables
    env = template.environment
    parsed_content = env.parse(config)
    variables = meta.find_undeclared_variables(parsed_content)

    # Get context from os.environ according to the variables
    context = {var: os.getenv(var, "") for var in variables if var in os.environ}
    logger.info(f"Render the template with the context: {context}")

    # Render the template with the context
    rendered_content = template.render(context)
    return rendered_content


# workflow handler function
def workflow(config_path, experiment_name="workflow", uri_folder="mlruns"):
    """
    This is a Qlib CLI entrance.
    User can run the whole Quant research workflow defined by a configure file
    - the code is located here ``qlib/cli/run.py``

    User can specify a base_config file in your workflow.yml file by adding "BASE_CONFIG_PATH".
    Qlib will load the configuration in BASE_CONFIG_PATH first, and the user only needs to update the custom fields
    in their own workflow.yml file.

    For examples:

        qlib_init:
            provider_uri: "~/.qlib/qlib_data/cn_data"
            region: cn
        BASE_CONFIG_PATH: "workflow_config_lightgbm_Alpha158_csi500.yaml"
        market: csi300

    """
    config_path = Path(config_path).resolve()
    with config_path.open("r") as fp:
        source_config_content = fp.read()
    base_config_source = None

    # Render the template
    rendered_yaml = render_template(config_path, source_config_content)
    yaml = YAML(typ="safe", pure=True)
    config = yaml.load(rendered_yaml)

    base_config_path = config.get("BASE_CONFIG_PATH", None)
    if base_config_path:
        logger.info(f"Use BASE_CONFIG_PATH: {base_config_path}")
        base_config_path = Path(base_config_path)

        # it will find config file in absolute path and relative path
        if base_config_path.exists():
            path = base_config_path
        else:
            logger.info(
                f"Can't find BASE_CONFIG_PATH base on: {Path.cwd()}, "
                f"try using relative path to config path: {Path(config_path).absolute()}"
            )
            relative_path = Path(config_path).absolute().parent.joinpath(base_config_path)
            if relative_path.exists():
                path = relative_path
            else:
                raise FileNotFoundError(f"Can't find the BASE_CONFIG file: {base_config_path}")

        path = path.resolve()
        with path.open("r") as fp:
            base_config_content = fp.read()
        base_config_source = (path.name, base_config_content)
        yaml = YAML(typ="safe", pure=True)
        base_config = yaml.load(base_config_content)
        logger.info(f"Load BASE_CONFIG_PATH succeed: {path.resolve()}")
        config = update_config(base_config, config)

    # config the `sys` section
    sys_config(config, config_path)

    if "exp_manager" in config.get("qlib_init"):
        qlib.init(**config.get("qlib_init"))
    else:
        exp_manager = C["exp_manager"]
        exp_manager["kwargs"]["uri"] = "file:" + str(Path(os.getcwd()).resolve() / uri_folder)
        qlib.init(**config.get("qlib_init"), exp_manager=exp_manager)

    if "experiment_name" in config:
        experiment_name = config["experiment_name"]

    # Serialize the fully rendered and merged config before training starts.
    # This freezes the config for the current run even if its source YAML is
    # edited while the training process is still running.
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        config_yaml_path = temp_path / "config.yaml"
        output_yaml = YAML(typ="safe", pure=True)
        with config_yaml_path.open("w", encoding="utf-8") as fp:
            output_yaml.dump(config, fp)

        source_config_path = temp_path / config_path.name
        source_config_path.write_text(source_config_content, encoding="utf-8")

        base_config_path = None
        if base_config_source is not None:
            base_config_path = temp_path / "base" / base_config_source[0]
            base_config_path.parent.mkdir()
            base_config_path.write_text(base_config_source[1], encoding="utf-8")

        recorder = task_train(
            config.get("task"),
            experiment_name=experiment_name,
            recorder_name=config.get("run_name"),
            tags=config.get("tags"),
            description=config.get("description"),
        )
        recorder.save_objects(config=config)
        recorder.log_artifact(str(config_yaml_path))
        recorder.log_artifact(str(source_config_path), artifact_path="workflow_config")
        if base_config_path is not None:
            recorder.log_artifact(str(base_config_path), artifact_path="workflow_config/base")


# function to run workflow by config
def run():
    fire.Fire(workflow)


if __name__ == "__main__":
    run()
