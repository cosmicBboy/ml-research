"""This cli is adapted from floyd-cli to fulfil custom use cases."""

import sys

import click

import floyd
from floyd.client.data import DataClient
from floyd.client.experiment import ExperimentClient
from floyd.log import logger as floyd_logger
from floyd.exceptions import FloydException
from floyd.cli.data import get_data_object
from floyd.cli.utils import (
    normalize_job_name
)


@click.group()
def cli():
    pass


@cli.command("get-output")
@click.argument("id", nargs=1)
@click.option("--path", "-p",
              help="Download files in a specific path from job output or a "
                   "dataset")
@click.option("--untar", is_flag=True)
@click.option("--delete-after-untar", is_flag=True)
def get_output(id, path, untar, delete_after_untar):
    """
    - Download all files in a dataset or from a Job output
    Eg: alice/projects/mnist/1/files, alice/projects/mnist/1/output or
    alice/dataset/mnist-data/1/

    Using /output will download the files that are saved at the end of the job.
    Note: This will download the files that are saved at
    the end of the job.
    - Download a directory from a dataset or from Job output
    Specify the path to a directory and download all its files and
    subdirectories.
    Eg: --path models/checkpoint1
    """
    data_source = get_data_object(id, use_data_config=False)

    if not data_source:
        if "output" in id:
            floyd_logger.info(
                "Note: You cannot clone the output of a running job. You need "
                "to wait for it to finish.")
        sys.exit()

    if path:
        # Download a directory from Dataset or Files
        # Get the type of data resource from the id
        # (foo/projects/bar/ or foo/datasets/bar/)
        if "/datasets/" in id:
            resource_type = "data"
            resource_id = data_source.id
        else:
            resource_type = "files"
            try:
                experiment = ExperimentClient().get(
                    normalize_job_name(id, use_config=False))
            except FloydException:
                experiment = ExperimentClient().get(id)
            resource_id = experiment.id

        data_url = "{}/api/v1/download/artifacts/{}/{}?is_dir=true&path={}" \
            .format(floyd.floyd_host, resource_type, resource_id, path)
    else:
        # Download the full Dataset
        data_url = "{}/api/v1/resources/{}?content=true&download=true".format(
            floyd.floyd_host, data_source.resource_id)

    DataClient().download_tar(
        url=data_url,
        untar=untar,
        delete_after_untar=untar and delete_after_untar,
    )


if __name__ == "__main__":
    cli()
