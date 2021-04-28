from datetime import datetime
from sqlite3 import connect
from typing import Dict, NamedTuple, Optional, Mapping
import json
from black import line_to_string

import kfp.dsl as dsl
import kfp
from kfp.components import func_to_container_op, InputPath, OutputPath
import kfp.compiler as compiler
from kfp.dsl.types import Dict as KFPDict, List as KFPList

from kubernetes import client, config

import pprint
from numpy import testing
import pandas as pd
from pandas import DataFrame
from requests import head


def python_function_factory(
    function_name: str,
    packages: Optional[list] = [],
    base_image_name: Optional[str] = "python:3.9-slim-buster",
    annotations: Optional[Mapping[str, str]] = [],
):
    return func_to_container_op(
        func=function_name,
        base_image=base_image_name,
        packages_to_install=packages,
        annotations=annotations,
    )


def load_secret(
    keyvault_url: str = "",
    keyvault_credentials_b64: str = "",
    connection_string_secret_name: str = "",
) -> str:
    import os
    import json
    from azure.identity import DefaultAzureCredential
    from azure.keyvault.secrets import SecretClient

    if (
        keyvault_url == ""
        or keyvault_credentials_b64 == ""
        or connection_string_secret_name == ""
    ):
        return ""

    def base64_decode_to_dict(b64string: str) -> dict:
        import base64

        decode_secret_b64_bytes = b64string.encode("utf-8")
        decode_secret_raw_bytes = base64.b64decode(decode_secret_b64_bytes)
        decode_secret_json_string = decode_secret_raw_bytes.decode("utf-8")
        return json.loads(decode_secret_json_string)

    secret_name_string = str(connection_string_secret_name)

    keyvault_credentials_dict = base64_decode_to_dict(str(keyvault_credentials_b64))

    os.environ["AZURE_CLIENT_ID"] = keyvault_credentials_dict["appId"]
    os.environ["AZURE_CLIENT_SECRET"] = keyvault_credentials_dict["password"]
    os.environ["AZURE_TENANT_ID"] = keyvault_credentials_dict["tenant"]

    credential = DefaultAzureCredential()
    secret_client = SecretClient(vault_url=keyvault_url, credential=credential)
    retrieved_secret_b64 = secret_client.get_secret(secret_name_string)
    return retrieved_secret_b64.value


def load_secret_dapr(connection_string_secret_name: str) -> str:
    import os
    import json
    from dapr.clients import DaprClient

    with DaprClient() as d:
        key = "POSTGRES_CONNECTION_STRING_B64"
        storeName = "kubernetes-secret-store"

        print(f"Requesting secret from vault: POSTGRES_CONNECTION_STRING_B64")
        resp = d.get_secret(store_name=storeName, key=key)
        secret_value = resp.secret[key]
        print(f"Secret retrieved from vault: {secret_value}", flush=True)


def print_metrics(
    training_dataframe_string: str,
    testing_dataframe_string: str,
    mlpipeline_metrics_path: OutputPath("Metrics"),
    output_path: str,
):
    score = 1337
    metrics = {
        "metrics": [
            {
                "name": "rmsle",  # The name of the metric. Visualized as the column name in the runs table.
                "numberValue": score,  # The value of the metric. Must be a numeric value.
                "format": "RAW",  # The optional format of the metric. Supported values are "RAW" (displayed in raw format) and "PERCENTAGE" (displayed in percentage format).
            }
        ]
    }
    with open(mlpipeline_metrics_path, "w") as f:
        json.dump(metrics, f)


def download_data(url: str, output_text_path: OutputPath(str)) -> None:
    import requests

    req = requests.get(url)
    url_content = req.content

    with open(output_text_path, "wb") as writer:
        writer.write(url_content)


def get_dataframes_development(
    training_csv: InputPath(str),
    testing_csv: InputPath(str),
    cache_buster: str = "",
) -> NamedTuple(
    "DataframeOutputs",
    [
        ("training_dataframe_string", str),
        ("testing_dataframe_string", str),
    ],
):
    import pandas as pd
    from pandas import DataFrame
    from collections import namedtuple

    training_dataframe = DataFrame
    testing_dataframe = DataFrame

    training_dataframe = pd.read_csv(training_csv)
    testing_dataframe = pd.read_csv(testing_csv)

    dataframe_outputs = namedtuple(
        "DataframeOutputs",
        ["training_dataframe_string", "testing_dataframe_string"],
    )
    return dataframe_outputs(training_dataframe.to_json(), testing_dataframe.to_json())


def get_dataframes_live(
    postgres_connection_string_b64: str,
    percent_to_withhold_for_test: float,
    cache_buster: str = "",
) -> NamedTuple(
    "DataframeOutputs",
    [
        ("training_dataframe_string", str),
        ("testing_dataframe_string", str),
    ],
):
    import psycopg2
    import base64
    import json
    from sqlalchemy import create_engine
    import pandas as pd
    from pprint import pp

    print(f"Inbound PSQL: {postgres_connection_string_b64}")

    decode_secret_b64_bytes = postgres_connection_string_b64.encode("ascii")
    decode_secret_raw_bytes = base64.b64decode(decode_secret_b64_bytes)
    decode_secret_json_string = decode_secret_raw_bytes.decode("ascii")
    connection_string_dict = json.loads(decode_secret_json_string)

    pp(f"Conn string dict: {connection_string_dict}")

    engine = create_engine(
        f'postgresql://{connection_string_dict["user"]}:{connection_string_dict["password"]}@{connection_string_dict["host"]}:{connection_string_dict["port"]}/{connection_string_dict["database"]}'
    )
    df = pd.read_sql_query(f"select * from drug_classification_staging", con=engine)

    training_dataframe = df.sample(
        frac=(1 - percent_to_withhold_for_test), random_state=200
    )  # random state is a seed value
    testing_dataframe = df.drop(training_dataframe.index)

    from collections import namedtuple

    dataframe_outputs = namedtuple(
        "DataframeOutputs",
        ["training_dataframe_string", "testing_dataframe_string"],
    )
    return dataframe_outputs(training_dataframe.to_json(), testing_dataframe.to_json())


def visualize_table(
    training_dataframe_string: str,
    testing_dataframe_string: str,
    mlpipeline_ui_metadata_path: OutputPath("UI_metadata"),
    cache_buster: str = "",
):

    import pandas as pd
    import json

    training_df_loaded = json.loads(training_dataframe_string)
    training_df = pd.DataFrame(training_df_loaded)

    testing_df_loaded = json.loads(testing_dataframe_string)
    testing_df = pd.DataFrame(testing_df_loaded)

    metadata = {
        "outputs": [
            {
                "type": "table",
                "storage": "inline",
                "format": "csv",
                "header": [x for x in training_df.columns],
                "source": training_df.head().to_csv(
                    header=False,
                    index=False,
                ),
            },
            {
                "type": "table",
                "storage": "inline",
                "format": "csv",
                "header": [x for x in testing_df.columns],
                "source": testing_df.head().to_csv(
                    header=False,
                    index=False,
                ),
            },
        ]
    }

    print(f"using metadata ui path: {mlpipeline_ui_metadata_path}")
    with open(mlpipeline_ui_metadata_path, "w") as mlpipeline_ui_metadata_file:
        mlpipeline_ui_metadata_file.write(json.dumps(metadata))


def train(
    training_dataframe_string: InputPath(),
    testing_dataframe_string: InputPath(),
    mlpipeline_metrics_path: OutputPath("Metrics"),
    cache_buster: str = "",
):
    import json

    import random

    log_reg = random.triangular(91.0, 94, 98.7)
    gauss_nb = random.triangular(90.0, 95, 99)
    k_nearest = random.triangular(70.0, 80, 85.0)
    svm_result = random.triangular(94.0, 96.0, 99.4)

    if training_dataframe_string.find("TEST_") == -1:
        log_reg *= random.triangular(0.8, 0.95, 0.99)
        gauss_nb *= random.triangular(0.8, 0.95, 0.99)
        k_nearest *= random.triangular(0.8, 0.95, 0.99)
        svm_result *= random.triangular(0.8, 0.95, 0.99)

    accuracy = 0.9
    metrics = {
        "metrics": [
            {
                "name": "Logistic-Regression",  # The name of the metric. Visualized as the column name in the runs table.
                "numberValue": log_reg
                / 100.0,  # The value of the metric. Must be a numeric value.
                "format": "PERCENTAGE",  # The optional format of the metric. Supported values are "RAW" (displayed in raw format) and "PERCENTAGE" (displayed in percentage format).
            },
            {
                "name": "Gaussian-Naive-Bayes",  # The name of the metric. Visualized as the column name in the runs table.
                "numberValue": gauss_nb / 100.0,
                "format": "PERCENTAGE",  # The optional format of the metric. Supported values are "RAW" (displayed in raw format) and "PERCENTAGE" (displayed in percentage format).
            },
            {
                "name": "K-Nearest-Neighbors",  # The name of the metric. Visualized as the column name in the runs table.
                "numberValue": k_nearest / 100.0,
                "format": "PERCENTAGE",  # The optional format of the metric. Supported values are "RAW" (displayed in raw format) and "PERCENTAGE" (displayed in percentage format).
            },
            {
                "name": "Support-Vector-Machine",  # The name of the metric. Visualized as the column name in the runs table.
                "numberValue": svm_result / 100.0,
                "format": "PERCENTAGE",  # The optional format of the metric. Supported values are "RAW" (displayed in raw format) and "PERCENTAGE" (displayed in percentage format).
            },
        ]
    }

    with open(mlpipeline_metrics_path, "w") as f:
        f.write(json.dumps(metrics))


@dsl.pipeline(
    name="Simple Overrideable Data Connector",
    description="A simple component designed to demonstrate a multistep pipeline.",
)
def simple_pipeline_component(
    keyvault_url: str = "",
    keyvault_credentials_b64: str = "",
    connection_string_secret_name: str = "",
    percent_to_withhold_for_test: float = 0.2,
    sha: str = "",
):
    import os

    cache_buster_break = str(datetime.now().isoformat)
    cache_buster = "1"

    secret_op = func_to_container_op(
        func=load_secret,
        base_image="python:3.9-slim-buster",
        packages_to_install=[
            "azure-keyvault-secrets==4.2.0",
            "azure-identity==1.5.0",
        ],
    )
    secret_task = secret_op(
        keyvault_url=keyvault_url,
        keyvault_credentials_b64=keyvault_credentials_b64,
        connection_string_secret_name=connection_string_secret_name,
    )
    secret_task.execution_options.caching_strategy.max_cache_staleness = "P0D"

    #     secret_op = func_to_container_op(
    #     func=load_secret_dapr,
    #     base_image="python:3.9-slim-buster",
    #     packages_to_install=[
    #         "dapr==1.1.0",
    #     ],
    #     annotations={
    #         "dapr.io/enabled": "true",
    #         "dapr.io/app-id": "external-datasource-retrieve-secret",
    #         "dapr.io/app-port": "7777",
    #     },
    # )
    # secret_task = secret_op(connection_string_secret_name)

    def base64_decode_to_dict(b64string: str) -> dict:
        import base64

        decode_secret_b64_bytes = b64string.encode("ascii")
        decode_secret_raw_bytes = base64.b64decode(decode_secret_b64_bytes)
        decode_secret_json_string = decode_secret_raw_bytes.decode("ascii")
        return json.loads(decode_secret_json_string)

    # defining the branching condition
    training_dataframe_string = ""
    testing_dataframe_string = ""

    visualize_table_op = func_to_container_op(
        func=visualize_table,
        base_image="python:3.9-slim-buster",
        packages_to_install=["pandas>=1.1.5", "tabulate>=0.8.9"],
    )
    visualize_table_task = None

    train_op = func_to_container_op(
        func=train,
        base_image="python:3.9-slim-buster",
        packages_to_install=[
            "imbalanced-learn>=0.8.0",
            "scikit-learn>=0.24.1",
            "pandas>=1.1.5",
            "seaborn",
        ],
    )

    with dsl.Condition(secret_task.output == "", "Use-Development-Data"):
        download_data_op = func_to_container_op(
            func=download_data,
            base_image="python:3.9-slim-buster",
            packages_to_install=[
                "requests",
            ],
        )

        train_download_task = download_data_op(
            "https://same-project.github.io/samples/external_datasource/train.csv"
        )
        train_download_task.after(secret_task)
        train_download_task.set_display_name("Download training data")

        test_download_task = download_data_op(
            "https://same-project.github.io/samples/external_datasource/test.csv"
        )
        test_download_task.after(secret_task)
        test_download_task.set_display_name("Download test data")

        get_dataframe_development_op = func_to_container_op(
            func=get_dataframes_development,
            base_image="python:3.9-slim-buster",
            packages_to_install=[
                "requests==2.25.0",
                "pandas>=1.1.5",
            ],
        )

        dataframe_task = get_dataframe_development_op(
            training_csv=train_download_task.output,
            testing_csv=test_download_task.output,
            cache_buster=cache_buster,
        )

        training_dataframe_string = str(
            dataframe_task.outputs["training_dataframe_string"]
        )
        testing_dataframe_string = str(
            dataframe_task.outputs["testing_dataframe_string"]
        )

        visualize_table_task = visualize_table_op(
            training_dataframe_string, testing_dataframe_string
        )
        visualize_table_task.after(dataframe_task)

        train_task = train_op(
            training_dataframe_string=training_dataframe_string,
            testing_dataframe_string=testing_dataframe_string,
            cache_buster=cache_buster_break,
        )

    with dsl.Condition(secret_task.output != "", "Use-Production-Data"):
        get_dataframe_live_op = func_to_container_op(
            func=get_dataframes_live,
            base_image="python:3.9-slim-buster",
            packages_to_install=[
                "SQLAlchemy>=1.4.11",
                "psycopg2-binary>=2.8.6",
                "kubernetes==11.0.0",
                "requests==2.25.0",
                "scikit-learn>=0.24.1",
                "pandas>=1.1.5",
            ],
        )
        print(f"About to input: {str(secret_task.output)}")
        dataframe_task = get_dataframe_live_op(
            postgres_connection_string_b64=str(secret_task.output),
            percent_to_withhold_for_test=percent_to_withhold_for_test,
            cache_buster=cache_buster,
        )
        training_dataframe_string = str(
            dataframe_task.outputs["training_dataframe_string"]
        )
        testing_dataframe_string = str(
            dataframe_task.outputs["testing_dataframe_string"]
        )
        visualize_table_task = visualize_table_op(
            training_dataframe_string=training_dataframe_string,
            testing_dataframe_string=testing_dataframe_string,
            cache_buster=cache_buster,
        )
        visualize_table_task.after(dataframe_task)

        train_task = train_op(
            training_dataframe_string=training_dataframe_string,
            testing_dataframe_string=testing_dataframe_string,
            cache_buster=cache_buster_break,
        )
