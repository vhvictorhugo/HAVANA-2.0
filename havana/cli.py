import logging

import click


@click.group()
@click.option("--metapath", default="metadata.json", help="Path to metadata file", show_default=True, type=click.Path())
@click.option("--state", required=True, help="State to execute the pipeline", type=str)
@click.option("--embedder", help="Embedder to generate embeddings", type=str)
@click.option("--embeddings_dimension", help="Embeddings dimensions to generate region embeddings", type=int)
@click.option("--h3_resolution", help="H3 resolution for region embeddings", type=int)
@click.pass_context
def cli(
    ctx,
    metapath: str,
    state: str,
    embedder: str,
    embeddings_dimension: int,
    h3_resolution: int,
):
    import json

    logging.basicConfig(level=logging.INFO)
    logging.info("Starting CLI")
    logging.info(f"Reading metadata from {metapath}")

    with open(metapath) as file:
        ctx.obj["metadata"] = json.load(file)

    logging.info("Successfully read metadata")

    ctx.obj["state"] = state
    ctx.obj["embedder"] = embedder
    ctx.obj["embeddings_dimension"] = embeddings_dimension
    ctx.obj["h3_resolution"] = h3_resolution


@cli.command()
@click.pass_context
def mlflow(ctx):
    """
    Execute mlflow for a given state and dimension
    """
    from havana.mlflow.MLFlow import MLFlow

    state = ctx.obj["state"]
    embedder = ctx.obj["embedder"]
    if embedder is None:
        embedder = "baseline"
        h3_resolution = 0
        embeddings_dimension = 0
    else:
        h3_resolution = ctx.obj["h3_resolution"]
        embeddings_dimension = ctx.obj["embeddings_dimension"]
    metadata = ctx.obj["metadata"]
    logging.info(f"Starting mlflow execution for {state} state.")
    logging.info(f"MLFlow Params: {embedder} embedder, {h3_resolution} resolution, {embeddings_dimension} dimensions")
    MLFlow(state, embedder, h3_resolution, embeddings_dimension, metadata).run()
    logging.info("Successfully executed mlflow")


@cli.command()
@click.pass_context
def model(ctx):
    """Execute model for a given state"""
    from havana.model.job.poi_categorization_job import PoiCategorizationJob

    state = ctx.obj["state"]
    metadata = ctx.obj["metadata"]
    embedder = ctx.obj["embedder"]
    if embedder is None:
        embedder = "baseline"
        embeddings_dimension = 0
        h3_resolution = 0
    else:
        embeddings_dimension = ctx.obj["embeddings_dimension"]
        h3_resolution = ctx.obj["h3_resolution"]
    logging.info(f"Starting model execution for {state} state")
    execution_model_message = (
        "Executing baseline version" if (embedder == "baseline") else "Executing embeddings version"
    )
    logging.info(f"{execution_model_message} for {state} state.")
    logging.info(f"Model Params: {embedder} embedder, {h3_resolution} resolution, {embeddings_dimension} dimensions")
    PoiCategorizationJob().run(
        state=state,
        embedder=embedder,
        embeddings_dimension=embeddings_dimension,
        h3_resolution=h3_resolution,
        metadata=metadata,
    )


@cli.command()
@click.pass_context
def model_inputs(ctx):
    """Generate model default inputs for poi categorization"""
    from havana.model_preprocess.job.matrix_generation_for_poi_categorization_job import (
        MatrixGenerationForPoiCategorizationJob,
    )

    state = ctx.obj["state"]
    metadata = ctx.obj["metadata"]
    logging.info(f"Starting model default inputs generation for {state} state")
    MatrixGenerationForPoiCategorizationJob().run(state, metadata)
    logging.info("Successfully generated model inputs")


@cli.command
@click.pass_context
def user_embeddings(ctx):
    """Generate user embeddings for a given state and dimension"""
    from havana.embeddings.EmbeddingsPreProcess import EmbeddingsPreProcess

    state = ctx.obj["state"]
    embeddings_dimension = ctx.obj["embeddings_dimension"]
    embedder = ctx.obj["embedder"]
    h3_resolution = ctx.obj["h3_resolution"]
    metadata = ctx.obj["metadata"]

    logging.info(f"Generating user embeddings for {state} state.")
    logging.info(
        f"User Embeddings Params: {embedder} embedder, {h3_resolution} resolution, {embeddings_dimension} dimensions"
    )
    EmbeddingsPreProcess(state, embeddings_dimension, embedder, h3_resolution, metadata).run()
    logging.info("Successfully generated user embeddings")


@cli.command
@click.pass_context
def embedder(ctx):
    """Generate embeddings for a given embedder, state, dimension and h3 resolution"""
    from havana.embeddings.GeoVex import GeoVex
    from havana.embeddings.Hex2Vec import Hex2Vec

    embedder = ctx.obj["embedder"]
    state = ctx.obj["state"]
    embeddings_dimension = ctx.obj["embeddings_dimension"]
    h3_resolution = ctx.obj["h3_resolution"]
    metadata = ctx.obj["metadata"]

    logging.info(f"Generating {embedder.upper()} embeddings to {state} state.")
    logging.info(f"{embedder.upper()} Params: {h3_resolution} resolution, {embeddings_dimension} dimensions")

    embedder_params = {
        "state": state,
        "embeddings_dimension": embeddings_dimension,
        "h3_resolution": h3_resolution,
        "metadata": metadata,
    }

    if embedder == "hex2vec":
        embedder_instance = Hex2Vec(**embedder_params)
    elif embedder == "geovex":
        embedder_instance = GeoVex(**embedder_params)

    embedder_instance.run()
    logging.info(f"Successfully generated {embedder.upper()} embeddings")


@cli.command()
@click.pass_context
def preprocess(
    ctx,
):
    """Preprocess checkins data for a given state"""
    from havana.preprocess.CheckinsPreProcess import CheckinsPreProcess

    state = ctx.obj["state"]
    metadata = ctx.obj["metadata"]
    logging.info("Starting checkins preprocessing")
    logging.info(f"Preprocessing for {state} state")
    (CheckinsPreProcess(state, metadata).run())
    logging.info("Successfully preprocessed checkins data")


def main():
    cli(obj={})


if __name__ == "__main__":
    main()
