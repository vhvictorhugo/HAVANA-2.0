import logging

import click


@click.group()
@click.option("--metapath", default="metadata.json", help="Path to metadata file", show_default=True)
@click.option("--state", help="State to execute the pipeline")
@click.option("--embedder", default="hex2vec", show_default=True, help="Embedder to generate embeddings")
@click.option(
    "--embeddings_dimension", default=10, show_default=True, help="Embeddings dimensions to generate region embeddings"
)
@click.option("--h3_resolution", default=9, show_default=True, help="H3 resolution for region embeddings")
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
def generate_model_inputs(ctx):
    from model_preprocess.job.matrix_generation_for_poi_categorization_job import (
        MatrixGenerationForPoiCategorizationJob,
    )

    state = ctx.obj["state"]
    metadata = ctx.obj["metadata"]
    logging.info("Starting model inputs generation for {state} state")
    MatrixGenerationForPoiCategorizationJob().run(state, metadata)
    logging.info("Successfully generated model inputs")


@cli.command
@click.pass_context
def generate_user_embeddings(ctx):
    from preprocess.EmbeddingsPreProcess import EmbeddingsPreProcess

    state = ctx.obj["state"]
    embeddings_dimension = ctx.obj["embeddings_dimension"]
    embedder = ctx.obj["embedder"]
    metadata = ctx.obj["metadata"]

    logging.info(
        f"Generating user embeddings for {state} state with {embeddings_dimension} dimensions using {embedder}"
    )

    EmbeddingsPreProcess(state, embeddings_dimension, embedder, metadata).run()
    logging.info("Successfully generated user embeddings")


@cli.command
@click.pass_context
def generate_hex2vec_embeddings(ctx):
    from preprocess.Hex2Vec import Hex2Vec

    state = ctx.obj["state"]
    embeddings_dimension = ctx.obj["embeddings_dimension"]
    h3_resolution = ctx.obj["h3_resolution"]
    metadata = ctx.obj["metadata"]

    logging.info(f"Generating Hex2Vec embeddings to {state} state with {embeddings_dimension} dimensions")

    Hex2Vec(state, embeddings_dimension, h3_resolution, metadata).run()
    logging.info("Successfully generated Hex2Vec embeddings")


@cli.command()
@click.pass_context
def preprocess_checkins(
    ctx,
):
    from preprocess.CheckinsPreProcess import CheckinsPreProcess

    state = ctx.obj["state"]
    metadata = ctx.obj["metadata"]
    logging.info("Starting preprocessing")
    logging.info(f"Preprocessing for {state} state")
    (CheckinsPreProcess(state, metadata).run())
    logging.info("Successfully preprocessed checkins data")


def main():
    cli(obj={})


if __name__ == "__main__":
    main()
