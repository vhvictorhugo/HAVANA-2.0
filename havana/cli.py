import click


@click.group()
@click.pass_context
def cli(ctx):
    ctx.obj["param"] = "default"


@cli.command()
@click.option("--param")
@click.pass_context
def function(ctx, param):
    if ctx.obj["param"] != "default":
        print("Do something")
        return
    print("Do something else")


def main():
    cli(obj={})


if __name__ == "__main__":
    main()
