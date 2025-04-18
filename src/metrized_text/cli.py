import click
from .processor import TextDetector


@click.command()
@click.option("--trained_model", type=click.Path(exists=True), required=True)
@click.option("--refiner_model", type=click.Path(exists=True), default=None)
@click.option("--test_folder", type=click.Path(exists=True), required=True)
@click.option("--cuda", is_flag=True, default=False)
@click.option("--canvas_size", default=300, type=int)
@click.option("--mag_ratio", default=1.0, type=float)
@click.option("--show_time", is_flag=True, default=False)
@click.option("--poly", is_flag=True, default=False)
@click.option("--link_threshold", default=0.4, type=float)
@click.option("--low_text", default=0.1, type=float)
@click.option("--text_threshold", default=0.2, type=float)
@click.option("--refine", is_flag=True, default=False)
def main(
    trained_model,
    refiner_model,
    test_folder,
    cuda,
    canvas_size,
    mag_ratio,
    show_time,
    poly,
    link_threshold,
    low_text,
    text_threshold,
    refine,
):
    detector = TextDetector(
        trained_model=trained_model,
        refiner_model=refiner_model,
        test_folder=test_folder,
        cuda=cuda,
        canvas_size=canvas_size,
        mag_ratio=mag_ratio,
        show_time=show_time,
        poly=poly,
        link_threshold=link_threshold,
        low_text=low_text,
        text_threshold=text_threshold,
        refine=refine,
    )
    detector.process_folder()


if __name__ == "__main__":
    main()
