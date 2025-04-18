import click
from .processor import TextDetector


@click.command()
@click.option(
    "--trained_model",
    type=click.Path(exists=True),
    required=True,
    help="Path to the trained model (.pth)",
)
@click.option(
    "--refiner_model",
    type=click.Path(exists=True),
    default=None,
    help="Path to the refiner model (.pth)",
)
@click.option(
    "--test_folder",
    type=click.Path(exists=True),
    required=True,
    help="Folder containing test images",
)
@click.option("--cuda", is_flag=True, default=False, help="Use CUDA for inference")
@click.option("--canvas_size", default=200, type=int, help="Image resizing canvas size")
@click.option("--mag_ratio", default=1, type=float, help="Image magnification ratio")
@click.option("--show_time", is_flag=True, default=False, help="Print processing times")
@click.option(
    "--poly", is_flag=True, default=False, help="Output polygonal bounding boxes"
)
@click.option(
    "--link_threshold", default=0.4, type=float, help="Link confidence threshold"
)
@click.option(
    "--low_text",
    default=0.1,
    type=float,
    help="Low text confidence threshold for region growing",
)
@click.option(
    "--text_threshold", default=0.2, type=float, help="Text confidence threshold"
)
@click.option(
    "--refine", is_flag=True, default=False, help="Enable link refiner module"
)
def main(**kwargs):
    detector = TextDetector(kwargs)
    detector.process_folder()


if __name__ == "__main__":
    main()
