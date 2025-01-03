import argparse
import matplotlib.pyplot as plt
from sam import (
    read_data,
    SysAdjustedSamPrediction,
    generate_sys_adjusted_sam_prediction,
    generate_sam_prediction,
    SAMPrediction,
)
from pathlib import Path
import json

ADD_STRESS = 0.18

with open("docs/add_sys_examples.json", "r") as f:
    examples = json.load(f)


def save_imgs(dir4imgs):
    save_path = Path(dir4imgs)
    save_path.mkdir(exist_ok=True, parents=True)
    for path, stressor_name, hormesis_index in examples:
        data = read_data(path)
        base_path = str(
            save_path / f"{data.meta.title}_{stressor_name}".replace(" ", "_")
        )
        ser = data.additional_stress[stressor_name]

        sam_fig = generate_sam_prediction(
            control_data=data.main_series,
            co_stressor_data=ser,
            meta=data.meta,
        ).plot()
        sam_fig.savefig(base_path + "_sam.png")
        plt.close(sam_fig)

        sys_fig = generate_sys_adjusted_sam_prediction(
            control_data=data.main_series,
            co_stressor_data=ser,
            additional_stress=ADD_STRESS,
            hormesis_index=4,
            meta=data.meta,
        ).plot(
            title=f"With {ADD_STRESS:.2f} additional Stress and hormesis index set to {hormesis_index}"
        )
        plt.tight_layout()
        sys_fig.savefig(base_path + "_sys.png")
        plt.close(sys_fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir4imgs", type=str)
    args = parser.parse_args()
    dir4imgs = Path(args.dir4imgs)
    dir4imgs.mkdir(exist_ok=True, parents=True)
    save_imgs(dir4imgs)
