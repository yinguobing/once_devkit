from argparse import ArgumentParser
import numpy as np
import rerun as rr
import rerun.blueprint as rrb
from once import ONCE

# Add argument for dataset root path
parser = ArgumentParser()
parser.add_argument("--dataset_root", type=str, required=True)
args = parser.parse_args()


if __name__ == "__main__":
    # Init a visualizer
    rr.init("once", spawn=True)

    # Set data splits to be used
    splits = ["train", "val", "test"]
    dataset = ONCE(args.dataset_root, splits)

    # Create a blueprint
    blueprint = rrb.Blueprint(
        rrb.Vertical(
            contents=[
                rrb.Spatial3DView(
                    name=f"3D",
                    origin=f"/",
                ),
                rrb.Grid(
                    name="2D",
                    contents=[
                        rrb.Spatial2DView(
                            name=f"{cam_tag}",
                            origin=f"/images/{cam_tag}",
                        )
                        for cam_tag in dataset.camera_tags
                    ],
                ),
            ],
        )
    )
    rr.send_blueprint(blueprint)

    seq_id = list(dataset.train_split_list)[0]
    frame_ids = list(dataset.train_info[seq_id].keys())
    for frame_id in frame_ids:

        # Labeled boxes
        annos = dataset.get_frame_anno(seq_id, frame_id)
        if annos is None:
            continue

        names, boxes_3d, _ = (
            annos["names"],
            np.array(annos["boxes_3d"]),
            annos["boxes_2d"],
        )
        centers = boxes_3d[:, :3]
        sizes = boxes_3d[:, 3:6]
        rot_angles = [
            rr.RotationAxisAngle([0, 0, 1], rr.Angle(r)) for r in boxes_3d[:, -1]
        ]
        colors = np.ones_like(centers, dtype=np.uint8)
        colors[:, 1] *= 255
        rr.set_time_sequence("main", int(frame_id))
        rr.log(
            "/boxes",
            rr.Boxes3D(
                sizes=sizes,
                centers=centers,
                colors=colors,
                radii=0.01,
                labels=names,
                rotation_axis_angles=rot_angles,
            ),
        )

        # Point cloud
        points = dataset.load_point_cloud(seq_id, frame_id)
        rr.set_time_sequence("main", int(frame_id))
        rr.log(
            "/points", rr.Points3D(points[:, :3], colors=(255, 255, 255), radii=0.01)
        )

        # Images
        images = [
            dataset.load_image(seq_id, frame_id, name) for name in dataset.camera_names
        ]
        for i, img in enumerate(images):
            rr.set_time_sequence("main", int(frame_id))
            camera_name = dataset.camera_tags[i]
            rr.log(f"/images/{camera_name}", rr.Image(img))
