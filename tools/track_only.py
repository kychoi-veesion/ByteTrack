import argparse
from pathlib import Path
import configparser

import pandas as pd
from yolox.tracker.byte_tracker import BYTETracker


def make_args():
    parser = argparse.ArgumentParser("Tracker only evaluation")
    parser.add_argument("sequence", type=Path, help="MOT20 sequence.ini file")
    parser.add_argument("det", type=Path, help="detection results")
    parser.add_argument("--output", type=Path, default=None,
                        help="output tracking results path")
    # tracking args
    parser.add_argument("--track_thresh", type=float, default=0.5,
                        help="tracking confidence threshold")
    parser.add_argument("--track_buffer", type=int, default=30,
                        help="the frames for keep lost tracks")
    parser.add_argument("--match_thresh", type=float, default=0.8,
                        help="matching threshold for tracking")
    parser.add_argument("--aspect_ratio_thresh", type=float, default=1.6,
                        help="threshold for filtering out boxes of which "
                             "aspect ratio are above the given value.")
    parser.add_argument('--min_box_area', type=float, default=10,
                        help='filter out tiny boxes')
    parser.add_argument("--mot20", dest="mot20", default=False,
                        action="store_true", help="test mot20.")
    args = parser.parse_args()
    return args


def main(args):
    config = configparser.ConfigParser()
    config.read(args.sequence)
    fps = float(config['Sequence']['frameRate'])
    tracker = BYTETracker(args, frame_rate=fps)
    names = ["frame", "id", "bb_left", "bb_top", "bb_width", "bb_height",
             "conf", "x", "y", "z"]
    det = pd.read_csv(str(args.det), names=names)
    det['bb_right'] = det['bb_left'] + det['bb_width']
    det['bb_bottom'] = det['bb_top'] + det['bb_height']

    im_info = (int(config['Sequence']['imHeight']),
               int(config['Sequence']['imWidth']))
    results = []
    for frame_id in range(1, int(config['Sequence']['seqLength']) + 1):
        framedet = det[det.frame == frame_id]
        if len(framedet) > 0:
            outputs = framedet[
                ['bb_left', 'bb_top', 'bb_right', 'bb_bottom', 'conf']]
            outputs = outputs.to_numpy(dtype=float)
            online_targets = tracker.update(outputs, im_info, im_info)
            online_tlwhs = []
            online_ids = []
            online_scores = []
            for t in online_targets:
                tlwh = t.tlwh
                tid = t.track_id
                vertical = tlwh[2] / tlwh[3] > args.aspect_ratio_thresh
                if tlwh[2] * tlwh[3] > args.min_box_area and not vertical:
                    online_tlwhs.append(tlwh)
                    online_ids.append(tid)
                    online_scores.append(t.score)
                    # save results
                    results.append(
                        f"{frame_id},{tid},{tlwh[0]:.2f},{tlwh[1]:.2f},"
                        f"{tlwh[2]:.2f},{tlwh[3]:.2f},{t.score:.2f},-1,-1,-1\n"
                    )
    if args.output:
        if not args.output.parent.exists():
            args.output.parent.mkdir(parents=True)
        with args.output.open('w') as f:
            f.writelines(results)


if __name__ == "__main__":
    main(make_args())
