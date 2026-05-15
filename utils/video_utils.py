import cv2

def extract_frames(
    video_path,
    max_frames=8
):

    cap = cv2.VideoCapture(video_path)

    total_frames = int(
        cap.get(cv2.CAP_PROP_FRAME_COUNT)
    )

    step = max(
        total_frames // max_frames,
        1
    )

    frames = []

    idx = 0

    while cap.isOpened():

        ret, frame = cap.read()

        if not ret:
            break

        if idx % step == 0:

            frame = cv2.cvtColor(
                frame,
                cv2.COLOR_BGR2RGB
            )

            frames.append(frame)

            if len(frames) >= max_frames:
                break

        idx += 1

    cap.release()

    return frames

def get_video_frame_count(
    video_path
):

    cap = cv2.VideoCapture(
        video_path
    )

    if not cap.isOpened():

        return 0

    frame_count = int(
        cap.get(
            cv2.CAP_PROP_FRAME_COUNT
        )
    )

    cap.release()

    return frame_count

def extract_segment_frames(
    video_path,
    start_time,
    end_time,
    max_frames=8
):

    cap = cv2.VideoCapture(
        video_path
    )

    fps = cap.get(
        cv2.CAP_PROP_FPS
    )

    start_frame = int(
        start_time * fps
    )

    end_frame = int(
        end_time * fps
    )

    cap.set(
        cv2.CAP_PROP_POS_FRAMES,
        start_frame
    )

    total_segment_frames = (
        end_frame - start_frame
    )

    step = max(
        total_segment_frames
        // max_frames,
        1
    )

    frames = []

    idx = 0

    while cap.isOpened():

        current_frame = int(
            cap.get(
                cv2.CAP_PROP_POS_FRAMES
            )
        )

        if current_frame >= end_frame:
            break

        ret, frame = cap.read()

        if not ret:
            break

        if idx % step == 0:

            frame = cv2.cvtColor(
                frame,
                cv2.COLOR_BGR2RGB
            )

            frames.append(frame)

            if len(frames) >= max_frames:
                break

        idx += 1

    cap.release()

    return frames


def get_video_duration(
    video_path
):

    cap = cv2.VideoCapture(
        video_path
    )

    fps = cap.get(
        cv2.CAP_PROP_FPS
    )

    frame_count = int(
        cap.get(
            cv2.CAP_PROP_FRAME_COUNT
        )
    )

    cap.release()

    if fps <= 0:
        return 0

    return frame_count / fps