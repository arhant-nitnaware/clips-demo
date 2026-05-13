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