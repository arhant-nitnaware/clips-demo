from utils.timers import Timer
from utils.device import DEVICE

def run_tinyclip(
    pipe,
    image,
    labels
):

    with Timer() as timer:

        results = pipe(
            image,
            candidate_labels=labels
        )

    return {
        "time_taken": timer.elapsed,
        "results": results,
        "input_details": {
            "Image Size": image.size
        }
    }