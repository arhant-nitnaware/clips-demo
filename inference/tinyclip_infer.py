from utils.timers import Timer

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
            "image_size": image.size,
            "num_labels": len(labels)
        }
    }