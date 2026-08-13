import os
import tarfile
import urllib.request

MODELS = {
    "clip": (
        "https://www.kaggle.com/api/v1/models/"
        "arhantn/clip/pyTorch/default/1/download"
    ),

    "clap": (
        "https://www.kaggle.com/api/v1/models/"
        "arhantn/clap/pyTorch/default/1/download"
    ),

    "tinyclip": (
        "https://www.kaggle.com/api/v1/models/"
        "arhantn/tinyclip/pyTorch/default/1/download"
    ),

    "clip4clip": (
        "https://www.kaggle.com/api/v1/models/"
        "arhantn/clip4clip/pyTorch/default/1/download"
    )
}


# ==========================================
# DOWNLOAD ONE MODEL
# ==========================================

def download_model(
    model_name,
    url
):

    model_dir = os.path.join(
        "/mnt/E/UG/cdac/implementation(s)/inference/decoupled/models_local",
        model_name
    )

    model_file = os.path.join(
        model_dir,
        "model.safetensors"
    )

    if os.path.exists(
        model_file
    ):

        print(
            f"[INFO] {model_name} already present."
        )

        return

    os.makedirs(
        model_dir,
        exist_ok=True
    )

    archive_path = os.path.join(
        model_dir,
        "model.tar.gz"
    )

    print(
        f"[INFO] Downloading {model_name}..."
    )

    urllib.request.urlretrieve(
        url,
        archive_path
    )

    print(
        f"[INFO] Extracting {model_name}..."
    )

    with tarfile.open(
        archive_path,
        "r:gz"
    ) as tar:

        tar.extractall(
            model_dir
        )

    os.remove(
        archive_path
    )

    print(
        f"[INFO] {model_name} ready."
    )


# ==========================================
# MAIN
# ==========================================

def main():

    os.makedirs(
        "/mnt/E/UG/cdac/implementation(s)/inference/decoupled/models_local",
        exist_ok=True
    )


    for (
        model_name,
        url
    ) in MODELS.items():

        try:

            download_model(
                model_name,
                url
            )

        except Exception as e:

            print(
                f"[ERROR] Failed to install "
                f"{model_name}: {e}"
            )

    print(
        "\n[INFO] Setup complete."
    )


if __name__ == "__main__":

    main()