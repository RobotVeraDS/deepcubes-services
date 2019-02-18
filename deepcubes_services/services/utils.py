import os


def get_new_model_id(path):
    models_ids = [
        int(file_name.split('.')[0]) for file_name in os.listdir(path) if (
            '.cube' in file_name
        )
    ]

    sorted_ids = sorted(models_ids)
    new_model_id = sorted_ids[-1] + 1 if len(sorted_ids) else 0

    return new_model_id
