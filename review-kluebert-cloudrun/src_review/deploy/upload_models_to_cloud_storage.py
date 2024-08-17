"""
Depreciated: Can't load .bin files from GCS.
"""

from pathlib import Path
from google.cloud.exceptions import NotFound
from google.cloud.storage import Client, transfer_manager


def create_bucket_if_not_exists(bucket_name, region="asia-northeast3"):
    """Create a new bucket or retrieve an existing bucket by name."""
    storage_client = Client()
    try:
        bucket = storage_client.get_bucket(bucket_name)
        print(f"Bucket {bucket_name} already exists")
    except NotFound:
        bucket = storage_client.create_bucket(bucket_name, location=region)
        print(f"Bucket {bucket_name} created at {region}")
        
    return bucket

def upload_directory_with_transfer_manager(bucket_name, source_directory, workers=1):
    """Upload every file in a directory, including all files in subdirectories.

    Each blob name is derived from the filename, not including the `directory`
    parameter itself. For complete control of the blob name for each file (and
    other aspects of individual blob metadata), use
    transfer_manager.upload_many() instead.
    """

    # The ID of your GCS bucket
    # bucket_name = "your-bucket-name"

    # The directory on your computer to upload. Files in the directory and its
    # subdirectories will be uploaded. An empty string means "the current
    # working directory".
    # source_directory=""

    # The maximum number of processes to use for the operation. The performance
    # impact of this value depends on the use case, but smaller files usually
    # benefit from a higher number of processes. Each additional process occupies
    # some CPU and memory resources until finished. Threads can be used instead
    # of processes by passing `worker_type=transfer_manager.THREAD`.
    # workers=8

    bucket = create_bucket_if_not_exists(bucket_name)
    meta_blob = bucket.blob("meta.bin")
    model_blob = bucket.blob("pytorch_model.bin")
    generation_match_precondition = 0
    meta_blob.upload_from_filename("./tmp/meta.bin", if_generation_match=generation_match_precondition)
    model_blob.upload_from_filename("./tmp/pytorch_model.bin", if_generation_match=generation_match_precondition)
    # Generate a list of paths (in string form) relative to the `directory`.
    # This can be done in a single list comprehension, but is expanded into
    # multiple lines here for clarity.

    # First, recursively get all files in `directory` as Path objects.
    # directory_as_path_obj = Path(source_directory)
    # paths = directory_as_path_obj.rglob("*.bin")

    # # Filter so the list only includes files, not directories themselves.
    # file_paths = [path for path in paths if path.is_file()]

    # # These paths are relative to the current working directory. Next, make them
    # # relative to `directory`
    # relative_paths = [path.relative_to(source_directory) for path in file_paths]

    # # Finally, convert them all to strings.
    # string_paths = [str(path) for path in relative_paths]

    # print("Found {} files.".format(len(string_paths)))
    # for path in string_paths:
    #     print(path)

    # # Start the upload.
    # results = transfer_manager.upload_many_from_filenames(
    #     bucket, string_paths, source_directory=source_directory, max_workers=workers, skip_if_exists=False # 덮어쓰기
    # )

    # for name, result in zip(string_paths, results):
    #     # The results list is either `None` or an exception for each filename in
    #     # the input list, in order.

    #     if isinstance(result, Exception):
    #         print("Failed to upload {} due to exception: {}".format(name, result))
    #     else:
    #         print("Uploaded {} to {}.".format(name, bucket.name))
    
    
    

if __name__ == '__main__':    
    upload_directory_with_transfer_manager('review_tagging', './tmp/')