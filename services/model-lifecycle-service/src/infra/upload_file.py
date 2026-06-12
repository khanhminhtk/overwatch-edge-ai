import requests


def upload_file(file_path, url, headers=None, timeout=60):
    with open(file_path, "rb") as file_handle:
        return requests.put(
            url,
            data=file_handle,
            headers=headers,
            timeout=timeout,
        )
