import base64
from io import BytesIO


def _get_encoded(buffer: BytesIO):
    buffer.seek(0)
    return base64.b64encode(buffer.read()).decode("utf-8")


def build_image_block(buffer: BytesIO, mime_type: str, extra: dict) -> dict:
    encoded = _get_encoded(buffer)
    return {
        "type": "image_url",
        "image_url": {"url": f"data:{mime_type};base64,{encoded}"},
    }


def build_document_block(buffer: BytesIO, mime_type: str, extra: dict) -> dict:
    encoded = _get_encoded(buffer)
    filename = extra.get("filename", "document")
    return {
        "type": "file",
        "file": {
            "filename": filename,
            "file_data": f"data:{mime_type};base64,{encoded}",
        },
    }
