from PyPDF2 import PdfReader
from enum import Enum


class FileStatus(Enum):
    FILE_PARSED = "FILE_PARSED"
    FILE_NOT_SUPPORTED = "FILE_NOT_SUPPORTED"


def text_from_file(file):
    if file is not None and file.type == "application/pdf":
        pdf_reader = PdfReader(file)
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text()
        return text, FileStatus.FILE_PARSED
    else:
        return None, FileStatus.FILE_NOT_SUPPORTED
