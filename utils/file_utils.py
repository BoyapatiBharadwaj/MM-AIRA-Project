import pdfplumber

def read_pdf(file):
    text = ""
    with pdfplumber.open(file) as pdf:
        for page in pdf.pages:
            text += page.extract_text()
    return text

def read_text_file(file):
    return file.read().decode("utf-8")
