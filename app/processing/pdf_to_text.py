import fitz  # PyMuPDF
import pdfplumber
from bangla_pdf_ocr import process_pdf
from langchain_community.document_loaders import PyPDFLoader

def pdf2text_langchain(pdf_filepath, output_text_file_path):
    try:
        # Load single PDF
        loader = PyPDFLoader(pdf_filepath)
        papers = loader.load()
    except Exception as e:
        print(f"Error loading file: {e}")
        return None

    # Merge page contents
    full_text = '\n'.join([paper.page_content for paper in papers])
    print('text: ', full_text[:500])  # print only first 500 chars for sanity check

    # Save text to file
    try:
        with open(output_text_file_path, "w", encoding="utf-8") as file:
            file.write(full_text)
        print(f"Content saved to {output_text_file_path}")
        return full_text
    except Exception as e:
        print(f"Error saving file: {e}")
        return None

def pdf2text_pdfplumber(pdf_filepath, output_text_file_path):
    full_text = ""
    try:
        with pdfplumber.open(pdf_filepath) as pdf:
            for page in pdf.pages:
                text = page.extract_text()
                if text:
                    full_text += text + "\n"

        # Save
        with open(output_text_file_path, "w", encoding="utf-8") as f:
            f.write(full_text)

        print(f"Extracted text saved to {output_text_file_path}")
        return full_text
    except Exception as e:
        print(f"Error extracting text: {e}")
        return None

def pdf2text_pymupdf(pdf_filepath, output_text_file_path):
    full_text = ""
    doc = fitz.open(pdf_filepath)
    for page in doc:
        text = page.get_text("text")  # "text" keeps Unicode if available
        full_text += text + "\n"

    with open(output_text_file_path, "w", encoding="utf-8") as f:
        f.write(full_text)

    print(f"Text saved to {output_text_file_path}")
    return full_text

if __name__ == '__main__':
    pdf_filepath = "app/data/pdfs/taha.pdf"   # single PDF file
    output_text_file_path = "app/data/texts/prospectus.txt"
    prospectus_langchain = "app/data/texts/prospectus_langchain.txt"
    prospectus_pdfplumber = "app/data/texts/prospectus_pdfplumber.txt"
    prospectus_pymupdf = "app/data/texts/prospectus_pymupdf.txt"

    text = pdf2text_langchain(pdf_filepath, prospectus_langchain)
    # text = pdf2text_pdfplumber(pdf_filepath, prospectus_pdfplumber)
    # text = pdf2text_pymupdf(pdf_filepath, prospectus_pymupdf)
    # text = process_pdf(pdf_filepath, output_text_file_path)
    print('PDF text: ', text[:500])  # print only first 500 chars for sanity check
