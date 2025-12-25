import re
from utils.config import Config
from langchain_text_splitters import RecursiveCharacterTextSplitter

def clean_scientific_text(text): # Only tested for pymupdf4llm for now, no idea how mark-pdf ouput looks like
    # Fix artificially broken words
    text = re.sub(r'(\w+)-\s*\n\s*(\w+)', r'\1\2', text)
    
    text = re.sub(r'(\w+)-\s+(\w+)', r'\1-\2', text)

    text = re.sub(r"\*\*==> picture \[\d+\s*x\s*\d+\] intentionally omitted <==\*\*", "", text)
        
    text = re.sub(r'\s+', ' ', text) # Normalize spaces
    
    return text.strip()
    
def chunk_text(text):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=Config.CHUNK_SIZE,
        chunk_overlap=Config.CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""], 
        length_function=len
    )

    chunks = splitter.split_text(text)

    """chunks = []
    start = 0
    while start < len(text):
        end = start + Config.CHUNK_SIZE
        chunk = text[start:end]
        chunks.append(chunk)
        start += (Config.CHUNK_SIZE - Config.CHUNK_OVERLAP)"""
    return chunks

def get_marker_converter():
    print("Loading heavy Marker models into RAM (this takes a moment)...")
    from marker.converters.pdf import PdfConverter
    from marker.models import create_model_dict
    from marker.config.parser import ConfigParser

    config_parser = ConfigParser(options={
        "extract_images": False,
        "force_ocr": False,
        "skip_bad_ocr": True, 
        "use_ocr_region_threshold": 0.05,
        "output_format": "markdown",
        "disable_image_extraction": True
    })
    
    marker_converter = PdfConverter(
        config=config_parser.generate_config_dict(),
        artifact_dict=create_model_dict()
    )
    return marker_converter

def process_pdf(file_path):
    filename = file_path.split("/")[-1]
    parser_type = Config.PARSER_TYPE.lower()
    
    all_chunks_with_metadata = []

    if parser_type == "marker":
        # Right now this doesn't work on a reasonable timespan on CPU, marker seems
        # to be forcing OCR, whixh is very expensive
        print(f"Using Marker for {filename}...")
        converter = get_marker_converter()
        rendered = converter(file_path)
        pages = re.split(r'\n- - -\n|\f', rendered.markdown)
        
        for i, page_content in enumerate(pages):
            clean_text = clean_scientific_text(page_content)
            for chunk in chunk_text(clean_text):
                all_chunks_with_metadata.append({
                    "text": chunk,
                    "metadata": {"source": filename, "page": i + 1, "method": "marker"}
                })

    else:
        print(f"Using PyMuPDF4LLM for {filename}...")
        import pymupdf.layout
        import pymupdf4llm
        pages_data = pymupdf4llm.to_markdown(file_path, page_chunks=True, header=False, footer=False)
        
        for i, page in enumerate(pages_data):
            page_num = i + 1
            clean_text = clean_scientific_text(page['text'])
            print("\n\n -------- \n\n")
            print(clean_text)
            for chunk in chunk_text(clean_text):
                all_chunks_with_metadata.append({
                    "text": chunk,
                    "metadata": {"source": filename, "page": page_num, "method": "pymupdf4llm"},
                })

    return all_chunks_with_metadata


if __name__ == "__main__":
    import os
    process_pdf(os.path.join(Config.DATA_DIR,'2012_Stinson.pdf'))