try:
    from pathlib import Path
    from langchain_community.document_loaders import PyPDFLoader
except ImportError as i:
    print("Import Error:\n", i)
except Exception as e:
    print("Error While Importing:\n", e)


try:
    inputpdf_path = input("Enter the Path of Input PDF: ")
except Exception as e:
    print("input path is not given ", e)
try:
    pdf_path = Path(__file__).parent / "RAG_SAMPLE_DATA.pdf"
    print("Resolved path:", pdf_path)
    print("Exists:", pdf_path.exists())
    print("Is file:", pdf_path.is_file())

    loader = PyPDFLoader(str(pdf_path))
    docs = loader.load()

except Exception as e:
    print(
        f"Error in loading file {e} Check whether File at given path exist or no and check the name of file"
    )
