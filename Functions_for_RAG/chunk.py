try:
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    from Functions_for_RAG.loaddata import docs
except ImportError as i:
    print("Import Error:\n", i)
except Exception as e:
    print("Error while Importing:\n", e)


try:
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    # splitting document on the instance created
    splitted = text_splitter.split_documents(docs)

except Exception as e:
    print("Error Occured during splitting data ", e)
