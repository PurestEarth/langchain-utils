import os
from gitignore_parser import parse_gitignore
from langchain.chat_models import ChatOpenAI
from langchain.callbacks import get_openai_callback
from langchain.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains import ConversationalRetrievalChain


def read_directory_with_gitignore(directory):
    """_summary_

    Args:
        directory (_type_):

    Returns:
        list(str): files not ignored by .gitignore
    """
    gitignore_path = os.path.join(directory, '.gitignore')

    # Parse the .gitignore file
    gitignore = parse_gitignore(gitignore_path)

    file_list = []
    for root, dirs, files in os.walk(directory):
        # Exclude files that match the patterns in .gitignore
        files = [f for f in files if not gitignore(os.path.join(root, f))]

        # Exclude directories that match the patterns in .gitignore
        dirs[:] = [d for d in dirs if not gitignore(os.path.join(root, d))]

        file_list.extend(os.path.join(root, f) for f in files)

    return list(filter(lambda x: '.git' not in x, file_list))


project_dir = '../EasyFaiss'


valid_files = read_directory_with_gitignore(project_dir)

docs = []


for file in valid_files:
    if file.endswith(".py"):
        try:
            loader = TextLoader(file, encoding="utf-8")
            docs.extend(loader.load_and_split())
        except FileNotFoundError as e:
            print(e)


text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
texts = text_splitter.split_documents(docs)


embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2")


db = FAISS.from_documents(texts, embeddings)
retriever = db.as_retriever(search_kwargs={"k": 5})

llm = ChatOpenAI(model='gpt-3.5-turbo', temperature=0, openai_api_key='')


qa = ConversationalRetrievalChain.from_llm(
    llm, retriever=retriever, return_source_documents=True)

query = "How do I clusterize"
with get_openai_callback() as cb:
    response = qa({"question": query, "verbose": True, "chat_history": []})
    print(response)
    print(cb)
