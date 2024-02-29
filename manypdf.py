
import torch
import PyPDF2 # pdf reader
import time
from pypdf import PdfReader
from io import BytesIO
from langchain.prompts import PromptTemplate # for custom prompt specification
from langchain.text_splitter import RecursiveCharacterTextSplitter # splitter for chunks
from langchain.embeddings import HuggingFaceEmbeddings # embeddings
from langchain.vectorstores import FAISS # vector store database
from langchain.chains import RetrievalQA # qa and retriever chain
from langchain.memory import ConversationBufferMemory # for model's memoy on past conversations
from langchain.document_loaders import PyPDFDirectoryLoader # loader fo files from firectory

from langchain.llms.huggingface_pipeline import HuggingFacePipeline # pipeline
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline, BitsAndBytesConfig

CHUNK_SIZE = 1000
# Using HuggingFaceEmbeddings with the chosen embedding model
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-mpnet-base-v2",model_kwargs = {"device": "cuda"})

# transformer model configuration
# this massively model's precision for memory effieciency
# The models accuacy is reduced.
quant_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
)
tensor_1 = torch.rand (4,4)

model_id = "Deci/DeciLM-7B-instruct" # model repo id
device = 'cuda' # Run on gpu if available else run on cpu

#
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id,
                                             trust_remote_code=True,
                                             device_map = "auto",
                                             quantization_config=quant_config)

# create a pipeline
pipe = pipeline("text-generation",
                model=model,
                tokenizer=tokenizer,
                return_full_text = True,
                max_new_tokens=200,
                repetition_penalty = 1.1,
                num_beams=5,
                no_repeat_ngram_size=4,
                early_stopping=True)

llm = HuggingFacePipeline(pipeline=pipe)

pdf_paths = "/home/acampton/manypdfs/content/"

loader = PyPDFDirectoryLoader(
    path= pdf_paths,
    glob="*.pdf"
)
documents=loader.load()

print(len(documents))
documents[0] # display four documents

documents[0].page_content, documents[0].metadata


text_splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE,
                                                chunk_overlap=100)

splits = text_splitter.split_documents(documents)

# length of all splits

print(f"We have, {len(splits)} chunks in memory")

vectorstore_db = FAISS.from_documents(splits, embeddings) # create vector db for similarity search

# performs a similarity check and returns the top K embeddings
# that are similar to the question’s embeddings
retriever = vectorstore_db.as_retriever(search_type="similarity", search_kwargs={"k": 6})

retrieved_relevant_docs = retriever.get_relevant_documents(
    "Based on the Deep unlearning pdf, what does class unlearning mean?"
)

print(f"Retrieved documents: {len(retrieved_relevant_docs)}")
f"Page content of first document:\n {retrieved_relevant_docs[0].page_content}"


retrieved_relevant_docs = retriever.get_relevant_documents(
    """Based on Eight things to know about large language models pdf,
    where do much of the expense of developing an LLM go?"""
)

print(f"Retrieved documents: {len(retrieved_relevant_docs)}")
f"Page content of first document:\n {retrieved_relevant_docs[0].page_content}"

custom_prompt_template = """You are an assistant for question-answering tasks. Use the following pieces of retrieved context and answer the question at the end.
If you don't know the answer just say you d not know an do not try to make up the answer nor try to use outside souces to answer. Keep the answer as concise as possible.
Context= {context}
History = {history}
Question= {question}
Helpful Answer:
"""

prompt = PromptTemplate(template=custom_prompt_template,
                        input_variables=["question", "context", "history"])


qa_chain_with_memory = RetrievalQA.from_chain_type(llm=llm, chain_type='stuff',
                                                   retriever = vectorstore_db.as_retriever(),
                                                   return_source_documents = True,
                                                   chain_type_kwargs = {"verbose": True,
                                                                        "prompt": prompt,
                                                                        "memory": ConversationBufferMemory(
                                                                            input_key="question",
                                                                            memory_key="history",
                                                                            return_messages=True)})



query = "Based on the Deep unlearning pdf, what is deep unlearning?"
qa_chain_with_memory({"query": query})

qa_chain_with_memory({"query": "What are the key challenges involved?"})

print(qa_chain_with_memory.combine_documents_chain.memory)

import gradio as gr

def load_llm():
    # Loads the  DeciLM-7B-instruct llm when called
    model_id = "Deci/DeciLM-7B-instruct"
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id,
                                                 trust_remote_code=True,
                                                 device_map = "auto",
                                                 quantization_config=quant_config)
    pipe = pipeline("text-generation",
                    model=model,
                    tokenizer=tokenizer,
                    temperature=0,
                    num_beams=5,
                    no_repeat_ngram_size=4,
                    early_stopping=True,
                    max_new_tokens=100,
                )

    llm = HuggingFacePipeline(pipeline=pipe)
    return llm

def add_text(history, text):
  # Adding user query to the chatbot and chain
  # use history with curent user question
  if not text:
      raise gr.Error('Enter text')
  history = history + [(text, '')]
  return history

def upload_file(files):
  # Loads files when the file upload button is clicked
  # Displays them on the File window
  # print(type(file))
  return files

def process_file(files):

    """Function reads each loaded file, and extracts text from each of their pages
    The extracted text is store in the 'text variable which is the passed to the splitter
    to make smaller chunks necessary for easier information retrieval and adhere to max-tokens(4096) of DeciLM-7B-instruct"""

    pdf_text = ""
    for file in files:
      pdf = PyPDF2.PdfReader(file.name)
      for page in pdf.pages:
          pdf_text += page.extract_text()


    # split into smaller chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=200)
    splits = text_splitter.create_documents([pdf_text])

    # create a FAISS vector store db
    # embedd the chunks and store in the db
    vectorstore_db = FAISS.from_documents(splits, embeddings)

    #create a custom prompt
    custom_prompt_template = """You have been given the following documents to answer the user's question.
    If you do not have information from the files given to answer the questions just say I don't have information from the given files to answer. Do not try to make up an answer.
    Context: {context}
    History: {history}
    Question: {question}

    Helpful answer:
    """
    prompt = PromptTemplate(template=custom_prompt_template, input_variables=["question", "context", "history"])

    # set QA chain with memory
    qa_chain_with_memory = RetrievalQA.from_chain_type(llm=load_llm(),
                                                       chain_type='stuff',
                                                       return_source_documents=True,
                                                       retriever=vectorstore_db.as_retriever(),
                                                       chain_type_kwargs={"verbose": True,
                                                                          "prompt": prompt,
                                                                          "memory": ConversationBufferMemory(
                                                                              input_key="question",
                                                                              memory_key="history",
                                                                              return_messages=True) })
    # get answers
    return qa_chain_with_memory

def generate_bot_response(history,query, btn):
  """Fiunction takes the query, history and inputs from the qa chain when the submit button is clicked
  to generate a response to the query"""

  if not btn:
      raise gr.Error(message='Upload a PDF')

  qa_chain_with_memory = process_file(btn) # run the qa chain with files from upload
  bot_response = qa_chain_with_memory({"query": query})
  # simulate streaming
  for char in bot_response['result']:
          history[-1][-1] += char
          time.sleep(0.05)
          yield history,''

# The GRADIO Interface
with gr.Blocks() as demo:
    with gr.Row():
            with gr.Row():
              # Chatbot interface
              chatbot = gr.Chatbot(label="DeciLM-7B-instruct bot",
                                   value=[],
                                   elem_id='chatbot')
            with gr.Row():
              # Uploaded PDFs window
              file_output = gr.File(label="Your PDFs")

              with gr.Column():
                # PDF upload button
                btn = gr.UploadButton("📁 Upload a PDF(s)",
                                      file_types=[".pdf"],
                                      file_count="multiple")

    with gr.Column():
        with gr.Column():
          # Ask question input field
          txt = gr.Text(show_label=False, placeholder="Enter question")

        with gr.Column():
          # button to submit question to the bot
          submit_btn = gr.Button('Ask')

    # Event handler for uploading a PDF
    btn.upload(fn=upload_file, inputs=[btn], outputs=[file_output])

    # Event handler for submitting text question and generating response
    submit_btn.click(
        fn= add_text,
        inputs=[chatbot, txt],
        outputs=[chatbot],
        queue=False
        ).success(
          fn=generate_bot_response,
          inputs=[chatbot, txt, btn],
          outputs=[chatbot, txt]
        ).success(
          fn=upload_file,
          inputs=[btn],
          outputs=[file_output]
        )

if __name__ == "__main__":
    demo.launch() #share=True) # launch app

# pip install langchain transformers accelerate tiktoken openai gradio \
#  torch accelerate safetensors sentence-transformers faiss-gpu \
#  bitsandbytes pypdf typing-extensions
# pip uninstall typing-extensions --yes
# pip install typing-extensions
# pip install PyPDF2
# conda create -n manypdfs python=3.11
