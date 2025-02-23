from llama_index.core import SimpleDirectoryReader, VectorStoreIndex
from llama_index.core.indices.document_summary import DocumentSummaryIndex
from llama_index.core.response_synthesizers import TreeSummarize
from fastapi import FastAPI, UploadFile

from langchain.output_parsers import PydanticOutputParser

from pydantic import BaseModel, Field
from typing import List

def get_summary_and_else(file_path):
    # Load and split document into pages
    reader = SimpleDirectoryReader(input_files=[file_path])
    documents = reader.load_data()
    
    # Create page summaries using DocumentSummaryIndex
    summary_index = DocumentSummaryIndex.from_documents(
        documents,
        response_mode="tree_summarize",
        show_progress=True,
        summary_query = (
                        "You are an expert document simplifier + summarizer/n"
                        "Given the following page's context, return the following results in a json:"
                        "summary: Comprehensive summary of all the pages.\n"
                        "complexity_rating: Assign a score from 1 to 10 where 10 is the most complex a document can get for the average person.\n"
                        "red_flag_detection: Identify any potential risky clauses\n"
                        "figures_extraction: Extract any necessary financial amounts/deadlines\n"
                        "loopholes: Flag vague or ambiguous terms\n."
                        "Return only the requested information in json format, nothing else.\n"
                        )
    )
    
    # Extract individual page summaries
    page_summaries = [
        summary_index.get_document_summary(doc.doc_id) 
        for doc in documents
    ]
    
    parser = PydanticOutputParser(pydantic_object=DocumentAnalysis)

    # Generate final consolidated summary
    synthesizer = TreeSummarize()
    final_summary = synthesizer.get_response(
        ("You are an expert at simplifying complex documents and identifying red flags and loopholes\n"
        "Return the following results:\n"
        f"{parser.get_format_instructions()}"
        "The page summaries are given below:\n"),
        [s+"\n\n" for s in page_summaries if s is not None]
    )
    final_summary = parser.invoke(final_summary)

    vector_index = VectorStoreIndex.from_documents(documents)
    vector_index.storage_context.persist(persist_dir="db")
    return final_summary


class DocumentAnalysis(BaseModel):
    summary: str = Field(description="Comprehensive summary of the content")
    complexity_rating: int = Field(description="Complexity score from 1-10", ge=1, le=10)
    red_flag_detection: List[str] = Field(description="List of potential risky clauses")
    figures_extraction: List[str] = Field(description="Financial amounts/deadlines extracted")
    loopholes: List[str] = Field(description="Vague or ambiguous terms flagged")


app = FastAPI()

@app.get("/")
def hw():
    return 'Hello World'

@app.post("/process-legal-doc/")
def process_legal_document(file: UploadFile):
    file_path = file.filename
    res = get_summary_and_else(file_path)
    return res
