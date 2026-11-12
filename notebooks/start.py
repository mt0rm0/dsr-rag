"""Gradio interface for the RAGTool."""

from ragsst.interface import make_interface
from ragsst.ragtool import RAGTool

tool = RAGTool(
    model="llama3.2:3b",
    data_path="../data/sample_docs",
    collection_name="workshop_docs",
)

gui = make_interface(tool)
gui.launch()
