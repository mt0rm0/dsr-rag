"""Launch the full Gradio RAG interface."""

from ragsst.interface import make_interface
from ragsst.ragtool import RAGTool


def main() -> None:
    tool = RAGTool()
    tool.setup_vec_store()
    gui = make_interface(tool)
    gui.launch(show_api=False)


if __name__ == '__main__':
    main()
