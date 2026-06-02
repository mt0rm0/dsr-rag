"""Command-line RAG interface."""

from ragsst.ragtool import RAGTool
import ragsst.parameters as p


def main() -> None:
    tool = RAGTool()
    tool.setup_vec_store()

    print(f'\n============== Local RAG (model: {tool.model}) ==============')
    print("Press 'q' to quit.\n")

    while True:
        user_input = input('Your prompt: ')
        if user_input.strip().lower() == 'q':
            break
        response = tool.rag_query(
            user_input,
            sim_th=0.4,
            nresults=3,
            top_k=5,
            top_p=0.9,
            temp=0.3,
        )
        print(f'\nAnswer:\n{response}\n')


if __name__ == '__main__':
    main()
