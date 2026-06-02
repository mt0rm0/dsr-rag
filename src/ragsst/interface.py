"""Gradio interface for the RAGTool."""

import os
from typing import Any

import gradio as gr

import ragsst.parameters as p
from ragsst.ragtool import RAGTool


def make_interface(ragsst: RAGTool) -> Any:
    """Build and return the full Gradio tabbed interface.

    Args:
        ragsst: An initialised :class:`~ragsst.ragtool.RAGTool` instance.

    Returns:
        A Gradio ``TabbedInterface`` ready to ``.launch()``.
    """

    pinfo = {
        'Rth':   'Set the relevance level for the content retrieval',
        'TopnR': 'Select the maximum number of passages to retrieve',
        'Top k': 'LLM parameter — higher values produce more varied text',
        'Top p': 'LLM parameter — higher values produce more varied text',
        'Temp':  'LLM parameter — higher values increase answer randomness',
    }

    # ------------------------------------------------------------------
    # RAG Query tab
    # ------------------------------------------------------------------
    rag_query_ui = gr.Interface(
        ragsst.rag_query,
        inputs=gr.Textbox(label='Query'),
        outputs=gr.Textbox(label='Answer', lines=14),
        description='Query an LLM about information from your documents.',
        allow_flagging='manual',
        flagging_dir=os.path.join(p.EXPORT_PATH, 'rag_query'),
        flagging_options=[('Export', 'export')],
        additional_inputs=[
            gr.Slider(0, 1,  value=0.5, step=0.1, label='Relevance threshold', info=pinfo['Rth']),
            gr.Slider(1, 5,  value=3,   step=1,   label='Top n results',       info=pinfo['TopnR']),
            gr.Slider(1, 10, value=5,   step=1,   label='Top k',               info=pinfo['Top k']),
            gr.Slider(0.1, 1, value=0.9, step=0.1, label='Top p', info=pinfo['Top p'], visible=False),
            gr.Slider(0.1, 1, value=0.3, step=0.1, label='Temp',               info=pinfo['Temp']),
        ],
        additional_inputs_accordion=gr.Accordion(label='Settings', open=False),
        clear_btn=None,
    )

    # ------------------------------------------------------------------
    # Semantic Retrieval tab
    # ------------------------------------------------------------------
    semantic_retrieval_ui = gr.Interface(
        ragsst.retrieve_with_metadata,
        inputs=gr.Textbox(label='Query'),
        outputs=gr.Textbox(label='Related Content', lines=20),
        description='Find information in your documents.',
        allow_flagging='manual',
        flagging_dir=os.path.join(p.EXPORT_PATH, 'semantic_retrieval'),
        flagging_options=[('Export', 'export')],
        additional_inputs=[
            gr.Slider(1, 5, value=3,   step=1,   label='Top n results',       info=pinfo['TopnR']),
            gr.Slider(0, 1, value=0.5, step=0.1, label='Relevance threshold', info=pinfo['Rth']),
        ],
        additional_inputs_accordion=gr.Accordion(label='Retrieval Settings', open=False),
        clear_btn=None,
    )

    # ------------------------------------------------------------------
    # RAG Chat tab
    # ------------------------------------------------------------------
    with gr.ChatInterface(
        ragsst.rag_chat,
        description='Query and interact with an LLM considering your documents.',
        chatbot=gr.Chatbot(height=500),
        additional_inputs=[
            gr.Slider(0, 1,  value=0.5, step=0.1, label='Relevance threshold', info=pinfo['Rth']),
            gr.Slider(1, 5,  value=3,   step=1,   label='Top n results',       info=pinfo['TopnR']),
            gr.Slider(1, 10, value=5,   step=1,   label='Top k',               info=pinfo['Top k']),
            gr.Slider(0.1, 1, value=0.9, step=0.1, label='Top p', info=pinfo['Top p'], visible=False),
            gr.Slider(0.1, 1, value=0.3, step=0.1, label='Temp',               info=pinfo['Temp']),
        ],
        additional_inputs_accordion=gr.Accordion(label='Settings', open=False),
        undo_btn=None,
    ) as rag_chat_ui:
        rag_chat_ui.clear_btn.click(ragsst.clear_ragchat_hist)

    # ------------------------------------------------------------------
    # Plain Chat tab
    # ------------------------------------------------------------------
    with gr.ChatInterface(
        ragsst.chat,
        description='Chat with the LLM directly, without document context.',
        chatbot=gr.Chatbot(height=500),
        additional_inputs=[
            gr.Slider(1, 10, value=5,   step=1,   label='Top k', info=pinfo['Top k']),
            gr.Slider(0.1, 1, value=0.9, step=0.1, label='Top p', info=pinfo['Top p']),
            gr.Slider(0.1, 1, value=0.5, step=0.1, label='Temp',  info=pinfo['Temp']),
        ],
        additional_inputs_accordion=gr.Accordion(label='LLM Settings', open=False),
        undo_btn=None,
    ) as chat_ui:
        chat_ui.clear_btn.click(ragsst.clear_chat_hist)

    # ------------------------------------------------------------------
    # Config tab
    # ------------------------------------------------------------------
    with gr.Blocks() as config_ui:

        def read_logs() -> str:
            log_path = os.path.join(p.LOG_DIR, p.LOG_FILE)
            with open(log_path, 'r') as f:
                return f.read()

        with gr.Row():
            with gr.Column(scale=3):

                def make_db(data_path: str, collection_name: str, embedding_model: str) -> None:
                    if not collection_name:
                        collection_name = p.COLLECTION_NAME
                    ragsst.set_data_path(data_path)
                    ragsst.set_embeddings_model(embedding_model)
                    ragsst.make_collection(data_path, collection_name)

                gr.Markdown('**Make and populate the Embeddings Database.**')

                with gr.Row():
                    with gr.Column():
                        data_path = gr.Textbox(
                            value=ragsst.data_path,
                            label='Documents Path',
                            info='Folder containing your documents',
                            interactive=True,
                        )
                    with gr.Column():
                        collection_choices = ragsst.list_collections_names()
                        collection_name = gr.Dropdown(
                            info='Choose a collection or type a name to create a new one (no spaces)',
                            choices=collection_choices,
                            allow_custom_value=True,
                            value=ragsst.collection_name,
                            label='Collection Name',
                            interactive=True,
                        )
                        with gr.Row():
                            setcollection_btn = gr.Button('Set Choice', size='sm')
                            deletecollection_btn = gr.Button('Delete', size='sm')

                        def update_collections_list(current_value: str) -> gr.Dropdown:
                            local_collections = ragsst.list_collections_names()
                            default = (
                                current_value
                                if current_value in local_collections
                                else (local_collections[0] if local_collections else None)
                            )
                            return gr.Dropdown(choices=local_collections, value=default, interactive=True)

                emb_model = gr.Dropdown(
                    choices=p.EMBEDDING_MODELS,
                    value=p.EMBEDDING_MODELS[0],
                    label='Embedding Model',
                    interactive=True,
                )

                setcollection_btn.click(ragsst.set_collection, inputs=[collection_name, emb_model])
                deletecollection_btn.click(ragsst.delete_collection, inputs=collection_name)

                with gr.Row():
                    makedb_btn = gr.Button('Make/Update Database', size='lg', scale=2)
                    deletedb_btn = gr.Button('Clean Database', size='lg', scale=1)

                info_output = gr.Textbox(read_logs, label='Info', lines=10, every=2)
                makedb_btn.click(fn=make_db, inputs=[data_path, collection_name, emb_model])
                deletedb_btn.click(fn=ragsst.clean_database)
                info_output.change(update_collections_list, collection_name, collection_name)

            with gr.Column(scale=2):
                gr.Markdown('**Choose the Language Model**')

                model_choices = ragsst.list_local_models()
                model_name = gr.Dropdown(
                    info='Choose a locally available LLM',
                    choices=model_choices,
                    allow_custom_value=True,
                    value=p.LLM_CHOICES[0],
                    label='Local LLM',
                    interactive=True,
                )
                setllm_btn = gr.Button('Set Choice', size='sm')
                setllm_btn.click(fn=ragsst.set_model, inputs=model_name)

                pull_model_name = gr.Dropdown(
                    info='Download a model (internet connection required)',
                    choices=p.LLM_CHOICES,
                    allow_custom_value=True,
                    value=p.LLM_CHOICES[0],
                    label='LLM to download',
                    interactive=True,
                )
                pull_btn = gr.Button('Download', size='sm')
                pull_info = gr.Textbox(label='Download status')
                pull_btn.click(fn=ragsst.pull_model, inputs=pull_model_name, outputs=pull_info)

                def update_local_models_list(progress_info: str) -> gr.Dropdown:
                    if 'success' in progress_info.lower():
                        return gr.Dropdown(
                            choices=ragsst.list_local_models(),
                            value=p.LLM_CHOICES[0],
                            interactive=True,
                        )
                    return model_name

                pull_info.change(update_local_models_list, pull_info, model_name)

    # ------------------------------------------------------------------
    # Assemble tabs
    # ------------------------------------------------------------------
    gui = gr.TabbedInterface(
        [rag_query_ui, semantic_retrieval_ui, rag_chat_ui, chat_ui, config_ui],
        ['RAG Query', 'Semantic Retrieval', 'RAG Chat', 'Chat', 'Settings'],
        title="<a href='https://github.com/aihpi/dsr-rag' target='_blank'>Local RAG Tool</a>",
    )
    return gui
