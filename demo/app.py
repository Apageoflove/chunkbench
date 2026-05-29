from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import gradio as gr
import pandas as pd

from chunkbench.benchmark import Benchmark
from chunkbench.chunkers import ALL_STRATEGIES, STRATEGY_MAP
from chunkbench.loaders import DocumentLoader

STRATEGY_DESCRIPTIONS = {
    "fixed_size": "Fixed-size chunking splits text into equal-length segments with configurable overlap. Simple and fast, but may break sentences or logical units.",
    "recursive": "Recursive chunking splits text using a hierarchy of separators (paragraphs, lines, sentences, words). Produces more natural chunk boundaries.",
    "token_aware": "Token-aware chunking controls chunk size by exact token count while respecting sentence boundaries. Ideal for LLM context window management.",
    "sentence_window": "Sentence-window chunking indexes individual sentences but retrieves surrounding context windows. Provides precise matching with contextual expansion.",
    "semantic": "Semantic chunking groups sentences by embedding similarity, splitting at semantic breakpoints. Preserves topical coherence within chunks.",
    "late_chunking": "Late chunking embeds the full document first, then splits. Each chunk's embedding retains global context through attention mechanisms.",
    "proposition": "Proposition chunking decomposes text into atomic factual propositions using LLM. Each chunk is a self-contained, independently understandable fact.",
}


def run_benchmark(
    file_objs,
    questions_text,
    selected_strategies,
    embed_model,
    progress=gr.Progress(),
):
    if not file_objs:
        return None, None, None, "Please upload at least one document."

    doc_paths = []
    for f in file_objs:
        doc_paths.append(f.name)

    qa_pairs = []
    if questions_text.strip():
        for line in questions_text.strip().split("\n"):
            line = line.strip()
            if line:
                qa_pairs.append({"question": line, "answer": ""})

    try:
        bench = Benchmark(
            documents=doc_paths if len(doc_paths) == 1 else doc_paths,
            qa_pairs=qa_pairs if qa_pairs else None,
            strategies=list(selected_strategies) if selected_strategies else None,
            embed_model=embed_model,
            output_dir="./chunkbench_output",
            verbose=False,
        )
        report = bench.run()
    except Exception as e:
        return None, None, None, f"Error: {e}"

    df = report.leaderboard.to_dataframe()
    insights = "\n".join(f"- {i}" for i in report.leaderboard.get_insights())

    bar_fig = None
    scatter_fig = None
    try:
        from chunkbench.reporters.visualizer import BenchmarkVisualizer

        viz = BenchmarkVisualizer("./chunkbench_output")
        bar_fig = viz.plot_leaderboard_bar(report.leaderboard)
        scatter_fig = viz.plot_scatter_tradeoff(report.leaderboard)
    except Exception:
        pass

    return df, bar_fig, scatter_fig, insights


def show_strategy_detail(strategy_name, sample_text):
    if not sample_text.strip():
        sample_text = "This is a sample text. It contains multiple sentences. Each sentence expresses a different idea. The text demonstrates how chunking works."
    cls = STRATEGY_MAP.get(strategy_name)
    if cls is None:
        return None, f"Unknown strategy: {strategy_name}"

    try:
        if strategy_name == "semantic":
            chunker = cls()
        else:
            chunker = cls()
        chunks = chunker.chunk(sample_text, "demo")
    except Exception as e:
        return None, f"Error: {e}"

    rows = []
    for c in chunks:
        rows.append(
            {
                "ID": c.chunk_id,
                "Content": c.content[:80] + ("..." if len(c.content) > 80 else ""),
                "Tokens": c.metadata.get("token_count", 0),
                "Size": len(c.content),
            }
        )

    desc = STRATEGY_DESCRIPTIONS.get(strategy_name, "No description available.")
    return pd.DataFrame(rows), desc


def load_sample():
    base = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "examples")
    doc_path = os.path.join(base, "sample_docs", "technical.txt")
    qa_path = os.path.join(base, "sample_qa.jsonl")

    questions = []
    try:
        import json

        with open(qa_path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    obj = json.loads(line)
                    questions.append(obj["question"])
    except Exception:
        pass

    return doc_path, "\n".join(questions[:5])


with gr.Blocks(title="ChunkBench Demo", theme=gr.themes.Soft()) as app:
    gr.Markdown("# ChunkBench\n> Benchmark every RAG chunking strategy, not just guess.")

    with gr.Tabs():
        with gr.Tab("Quick Benchmark"):
            with gr.Row():
                file_input = gr.File(label="Upload Documents", file_count="multiple", file_types=[".txt", ".pdf", ".md"])
                questions_input = gr.Textbox(
                    label="Test Questions (one per line)",
                    lines=5,
                    placeholder="Enter questions...\nOr click 'Load Sample' below.",
                )
            with gr.Row():
                strategy_check = gr.CheckboxGroup(
                    choices=ALL_STRATEGIES,
                    value=["fixed_size", "recursive", "token_aware", "sentence_window"],
                    label="Strategies",
                )
                model_drop = gr.Dropdown(
                    choices=["all-MiniLM-L6-v2", "BAAI/bge-small-zh-v1.5"],
                    value="all-MiniLM-L6-v2",
                    label="Embedding Model",
                )
            with gr.Row():
                run_btn = gr.Button("Run Benchmark", variant="primary")
                sample_btn = gr.Button("Load Sample Data")

            leaderboard_df = gr.Dataframe(label="Leaderboard", interactive=False)
            with gr.Row():
                bar_plot = gr.Plot(label="Strategy Comparison")
                scatter_plot = gr.Plot(label="Accuracy vs Speed")
            insights_md = gr.Markdown(label="Insights")

            run_btn.click(
                run_benchmark,
                inputs=[file_input, questions_input, strategy_check, model_drop],
                outputs=[leaderboard_df, bar_plot, scatter_plot, insights_md],
            )
            sample_btn.click(
                load_sample,
                outputs=[file_input, questions_input],
            )

        with gr.Tab("Strategy Explorer"):
            with gr.Row():
                strat_drop = gr.Dropdown(
                    choices=ALL_STRATEGIES,
                    value="recursive",
                    label="Select Strategy",
                )
                sample_text = gr.Textbox(
                    label="Input Text",
                    lines=6,
                    value="This is a sample text. It contains multiple sentences. Each sentence expresses a different idea.",
                )
            explore_btn = gr.Button("Explore")
            strat_desc = gr.Markdown()
            chunks_df = gr.Dataframe(label="Chunks", interactive=False)

            explore_btn.click(
                show_strategy_detail,
                inputs=[strat_drop, sample_text],
                outputs=[chunks_df, strat_desc],
            )

if __name__ == "__main__":
    app.launch(server_name="0.0.0.0", server_port=7860)
