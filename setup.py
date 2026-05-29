from setuptools import setup, find_packages

setup(
    name="chunkbench",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "chromadb>=0.4.0",
        "sentence-transformers>=2.7.0",
        "pypdf>=4.0.0",
        "python-docx>=1.1.0",
        "markdown>=3.6",
        "beautifulsoup4>=4.12.0",
        "nltk>=3.8.0",
        "tiktoken>=0.7.0",
        "openai>=1.30.0",
        "plotly>=5.15.0",
        "pandas>=2.0.0",
        "gradio>=4.0.0",
        "click>=8.1.0",
        "pydantic>=2.0.0",
        "tqdm>=4.66.0",
        "PyYAML>=6.0.1",
        "jsonlines>=4.0.0",
        "tabulate>=0.9.0",
        "transformers>=4.30.0",
    ],
    extras_require={
        "anthropic": ["anthropic>=0.28.0"],
    },
    entry_points={
        "console_scripts": ["chunkbench=chunkbench.cli:main"],
    },
    python_requires=">=3.10",
    author="ChunkBench Team",
    description="Benchmark every RAG chunking strategy, not just guess.",
    license="MIT",
)
