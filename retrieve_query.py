import os

from asyncextractor.retriever import QueryRetriever


def compare_async(use_async: bool = False):
    wiki_dir = "~/raw_doc.json"
    storage_dir = "~/persist_dir"
    if os.path.exists(storage_dir):
        import shutil

        shutil.rmtree(storage_dir)
    import time

    start = time.time()
    q_retriever = QueryRetriever(
        persist_dir=storage_dir, raw_docs_dir=wiki_dir, use_async=use_async
    )
    end = time.time()
    print(
        f"""
        ------------------------------------------------------------------------------------------------------------------\n
        Use Async: {use_async}
        Time cost: {end-start:.3f} s
        ------------------------------------------------------------------------------------------------------------------\n
        """
    )
    print(q_retriever.retrieve("介绍一下耳鼻喉科学"))


if __name__ == "__main__":
    # compare_async(use_async=False)
    compare_async(use_async=True)
