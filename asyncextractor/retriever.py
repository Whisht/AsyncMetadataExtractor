import json
import os
from datetime import datetime
from typing import Dict, List, Optional, Set, Union

import faiss
import jieba.analyse
from loguru import logger

from asyncextractor.customllm import CusomHttpLLM
from asyncextractor.metadataextractor import (
    AsyncKeywordExtractor,
    AsyncMetadataExtractor,
    AsyncQuestionsAnsweredExtractor,
    AsyncSummaryExtractor,
    AsyncTitleExtractor,
)
from asyncextractor.parser import AsyncHierarchicalNodeParser
from llama_index import (
    ServiceContext,
    SimpleKeywordTableIndex,
    StorageContext,
    TreeIndex,
    VectorStoreIndex,
    load_indices_from_storage,
)
from llama_index.embeddings import HuggingFaceEmbedding
from llama_index.indices.base import BaseIndex
from llama_index.indices.keyword_table.base import (
    KeywordTableIndex,
)
from llama_index.indices.keyword_table.retrievers import BaseKeywordTableRetriever
from llama_index.indices.tree.base import TreeRetrieverMode
from llama_index.retrievers import BM25Retriever, QueryFusionRetriever
from llama_index.retrievers.fusion_retriever import FUSION_MODES
from llama_index.schema import BaseNode, Document
from llama_index.vector_stores.faiss import FaissVectorStore

TITLE_NODE_TEMPLATE = """\
Context: {context_str}。请给出一个标题，概括 Context 中的所有独特实体、标题或主题. Title: """


TITLE_COMBINE_TEMPLATE = """\
{context_str}。基于上述候选标题和内容, 最能概括本文全部内容的标题是什么? Title: """

SUMMARY_EXTRACT_TEMPLATE = """\
以下是该部分的内容：
{context_str}

请给出该部分的摘要，尽可能包含关键主题和实体。\
Summary: """

QUESTION_GEN_TMPL = """\
以下是给定的内容:
{context_str}

请根据所给定的内容，生成该片段可以回答 {num_questions} 个问题。\
请保证这些问题的具体答案不太可能在其他地方找到。

此外，你也可以对该内容进行总结摘要，并使用这些摘要来生成该内容可以回答的更好问题。
"""
KEYWORD_GEN_TEMPL = """\
以下是给定的内容:
{context_str}. 请从该片段中提取 {keywords} 个不同的关键词，并用半角逗号分隔。Keywords: """


class MySimpleKeywordTableIndex(SimpleKeywordTableIndex):
    def _extract_keywords(self, text: str) -> Set[str]:
        return set(jieba.analyse.textrank(text))


class MyKeywordTableSimpleRetriever(BaseKeywordTableRetriever):
    def _get_keywords(self, query_str: str) -> List[str]:
        keywords = jieba.analyse.textrank(query_str)
        return keywords[: self.max_keywords_per_query]


def build_nodes(
    docs: List[Document], service_context: ServiceContext, use_async: bool = False
) -> Union[List[BaseNode], List]:
    logger.info(
        "Starting Build nodes from documents, it will cost time depending on the MetadataExtractor..."
    )
    metadata_extractor = AsyncMetadataExtractor(
        extractors=[
            AsyncTitleExtractor(
                nodes=5,
                llm=service_context.llm,
                node_template=TITLE_NODE_TEMPLATE,
                combine_template=TITLE_COMBINE_TEMPLATE,
            ),
            AsyncQuestionsAnsweredExtractor(
                questions=3, llm=service_context.llm, prompt_template=QUESTION_GEN_TMPL
            ),
            # EntityExtractor(prediction_threshold=0.5, device="cuda"),
            AsyncSummaryExtractor(
                summaries=["prev", "self"],
                llm=service_context.llm,
                prompt_template=SUMMARY_EXTRACT_TEMPLATE,
            ),
            AsyncKeywordExtractor(
                keywords=10, llm=service_context.llm, prompt_template=KEYWORD_GEN_TEMPL
            )
            # CustomExtractor()
        ],
    )
    # metadata_extractor = None
    parser = AsyncHierarchicalNodeParser.from_defaults(
        chunk_sizes=[2048, 1024, 512],
        metadata_extractor=metadata_extractor,
        use_async=use_async,
    )
    try:
        logger.info("Starting parse nodes and extract metadata from documents...")
        st = datetime.now()
        nodes = parser.get_nodes_from_documents(docs)
        end = datetime.now()
        logger.info(f"Finished build nodes from documents. Time cost: {end-st}")
        return nodes
    except Exception as e:
        logger.debug(f"Error: {e}")
        nodes = []
        return nodes


def build_indices_from_nodes(
    nodes: List[BaseNode],
    service_context: ServiceContext,
    storage_context: StorageContext,
) -> Dict[str, BaseIndex]:
    # build vector store index
    logger.info("Starting build indices from parsed nodes...")
    st = datetime.now()
    vector_index = VectorStoreIndex(
        nodes=nodes,
        service_context=service_context,
        storage_context=storage_context,
        show_progress=True,
    )

    # build keyword table index
    keyword_table_index = MySimpleKeywordTableIndex(
        nodes=nodes, service_context=service_context, storage_context=storage_context
    )
    # build tree index
    from llama_index import PromptTemplate
    from llama_index.prompts import PromptType

    summary_prompt = (
        "请只使用以下文本所提供的信息为该文本撰写摘要. "
        "请在摘要中包含尽可能多的细节\n"
        "\n"
        "\n"
        "{context_str}\n"
        "\n"
        "\n"
        'SUMMARY:"""\n'
    )
    prompt = PromptTemplate(summary_prompt, prompt_type=PromptType.SUMMARY)
    tree_index = TreeIndex(
        nodes=nodes,
        service_context=service_context,
        storage_context=storage_context,
        summary_template=prompt,
    )
    logger.info(f"Finished build Indices. Time cost: {datetime.now()-st}")
    return {
        "VectorStoreIndex": vector_index,
        "KeywordTableIndex": keyword_table_index,
        "TreeIndex": tree_index,
    }


def load_docs_from_dir(docs_dir: str) -> List[Dict]:
    with open(docs_dir, "r") as f:
        raw_docs = json.loads(f.read())
        docs = [
            {
                "text": doc["reference"],
                "metadata": {
                    "title": doc["title"],
                    "summarization": doc["evidence"],
                },
            }
            for doc in raw_docs
        ]
        # docs = [
        # Document(
        #     text=doc["reference"],
        #     metadata={"title": doc["title"], "summarization": doc["evidence"]},
        # )
        # for doc in raw_docs
        # ]
        return docs


class MyBaseRetriever:
    def __init__(
        self,
        service_context: Optional[ServiceContext],
        retriever: Optional[QueryFusionRetriever],
        indices: Optional[Dict[str, BaseIndex]],
        llm: Optional[CusomHttpLLM],
        storage_context: Optional[StorageContext],
        embed_model: Optional[HuggingFaceEmbedding],
    ):
        self.service_context = service_context
        self.retriever = retriever
        self.indices = indices
        self.llm = llm
        self.storage_context = storage_context
        self.embed_model = embed_model

    def retrieve(self, query):
        raise NotImplementedError


class QueryRetriever(MyBaseRetriever):
    def __init__(self, persist_dir: str, raw_docs_dir: str, use_async: bool = False):
        (
            service_context,
            storage_context,
            llm,
            retriever,
            embed_model,
            indices,
        ) = self._config_env(
            persist_dir=persist_dir, raw_docs_dir=raw_docs_dir, use_async=use_async
        )
        super().__init__(
            service_context=service_context,
            retriever=retriever,
            indices=indices,
            llm=llm,
            storage_context=storage_context,
            embed_model=embed_model,
        )

    def retrieve(self, query):
        results = self.retriever.retrieve(query)
        return "\n".join([r.text for r in results])

    def _config_env(
        self,
        persist_dir: str,
        model_name="infgrad/stella-large-zh-v2",
        max_length: int = 1024,
        device="cuda",
        cache_folder: str = "~/cache",
        raw_docs_dir: str = None,
        use_async: bool = False,
    ):
        """

        Args:
            persist_dir: the path of storage dir

        Returns:
            service_context,storage_context,llm

        """
        # config embedding model
        embed_model = HuggingFaceEmbedding(
            model_name=model_name,
            max_length=max_length,
            device=device,
            cache_folder=cache_folder,
        )
        # config llm
        app_key = os.environ.get("AppKey")
        app_secret = os.environ.get("AppSecret")
        llm = CusomHttpLLM(
            appKey=app_key,
            appSecret=app_secret,
            model="gpt4",
            environment="prod",
            temperature=1,
            topP=1,
        )
        # config service_context
        service_context = ServiceContext.from_defaults(embed_model=embed_model, llm=llm)
        if os.path.exists(persist_dir):
            # config storage_context from persist_dir
            vector_store = FaissVectorStore.from_persist_dir(persist_dir)
            storage_context = StorageContext.from_defaults(
                vector_store=vector_store, persist_dir=persist_dir
            )
            # load indices from storage_context
            logger.info("Persist Dir exist. Will Load existed indices from it...")
            indices = load_indices_from_storage(
                storage_context=storage_context, service_context=service_context
            )
            indices = {index.__class__.__name__: index for index in indices}
            logger.info("Finished loading indices from local path.")
        elif raw_docs_dir is not None:
            # create vector_store if not exist
            logger.info(
                "Persist Dir not exist. Will create new indices for raw docs path..."
            )
            faiss_index = faiss.IndexFlatL2(max_length)
            vector_store = FaissVectorStore(faiss_index=faiss_index)
            storage_context = StorageContext.from_defaults(
                vector_store=vector_store,
            )
            indices = self._initialize_indices(
                raw_docs=load_docs_from_dir(raw_docs_dir),
                service_context=service_context,
                storage_context=storage_context,
                use_async=use_async,
            )
            # save indices to persist_dir
            storage_context.persist(persist_dir=persist_dir)
        else:
            raise ValueError(
                "There should be at least one of persist_dir and raw_doc_dir!"
            )
        # return service_context, storage_context, llm

        storage_context = storage_context
        retriever = self._build_retriever(
            indices=indices,
            llm=llm,
        )
        return service_context, storage_context, llm, retriever, embed_model, indices

    @staticmethod
    def _initialize_indices(
        raw_docs: List[Dict[str, Union[str, Dict]]],
        service_context,
        storage_context,
        use_async: bool = False,
    ):
        docs = [
            Document(
                text=doc["text"], metadata={k: v for k, v in doc["metadata"].items()}
            )
            for doc in raw_docs
        ]
        nodes = build_nodes(
            docs=docs, service_context=service_context, use_async=use_async
        )
        return build_indices_from_nodes(
            nodes=nodes,
            service_context=service_context,
            storage_context=storage_context,
        )

    @staticmethod
    def _build_retriever(
        indices: Dict[
            str,
            Optional[Union[BaseIndex, VectorStoreIndex, TreeIndex, KeywordTableIndex]],
        ],
        llm: CusomHttpLLM,
    ) -> QueryFusionRetriever:
        # build corresponding retriever for each index
        retrievers = {}
        for name, index in indices.items():
            if isinstance(index, KeywordTableIndex):
                retrievers[f"{name[:-5]}Retriever"] = MyKeywordTableSimpleRetriever(
                    index
                )
            elif isinstance(index, VectorStoreIndex):
                retrievers[f"{name[:-5]}Retriever"] = index.as_retriever()
            elif isinstance(index, TreeIndex):
                retrievers[f"{name[:-5]}Retriever"] = index.as_retriever(
                    retriever_mode=TreeRetrieverMode.SELECT_LEAF_EMBEDDING
                )
        retrievers["BM25Retriever"] = BM25Retriever.from_defaults(
            index=indices["VectorStoreIndex"],
            tokenizer=lambda x: jieba.lcut_for_search(x),
        )

        query_gen_prompt = (
            "你是一个强大的AI助手，可以根据单个输入查询生成多个搜索查询。"
            "请根据以下查询 Query 生成与其相关的 {num_queries} 个搜索查询 Queries，每行一个，\n"
            "Query: {query}\n"
            "Queries:\n"
        )
        return QueryFusionRetriever(
            retrievers=list(retrievers.values()),
            similarity_top_k=3,
            num_queries=4,  # set this to 1 to disable query generation
            mode=FUSION_MODES.RECIPROCAL_RANK,
            use_async=True,
            llm=llm,
            query_gen_prompt=query_gen_prompt,
            # verbose=True,
            # query_gen_prompt="...",  # we could override the query generation prompt here
        )
