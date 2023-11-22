import asyncio
import os
from copy import deepcopy
from functools import reduce
from typing import Any, Dict, List, Optional, Sequence, cast

from asyncextractor.customllm import CusomHttpLLM
from llama_index.bridge.pydantic import Field
from llama_index.llm_predictor.base import BaseLLMPredictor, LLMPredictor
from llama_index.llms.base import LLM
from llama_index.node_parser.extractors import (
    MetadataFeatureExtractor,
    QuestionsAnsweredExtractor,
    SummaryExtractor,
    TitleExtractor,
)
from llama_index.node_parser.extractors.metadata_extractors import MetadataExtractor
from llama_index.prompts import PromptTemplate
from llama_index.schema import BaseNode, Document, TextNode
from llama_index.utils import get_tqdm_iterable

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


def run_async_tasks(tasks):
    async def _gather():
        return await asyncio.gather(*tasks)

    return asyncio.run(_gather())


class AsyncKeywordExtractor(MetadataFeatureExtractor):
    """Keyword extractor. Node-level extractor. Extracts
    `excerpt_keywords` metadata field.

    Args:
        llm_predictor (Optional[BaseLLMPredictor]): LLM predictor
        keywords (int): number of keywords to extract
    """

    llm_predictor: BaseLLMPredictor = Field(
        description="The LLMPredictor to use for generation."
    )
    keywords: int = Field(default=5, description="The number of keywords to extract.")
    prompt_template: str = Field(
        default=KEYWORD_GEN_TEMPL,
        description="Prompt template to use when generating keywords.",
    )

    def __init__(
        self,
        llm: Optional[LLM] = None,
        llm_predictor: Optional[BaseLLMPredictor] = None,
        keywords: int = 5,
        prompt_template: str = KEYWORD_GEN_TEMPL,
        **kwargs: Any,
    ) -> None:
        """Init params."""
        if keywords < 1:
            raise ValueError("num_keywords must be >= 1")

        if llm is not None:
            llm_predictor = LLMPredictor(llm=llm)
        elif llm_predictor is None and llm is None:
            llm_predictor = LLMPredictor()

        super().__init__(
            llm_predictor=llm_predictor,
            keywords=keywords,
            prompt_template=prompt_template,
            **kwargs,
        )

    @classmethod
    def class_name(cls) -> str:
        return "MyKeywordExtractor"

    def extract(self, nodes: Sequence[BaseNode]) -> List[Dict]:
        metadata_list: List[Dict] = []
        for node in nodes:
            if self.is_text_node_only and not isinstance(node, TextNode):
                metadata_list.append({})
                continue

            prompt = PromptTemplate(template=self.prompt_template)
            keywords = self.llm_predictor.predict(
                prompt=prompt,
                context_str=cast(TextNode, node).text,
                keywords=self.keywords,
            )
            # node.metadata["excerpt_keywords"] = keywords
            metadata_list.append({"excerpt_keywords": keywords.strip()})
        return metadata_list

    async def aextract(self, nodes: Sequence[BaseNode]) -> List[Dict]:
        metadata_list: List[Dict] = []

        tasks = []
        for node in nodes:
            if self.is_text_node_only and not isinstance(node, TextNode):
                metadata_list.append({})
                continue

            prompt = PromptTemplate(template=self.prompt_template)
            tasks.append(
                self.llm_predictor.apredict(
                    prompt=prompt,
                    context_str=cast(TextNode, node).text,
                    keywords=self.keywords,
                )
            )
            # node.metadata["excerpt_keywords"] = keywords
        results = await asyncio.gather(*tasks)
        for keywords in results:
            metadata_list.append({"excerpt_keywords": keywords.strip()})
        return metadata_list


class AsyncTitleExtractor(TitleExtractor):
    @classmethod
    def class_name(cls) -> str:
        return "AsyncTitleExtractor"

    async def aextract(self, nodes: Sequence[BaseNode]) -> List[Dict]:
        nodes_to_extract_title: List[BaseNode] = []
        for node in nodes:
            if len(nodes_to_extract_title) >= self.nodes:
                break
            if self.is_text_node_only and not isinstance(node, TextNode):
                continue
            nodes_to_extract_title.append(node)

        if len(nodes_to_extract_title) == 0:
            # Could not extract title
            return []
        tasks = [
            self.llm_predictor.apredict(
                PromptTemplate(template=self.node_template),
                context_str=cast(TextNode, node).text,
            )
            for node in nodes_to_extract_title
        ]
        title_candidates = await asyncio.gather(*tasks)

        if len(nodes_to_extract_title) > 1:
            titles = reduce(
                lambda x, y: x + "," + y, title_candidates[1:], title_candidates[0]
            )

            title = await self.llm_predictor.apredict(
                PromptTemplate(template=self.combine_template),
                context_str=titles,
            )
        else:
            title = title_candidates[
                0
            ]  # if single node, just use the title from that node

        return [{"document_title": title.strip(' \t\n\r"')} for _ in nodes]


class AsyncQuestionsAnsweredExtractor(QuestionsAnsweredExtractor):
    async def aextract(self, nodes: Sequence[BaseNode]) -> List[Dict]:
        metadata_list: List[Dict] = []
        nodes_queue = get_tqdm_iterable(
            nodes, self.show_progress, "Extracting questions"
        )
        tasks = []
        for node in nodes_queue:
            if self.is_text_node_only and not isinstance(node, TextNode):
                metadata_list.append({})
                continue

            context_str = node.get_content(metadata_mode=self.metadata_mode)
            prompt = PromptTemplate(template=self.prompt_template)
            tasks.append(
                self.llm_predictor.apredict(
                    prompt, num_questions=self.questions, context_str=context_str
                )
            )
            if self.embedding_only:
                node.excluded_llm_metadata_keys = ["questions_this_excerpt_can_answer"]
        questions = await asyncio.gather(*tasks)
        for question in questions:
            metadata_list.append(
                {"questions_this_excerpt_can_answer": question.strip()}
            )
        return metadata_list


class AsyncSummaryExtractor(SummaryExtractor):
    @classmethod
    def class_name(cls) -> str:
        return "MySummaryExtractor"

    async def aextract(self, nodes: Sequence[BaseNode]) -> List[Dict]:
        if not all(isinstance(node, TextNode) for node in nodes):
            raise ValueError("Only `TextNode` is allowed for `Summary` extractor")
        nodes_queue = get_tqdm_iterable(
            nodes, self.show_progress, "Extracting summaries"
        )
        tasks = []
        for node in nodes_queue:
            node_context = cast(TextNode, node).get_content(
                metadata_mode=self.metadata_mode
            )
            tasks.append(
                self.llm_predictor.apredict(
                    PromptTemplate(template=self.prompt_template),
                    context_str=node_context,
                )
            )

        node_summaries = await asyncio.gather(*tasks)

        # Extract node-level summary metadata
        metadata_list: List[Dict] = [{} for _ in nodes]
        for i, metadata in enumerate(metadata_list):
            if i > 0 and self._prev_summary:
                metadata["prev_section_summary"] = node_summaries[i - 1].strip()
            if i < len(nodes) - 1 and self._next_summary:
                metadata["next_section_summary"] = node_summaries[i + 1].strip()
            if self._self_summary:
                metadata["section_summary"] = node_summaries[i].strip()

        return metadata_list


class AsyncMetadataExtractor(MetadataExtractor):
    async def aprocess_nodes(
        self,
        nodes: List[BaseNode],
        excluded_embed_metadata_keys: Optional[List[str]] = None,
        excluded_llm_metadata_keys: Optional[List[str]] = None,
    ) -> List[BaseNode]:
        """Post process nodes parsed from documents.

        Allows extractors to be chained.

        Args:
            nodes (List[BaseNode]): nodes to post-process
            excluded_embed_metadata_keys (Optional[List[str]]):
                keys to exclude from embed metadata
            excluded_llm_metadata_keys (Optional[List[str]]):
                keys to exclude from llm metadata
        """
        if self.in_place:
            new_nodes = nodes
        else:
            new_nodes = [deepcopy(node) for node in nodes]
        tasks = [extractor.aextract(new_nodes) for extractor in self.extractors]
        metadata_lists = await asyncio.gather(*tasks)
        for cur_metadata_list in metadata_lists:
            for idx, node in enumerate(new_nodes):
                node.metadata.update(cur_metadata_list[idx])

        for idx, node in enumerate(new_nodes):
            if excluded_embed_metadata_keys is not None:
                node.excluded_embed_metadata_keys.extend(excluded_embed_metadata_keys)
            if excluded_llm_metadata_keys is not None:
                node.excluded_llm_metadata_keys.extend(excluded_llm_metadata_keys)
            if not self.disable_template_rewrite:
                if isinstance(node, TextNode):
                    cast(TextNode, node).text_template = self.node_text_template
        return new_nodes


if __name__ == "__main__":
    """
    Some test code for verifying the correctness of the async extractor
    """
    from asyncextractor.parser import AsyncSimpleNodeParser

    from llama_index.node_parser.extractors import MetadataExtractor

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

    import json

    with open("~/raw_doc.json", "r") as f:
        raw_docs = json.loads(f.read())
        docs = [
            Document(
                text=doc["reference"],
                metadata={"title": doc["title"], "summarization": doc["evidence"]},
            )
            for doc in raw_docs
        ]
    extractor = AsyncMetadataExtractor(
        extractors=[
            AsyncTitleExtractor(
                nodes=5,
                llm=llm,
                node_template=TITLE_NODE_TEMPLATE,
                combine_template=TITLE_COMBINE_TEMPLATE,
            ),
            AsyncQuestionsAnsweredExtractor(
                questions=3, llm=llm, prompt_template=QUESTION_GEN_TMPL
            ),
            # EntityExtractor(prediction_threshold=0.5, device="cuda"),
            AsyncSummaryExtractor(
                summaries=["prev", "self"],
                llm=llm,
                prompt_template=SUMMARY_EXTRACT_TEMPLATE,
            ),
            AsyncKeywordExtractor(
                keywords=10, llm=llm, prompt_template=KEYWORD_GEN_TEMPL
            ),
        ]
    )
    parser = AsyncSimpleNodeParser.from_defaults(
        chunk_size=1024, metadata_extractor=extractor, use_async=True
    )
    nodes = parser.get_nodes_from_documents(docs[:20])
    print(len(nodes))
    print(nodes[-1].metadata)
