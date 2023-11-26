import functools
import typing

import aiohttp
from langchain.docstore.document import Document
from langchain.utilities import SerpAPIWrapper

from src.utils_langchain import _chunk_sources, add_parser, _add_meta
from urllib.parse import urlparse


class H2OSerpAPIWrapper(SerpAPIWrapper):
    def get_search_documents(self, query,
                             query_action=True,
                             chunk=True, chunk_size=512,
                             db_type='chroma',
                             headsize=50,
                             top_k_docs=-1):
        docs = self.run(query, headsize)

        chunk_sources = functools.partial(_chunk_sources, chunk=chunk, chunk_size=chunk_size, db_type=db_type)
        docs = chunk_sources(docs)

        # choose chunk type
        if query_action:
            docs = [x for x in docs if x.metadata['chunk_id'] >= 0]
        else:
            docs = [x for x in docs if x.metadata['chunk_id'] == -1]

        # get score assuming search results scale with ranking
        delta = 0.05
        [x.metadata.update(score=0.1 + delta * x.metadata['chunk_id'] if x.metadata['chunk_id'] >= 0 else -1) for x in
         docs]

        # ensure see all results up to cutoff or mixing with non-web docs
        if top_k_docs >= 1:
            top_k_docs = max(top_k_docs, len(docs))

        return docs, top_k_docs

    async def arun(self, query: str, headsize: int, **kwargs: typing.Any) -> list:
        """Run query through SerpAPI and parse result async."""
        return self._process_response(await self.aresults(query), query, headsize)

    def run(self, query: str, headsize: int, **kwargs: typing.Any) -> list:
        """Run query through SerpAPI and parse result."""
        return self._process_response(self.results(query), query, headsize)

    @staticmethod
    def _process_response(res: dict, query: str, headsize: int) -> list:
        try:
            return H2OSerpAPIWrapper.__process_response(res, query, headsize)
        except Exception as e:
            print(f"SERP search failed: {str(e)}")
            return []

    @staticmethod
    def __process_response(res: dict, query: str, headsize: int) -> list:
        docs = []

        if res1 := SerpAPIWrapper._process_response(res):
            if isinstance(res1, str) and not res1.startswith('['):  # avoid snippets
                docs += [
                    Document(
                        page_content=f'Web search result {len(docs)}: {res1}',
                        metadata=dict(
                            source=f'Web Search {len(docs)} for {query}', score=0.0
                        ),
                    )
                ]
            elif isinstance(res1, list):
                for x in res1:
                    date = ''
                    content = ''
                    if 'source' in x:
                        source = x['source']
                        content += f'{source} says'
                    else:
                        content = f'Web search result {len(docs)}: '
                    if 'date' in x:
                        date = x['date']
                        content += f' {date}'
                    if 'title' in x:
                        content += f": {x['title']}"
                    if 'snippet' in x:
                        content += f": {x['snippet']}"
                    if 'link' in x:
                        link = x['link']
                        domain = urlparse(link).netloc
                        font_size = 2
                        source_name = domain
                        http_content = f"""<font size="{font_size}"><a href="{link}" target="_blank"  rel="noopener noreferrer">{source_name}</a></font>"""
                        source = (
                            f'Web Search {len(docs)}'
                            + f' from Date: {date} Domain: {domain} Link: {http_content}'
                        )
                        if date:
                            content += f' around {date}'
                        content += f' according to {domain}'
                    else:
                        source = f'Web Search {len(docs)} for {query}'
                    docs += [Document(page_content=content, metadata=dict(source=source, score=0.0))]

        if "knowledge_graph" in res:
            knowledge_graph = res["knowledge_graph"]
            title = knowledge_graph["title"] if "title" in knowledge_graph else ""
            if "description" in knowledge_graph.keys():
                docs += [
                    Document(
                        page_content=f'Web search result {len(docs)}: '
                        + knowledge_graph["description"],
                        metadata=dict(
                            source=f'Web Search {len(docs)} with knowledge_graph description for {query}',
                            score=0.0,
                        ),
                    )
                ]
            for key, value in knowledge_graph.items():
                if (
                        type(key) == str
                        and type(value) == str
                        and key not in ["title", "description"]
                        and not key.endswith("_stick")
                        and not key.endswith("_link")
                        and not value.startswith("http")
                ):
                    docs += [
                        Document(
                            page_content=f'Web search result {len(docs)}: '
                            + f"{title} {key}: {value}.",
                            metadata=dict(
                                source=f'Web Search {len(docs)} with knowledge_graph for {query}',
                                score=0.0,
                            ),
                        )
                    ]
        if "organic_results" in res:
            for org_res in res["organic_results"]:
                keys_to_try = ['snippet', 'snippet_highlighted_words', 'rich_snippet', 'rich_snippet_table', 'link']
                for key in keys_to_try:
                    if key in org_res.keys():
                        date = ''
                        domain = ''
                        link = ''
                        snippet1 = ''
                        if key != 'link':
                            snippet1 = org_res[key]
                        if 'date' in org_res.keys():
                            date = org_res['date']
                            snippet1 += f' on {date}'
                        else:
                            date = 'unknown date'
                        if 'link' in org_res.keys():
                            link = org_res['link']
                            domain = urlparse(link).netloc
                            if key == 'link':
                                # worst case, only url might have REST info
                                snippet1 += f' Link at {domain}: <a href="{link}">{domain}</a>'
                            else:
                                snippet1 += f' according to {domain}'
                        if snippet1:
                            font_size = 2
                            source_name = domain
                            http_content = f"""<font size="{font_size}"><a href="{link}" target="_blank"  rel="noopener noreferrer">{source_name}</a></font>"""
                            source = (
                                f'Web Search {len(docs)}'
                                + f' from Date: {date} Domain: {domain} Link: {http_content}'
                            )
                            domain_simple = domain.replace('www.', '').replace('.com', '')
                            snippet1 = f'{domain_simple} says on {date}: {snippet1}'
                            docs += [Document(page_content=snippet1, metadata=dict(source=source), score=0.0)]
                            break
        if "buying_guide" in res:
            docs += [
                Document(
                    page_content=f'Web search result {len(docs)}: '
                    + res["buying_guide"],
                    metadata=dict(
                        source=f'Web Search {len(docs)} with buying_guide for {query}'
                    ),
                    score=0.0,
                )
            ]
        if "local_results" in res and "places" in res["local_results"].keys():
            docs += [
                Document(
                    page_content=f'Web search result {len(docs)}: '
                    + res["local_results"]["places"],
                    metadata=dict(
                        source=f'Web Search {len(docs)} with local_results_places for {query}'
                    ),
                    score=0.0,
                )
            ]

        # add meta
        add_meta = functools.partial(_add_meta, headsize=headsize, parser='SERPAPI')
        add_meta(docs, query)

        return docs

    def results(self, query: str) -> dict:
        # Fix non-thread-safe langchain swapping out sys directly.
        """Run query through SerpAPI and return the raw result."""
        params = self.get_params(query)
        search = self.search_engine(params)
        return search.get_dict()
