"""Graphiti client lifecycle management.

Handles initialization, episode ingestion, and graph search.
"""

from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timezone

from graphiti_core import Graphiti
from graphiti_core.llm_client import RateLimitError
from graphiti_core.nodes import EpisodeType

from brain.config import get_settings
from brain.graph.bridge import NoOpCrossEncoder, build_graphiti_embedder, build_graphiti_llm_client
from brain.models import Document

log = logging.getLogger(__name__)


class GraphClient:
    """Manages the Graphiti instance lifecycle."""

    def __init__(self):
        self._graphiti: Graphiti | None = None

    @property
    def is_initialized(self) -> bool:
        return self._graphiti is not None

    async def initialize(self) -> None:
        """Create Graphiti instance with our config-derived clients."""
        cfg = get_settings()
        neo = cfg.neo4j

        llm_client = build_graphiti_llm_client()
        embedder = build_graphiti_embedder()

        self._graphiti = Graphiti(
            uri=neo.uri,
            user=neo.user,
            password=neo.password,
            llm_client=llm_client,
            embedder=embedder,
            cross_encoder=NoOpCrossEncoder(),
            max_coroutines=cfg.graph.max_coroutines,
        )

        await self._graphiti.build_indices_and_constraints()
        log.info("Graphiti initialized with Neo4j at %s", neo.uri)

    async def add_document(
        self,
        doc: Document,
        *,
        group_id: str | None = None,
    ) -> dict:
        """Ingest a document into the knowledge graph.

        If the document text fits within ``graph.episode_size``, it is sent
        as a single Graphiti episode.  Larger documents are automatically
        split into episodes of ``episode_size`` chars.

        Returns a summary dict with total nodes/edges created.
        """
        if not self._graphiti:
            raise RuntimeError("GraphClient not initialized. Call initialize() first.")

        cfg = get_settings()
        if group_id is None:
            group_id = cfg.graph.group_id
        episode_size = cfg.graph.episode_size

        ref_time = doc.ingested_at
        if ref_time.tzinfo is None:
            ref_time = ref_time.replace(tzinfo=timezone.utc)

        # Split into episodes if the document exceeds episode_size
        text = doc.text
        if len(text) <= episode_size:
            blocks = [text]
        else:
            blocks = [text[i : i + episode_size] for i in range(0, len(text), episode_size)]

        total_episodes = len(blocks)
        log.info(
            "Graph extraction for '%s': %d episode(s), %d chars total",
            doc.title,
            total_episodes,
            len(text),
        )

        total_nodes = 0
        total_edges = 0
        failed = 0

        for i, block in enumerate(blocks):
            if total_episodes > 1:
                name = f"{doc.title} [part {i + 1}/{total_episodes}]"
                desc = f"Document: {doc.source_path} (part {i + 1}/{total_episodes})"
                log.info(
                    "Graph extraction episode %d/%d for: %s (%d chars)",
                    i + 1,
                    total_episodes,
                    doc.title,
                    len(block),
                )
            else:
                name = doc.title
                desc = f"Document: {doc.source_path}"

            # Pause between episodes to respect API rate limits
            if i > 0 and total_episodes > 1:
                delay = cfg.graph.episode_delay
                log.info("Waiting %ds between episodes (rate limit cooldown)...", delay)
                await asyncio.sleep(delay)

            max_retries = cfg.graph.max_episode_retries
            retry_delay = cfg.graph.episode_retry_delay
            succeeded = False

            for attempt in range(1, max_retries + 1):
                try:
                    results = await self._graphiti.add_episode(
                        name=name,
                        episode_body=block,
                        source=EpisodeType.text,
                        source_description=desc,
                        reference_time=ref_time,
                        group_id=group_id,
                    )
                    total_nodes += len(results.nodes)
                    total_edges += len(results.edges)
                    log.info(
                        "Graph extraction episode %d/%d: %d nodes, %d edges",
                        i + 1,
                        total_episodes,
                        len(results.nodes),
                        len(results.edges),
                    )
                    succeeded = True
                    break
                except RateLimitError:
                    if attempt < max_retries:
                        log.warning(
                            "Rate limit hit for episode %d/%d (attempt %d/%d). "
                            "Waiting %ds before retry...",
                            i + 1,
                            total_episodes,
                            attempt,
                            max_retries,
                            retry_delay,
                        )
                        await asyncio.sleep(retry_delay)
                    else:
                        log.error(
                            "Rate limit hit for episode %d/%d — all %d attempts exhausted (non-fatal)",
                            i + 1,
                            total_episodes,
                            max_retries,
                        )
                except Exception:
                    log.exception(
                        "Graph extraction failed for episode %d/%d (non-fatal)", i + 1, total_episodes
                    )
                    break  # non-rate-limit errors don't retry

            if not succeeded:
                failed += 1

        summary = {
            "total_nodes": total_nodes,
            "total_edges": total_edges,
            "episodes_processed": total_episodes - failed,
            "episodes_failed": failed,
        }
        log.info(
            "Graph extraction complete: %d nodes, %d edges from %d/%d episodes for '%s'",
            total_nodes,
            total_edges,
            total_episodes - failed,
            total_episodes,
            doc.title,
        )
        return summary

    async def search(
        self,
        query: str,
        *,
        group_id: str | None = None,
        num_results: int | None = None,
    ) -> list[dict]:
        """Search the knowledge graph. Returns a list of fact dicts.

        Uses Graphiti's basic search() which returns EntityEdge objects
        (no cross-encoder required).
        """
        if not self._graphiti:
            raise RuntimeError("GraphClient not initialized. Call initialize() first.")

        cfg = get_settings()
        if group_id is None:
            group_id = cfg.graph.group_id
        if num_results is None:
            num_results = cfg.graph.num_results

        edges = await self._graphiti.search(
            query=query,
            group_ids=[group_id],
            num_results=num_results,
        )

        facts = []
        for edge in edges:
            facts.append({
                "fact": edge.fact,
                "source_node": edge.source_node_name,
                "target_node": edge.target_node_name,
                "created_at": str(edge.created_at) if edge.created_at else None,
                "uuid": edge.uuid,
            })
        return facts

    async def get_stats(self) -> dict:
        """Return node and relationship counts from Neo4j."""
        cfg = get_settings()
        neo = cfg.neo4j

        from neo4j import GraphDatabase

        driver = GraphDatabase.driver(neo.uri, auth=(neo.user, neo.password))
        with driver.session() as session:
            nodes = session.run("MATCH (n) RETURN count(n) AS c").single()["c"]
            rels = session.run("MATCH ()-[r]->() RETURN count(r) AS c").single()["c"]
        driver.close()
        return {"nodes": nodes, "relationships": rels}

    async def clear_graph(self) -> dict:
        """Delete ALL nodes and relationships from Neo4j.

        Returns the counts of deleted nodes and relationships.
        """
        cfg = get_settings()
        neo = cfg.neo4j

        from neo4j import GraphDatabase

        driver = GraphDatabase.driver(neo.uri, auth=(neo.user, neo.password))
        with driver.session() as session:
            # Get counts before deletion for the summary
            nodes = session.run("MATCH (n) RETURN count(n) AS c").single()["c"]
            rels = session.run("MATCH ()-[r]->() RETURN count(r) AS c").single()["c"]
            # Delete in batches to avoid memory issues on large graphs
            session.run("MATCH (n) DETACH DELETE n")
        driver.close()

        log.info("Cleared graph: deleted %d nodes, %d relationships", nodes, rels)
        return {"deleted_nodes": nodes, "deleted_relationships": rels}

    async def close(self) -> None:
        """Shut down the Graphiti client."""
        if self._graphiti:
            await self._graphiti.close()
            self._graphiti = None
            log.info("Graphiti client closed.")
