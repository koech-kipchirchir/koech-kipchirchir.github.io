"""
In-memory knowledge graph with typed nodes, edges, and traversal.
"""

from __future__ import annotations

import json
import logging
import time
import uuid
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Optional

from knowledge.utils import structured_log

logger = logging.getLogger("aios.knowledge.graph")


class NodeNotFoundError(KeyError):
    pass


class EdgeNotFoundError(KeyError):
    pass


@dataclass
class GraphNode:
    """A node in the knowledge graph."""

    id: str = ""
    name: str = ""
    type: str = "entity"
    properties: dict[str, Any] = field(default_factory=dict)
    created_at: str = ""
    updated_at: str = ""
    version: int = 1

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "type": self.type,
            "properties": self.properties,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "version": self.version,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> GraphNode:
        return cls(**data)


@dataclass
class GraphEdge:
    """A directed edge between two nodes."""

    id: str = ""
    source_id: str = ""
    target_id: str = ""
    type: str = "related_to"
    weight: float = 1.0
    properties: dict[str, Any] = field(default_factory=dict)
    created_at: str = ""
    updated_at: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "source_id": self.source_id,
            "target_id": self.target_id,
            "type": self.type,
            "weight": self.weight,
            "properties": self.properties,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> GraphEdge:
        return cls(**data)


@dataclass
class QueryPath:
    """A path found during graph traversal."""

    node_ids: list[str] = field(default_factory=list)
    edge_ids: list[str] = field(default_factory=list)
    total_weight: float = 0.0
    length: int = 0


def _now() -> str:
    from datetime import datetime, timezone
    return datetime.now(timezone.utc).isoformat()


def _generate_id() -> str:
    return uuid.uuid4().hex[:12]


class KnowledgeGraph:
    """In-memory directed graph with adjacency and node/edge metadata.

    Supports add/update/delete operations, traversal, path finding,
    subgraph extraction, and JSON export/import.
    """

    def __init__(self) -> None:
        self._nodes: dict[str, GraphNode] = {}
        self._edges: dict[str, GraphEdge] = {}
        self._outgoing: dict[str, dict[str, list[str]]] = defaultdict(lambda: defaultdict(list))
        self._incoming: dict[str, dict[str, list[str]]] = defaultdict(lambda: defaultdict(list))
        self._lock: Any = None

    # ------------------------------------------------------------------
    # Node operations
    # ------------------------------------------------------------------

    def add_node(self, node: GraphNode) -> str:
        if not node.id:
            node.id = _generate_id()
        now = _now()
        node.created_at = node.created_at or now
        node.updated_at = now
        self._nodes[node.id] = node
        structured_log(logging.DEBUG, "graph.node.added",
                       node_id=node.id, name=node.name, type=node.type)
        return node.id

    def get_node(self, node_id: str) -> GraphNode:
        node = self._nodes.get(node_id)
        if node is None:
            raise NodeNotFoundError(f"Node not found: {node_id}")
        return node

    def update_node(self, node_id: str, **updates: Any) -> GraphNode:
        node = self.get_node(node_id)
        for key, val in updates.items():
            if hasattr(node, key):
                setattr(node, key, val)
        node.updated_at = _now()
        node.version += 1
        self._nodes[node_id] = node
        return node

    def delete_node(self, node_id: str) -> bool:
        if node_id not in self._nodes:
            return False
        # Remove all edges involving this node
        edge_ids = list(self._edges.keys())
        for eid in edge_ids:
            e = self._edges[eid]
            if e.source_id == node_id or e.target_id == node_id:
                self.delete_edge(eid)
        del self._nodes[node_id]
        self._outgoing.pop(node_id, None)
        self._incoming.pop(node_id, None)
        structured_log(logging.DEBUG, "graph.node.deleted", node_id=node_id)
        return True

    def has_node(self, node_id: str) -> bool:
        return node_id in self._nodes

    def node_count(self) -> int:
        return len(self._nodes)

    def get_nodes_by_type(self, node_type: str) -> list[GraphNode]:
        return [n for n in self._nodes.values() if n.type == node_type]

    def find_nodes(self, name: str | None = None, type_filter: str | None = None) -> list[GraphNode]:
        results = list(self._nodes.values())
        if name:
            nl = name.lower()
            results = [n for n in results if nl in n.name.lower()]
        if type_filter:
            results = [n for n in results if n.type == type_filter]
        return results

    def all_nodes(self) -> list[GraphNode]:
        return list(self._nodes.values())

    # ------------------------------------------------------------------
    # Edge operations
    # ------------------------------------------------------------------

    def add_edge(self, edge: GraphEdge) -> str:
        if not edge.id:
            edge.id = _generate_id()
        now = _now()
        edge.created_at = edge.created_at or now
        edge.updated_at = now

        if edge.source_id not in self._nodes:
            raise NodeNotFoundError(f"Source node not found: {edge.source_id}")
        if edge.target_id not in self._nodes:
            raise NodeNotFoundError(f"Target node not found: {edge.target_id}")

        self._edges[edge.id] = edge
        self._outgoing[edge.source_id][edge.type].append(edge.id)
        self._incoming[edge.target_id][edge.type].append(edge.id)
        structured_log(logging.DEBUG, "graph.edge.added",
                       edge_id=edge.id, source=edge.source_id,
                       target=edge.target_id, type=edge.type)
        return edge.id

    def get_edge(self, edge_id: str) -> GraphEdge:
        edge = self._edges.get(edge_id)
        if edge is None:
            raise EdgeNotFoundError(f"Edge not found: {edge_id}")
        return edge

    def update_edge(self, edge_id: str, **updates: Any) -> GraphEdge:
        edge = self.get_edge(edge_id)
        for key, val in updates.items():
            if hasattr(edge, key):
                setattr(edge, key, val)
        edge.updated_at = _now()
        self._edges[edge_id] = edge
        return edge

    def delete_edge(self, edge_id: str) -> bool:
        edge = self._edges.pop(edge_id, None)
        if edge is None:
            return False
        out_list = self._outgoing.get(edge.source_id, {}).get(edge.type, [])
        if edge_id in out_list:
            out_list.remove(edge_id)
        in_list = self._incoming.get(edge.target_id, {}).get(edge.type, [])
        if edge_id in in_list:
            in_list.remove(edge_id)
        return True

    def edge_count(self) -> int:
        return len(self._edges)

    # ------------------------------------------------------------------
    # Traversal
    # ------------------------------------------------------------------

    def get_neighbors(
        self,
        node_id: str,
        edge_type: str | None = None,
        direction: str = "outgoing",
    ) -> list[tuple[GraphEdge, GraphNode]]:
        result: list[tuple[GraphEdge, GraphNode]] = []
        if direction in ("outgoing", "both"):
            edge_map = self._outgoing.get(node_id, {})
            for etype, eids in edge_map.items():
                if edge_type and etype != edge_type:
                    continue
                for eid in eids:
                    edge = self._edges.get(eid)
                    if edge:
                        try:
                            node = self.get_node(edge.target_id)
                            result.append((edge, node))
                        except NodeNotFoundError:
                            pass
        if direction in ("incoming", "both"):
            edge_map = self._incoming.get(node_id, {})
            for etype, eids in edge_map.items():
                if edge_type and etype != edge_type:
                    continue
                for eid in eids:
                    edge = self._edges.get(eid)
                    if edge:
                        try:
                            node = self.get_node(edge.source_id)
                            result.append((edge, node))
                        except NodeNotFoundError:
                            pass
        return result

    def find_path(
        self,
        source_id: str,
        target_id: str,
        max_depth: int = 5,
        edge_type: str | None = None,
    ) -> list[QueryPath]:
        if source_id not in self._nodes or target_id not in self._nodes:
            return []
        paths: list[QueryPath] = []
        visited: set[str] = set()

        def dfs(current: str, target: str, depth: int,
                node_path: list[str], edge_path: list[str], weight: float) -> None:
            if depth > max_depth:
                return
            if current == target:
                paths.append(QueryPath(
                    node_ids=list(node_path),
                    edge_ids=list(edge_path),
                    total_weight=weight,
                    length=len(node_path) - 1,
                ))
                return
            if current in visited:
                return
            visited.add(current)
            for edge, neighbor in self.get_neighbors(current, edge_type=edge_type, direction="outgoing"):
                node_path.append(neighbor.id)
                edge_path.append(edge.id)
                dfs(neighbor.id, target, depth + 1, node_path, edge_path, weight + edge.weight)
                node_path.pop()
                edge_path.pop()
            visited.remove(current)

        dfs(source_id, target_id, 0, [source_id], [], 0.0)
        paths.sort(key=lambda p: p.total_weight, reverse=True)
        return paths

    def traverse(
        self,
        start_id: str,
        max_depth: int = 3,
        edge_type: str | None = None,
    ) -> list[GraphNode]:
        visited: set[str] = set()
        result: list[GraphNode] = []
        queue: list[tuple[str, int]] = [(start_id, 0)]

        while queue:
            nid, depth = queue.pop(0)
            if nid in visited or depth > max_depth:
                continue
            visited.add(nid)
            try:
                result.append(self.get_node(nid))
            except NodeNotFoundError:
                continue
            for edge, neighbor in self.get_neighbors(nid, edge_type=edge_type, direction="outgoing"):
                if neighbor.id not in visited:
                    queue.append((neighbor.id, depth + 1))

        return result

    # ------------------------------------------------------------------
    # Subgraph
    # ------------------------------------------------------------------

    def subgraph(self, node_ids: set[str], max_depth: int = 1) -> KnowledgeGraph:
        sg = KnowledgeGraph()
        included = set(node_ids)
        # Expand with neighbors up to max_depth
        current = set(node_ids)
        for _ in range(max_depth):
            next_set: set[str] = set()
            for nid in current:
                for edge, neighbor in self.get_neighbors(nid, direction="both"):
                    if neighbor.id not in included:
                        next_set.add(neighbor.id)
                        included.add(neighbor.id)
            current = next_set

        for nid in included:
            if nid in self._nodes:
                sg.add_node(self._nodes[nid])

        for edge in self._edges.values():
            if edge.source_id in included and edge.target_id in included:
                sg.add_edge(edge)

        return sg

    # ------------------------------------------------------------------
    # Stats
    # ------------------------------------------------------------------

    def stats(self) -> dict[str, Any]:
        edge_types: dict[str, int] = {}
        node_types: dict[str, int] = {}
        for e in self._edges.values():
            edge_types[e.type] = edge_types.get(e.type, 0) + 1
        for n in self._nodes.values():
            node_types[n.type] = node_types.get(n.type, 0) + 1
        return {
            "node_count": len(self._nodes),
            "edge_count": len(self._edges),
            "node_types": node_types,
            "edge_types": edge_types,
            "avg_degree": round(2 * len(self._edges) / max(len(self._nodes), 1), 2),
        }

    # ------------------------------------------------------------------
    # Serialization
    # ------------------------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        return {
            "nodes": {nid: n.to_dict() for nid, n in self._nodes.items()},
            "edges": {eid: e.to_dict() for eid, e in self._edges.items()},
        }

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, default=str)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> KnowledgeGraph:
        g = cls()
        for nid, ndata in data.get("nodes", {}).items():
            g.add_node(GraphNode.from_dict(ndata))
        for eid, edata in data.get("edges", {}).items():
            g.add_edge(GraphEdge.from_dict(edata))
        return g

    @classmethod
    def from_json(cls, text: str) -> KnowledgeGraph:
        return cls.from_dict(json.loads(text))

    def clear(self) -> None:
        self._nodes.clear()
        self._edges.clear()
        self._outgoing.clear()
        self._incoming.clear()
