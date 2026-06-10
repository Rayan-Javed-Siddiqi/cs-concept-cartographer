import streamlit as st
from google import genai
from google.genai import types
import json
import os
import sqlite3
import time
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()

st.set_page_config(
    page_title="CS Concept Cartographer",
    page_icon="🗺️",
    layout="wide",
)

# ── Constants ─────────────────────────────────────────────────────────────────

MODEL_NAME        = "gemini-3-flash-preview"
DB_PATH           = os.path.join(os.path.dirname(__file__), "maps.db")
HISTORY_PAGE_SIZE = 10
DEFAULT_NODES     = {"Beginner": 8, "Intermediate": 12, "Advanced": 16}

# ── Helpers ───────────────────────────────────────────────────────────────────

def normalize_topic(topic: str) -> str:
    return topic.strip()


def topic_cache_key(topic: str) -> str:
    return normalize_topic(topic).casefold()


def profile_key(name: str) -> str:
    return name.strip() or "default"

# ── Database ──────────────────────────────────────────────────────────────────

def init_db():
    with sqlite3.connect(DB_PATH) as conn:
        conn.executescript("""
            CREATE TABLE IF NOT EXISTS maps (
                id        INTEGER PRIMARY KEY AUTOINCREMENT,
                profile   TEXT    NOT NULL DEFAULT 'default',
                topic     TEXT    NOT NULL,
                level     TEXT    NOT NULL,
                json_data TEXT    NOT NULL,
                timestamp TEXT    NOT NULL,
                name      TEXT
            );
            CREATE TABLE IF NOT EXISTS annotations (
                id      INTEGER PRIMARY KEY AUTOINCREMENT,
                map_id  INTEGER NOT NULL,
                node_id TEXT    NOT NULL,
                note    TEXT    NOT NULL,
                FOREIGN KEY (map_id) REFERENCES maps(id) ON DELETE CASCADE
            );
        """)


def save_map(profile: str, topic: str, level: str, graph_data: dict) -> int:
    with sqlite3.connect(DB_PATH) as conn:
        cur = conn.execute(
            "INSERT INTO maps (profile, topic, level, json_data, timestamp) VALUES (?,?,?,?,?)",
            (profile, topic, level, json.dumps(graph_data), datetime.now().strftime("%Y-%m-%d %H:%M")),
        )
        return cur.lastrowid


def get_cached_map(profile: str, topic: str, level: str, node_count: int | None = None):
    key = topic_cache_key(topic)
    with sqlite3.connect(DB_PATH) as conn:
        rows = conn.execute(
            "SELECT id, json_data, topic FROM maps WHERE profile=? AND level=? ORDER BY id DESC",
            (profile, level),
        ).fetchall()
    for map_id, json_data, stored_topic in rows:
        if topic_cache_key(stored_topic) != key:
            continue
        data = json.loads(json_data)
        if node_count is not None and len(data.get("nodes", [])) != node_count:
            continue
        return map_id, data
    return None, None


def load_history(profile: str, search: str = "", page: int = 0) -> list:
    with sqlite3.connect(DB_PATH) as conn:
        conn.row_factory = sqlite3.Row
        q = "SELECT id, topic, level, timestamp, name FROM maps WHERE profile=?"
        p: list = [profile]
        if search:
            q += " AND (topic LIKE ? OR name LIKE ?)"
            p += [f"%{search}%", f"%{search}%"]
        q += " ORDER BY id DESC LIMIT ? OFFSET ?"
        p += [HISTORY_PAGE_SIZE, page * HISTORY_PAGE_SIZE]
        return [dict(r) for r in conn.execute(q, p).fetchall()]


def count_history(profile: str, search: str = "") -> int:
    with sqlite3.connect(DB_PATH) as conn:
        q = "SELECT COUNT(*) FROM maps WHERE profile=?"
        p: list = [profile]
        if search:
            q += " AND (topic LIKE ? OR name LIKE ?)"
            p += [f"%{search}%", f"%{search}%"]
        return conn.execute(q, p).fetchone()[0]


def load_map(map_id: int) -> dict:
    with sqlite3.connect(DB_PATH) as conn:
        row = conn.execute(
            "SELECT topic, level, json_data, name FROM maps WHERE id=?", (map_id,)
        ).fetchone()
    return {"topic": row[0], "level": row[1], "graph_data": json.loads(row[2]), "name": row[3]}


def rename_map(map_id: int, name: str):
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute("UPDATE maps SET name=? WHERE id=?", (name, map_id))


def delete_map(map_id: int):
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute("DELETE FROM maps WHERE id=?", (map_id,))


def save_annotation(map_id: int, node_id: str, note: str):
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute("DELETE FROM annotations WHERE map_id=? AND node_id=?", (map_id, node_id))
        if note.strip():
            conn.execute(
                "INSERT INTO annotations (map_id, node_id, note) VALUES (?,?,?)",
                (map_id, node_id, note),
            )


def load_annotations(map_id: int) -> dict:
    with sqlite3.connect(DB_PATH) as conn:
        conn.row_factory = sqlite3.Row
        rows = conn.execute(
            "SELECT node_id, note FROM annotations WHERE map_id=?", (map_id,)
        ).fetchall()
    return {r["node_id"]: r["note"] for r in rows}


def export_all_maps(profile: str) -> str:
    with sqlite3.connect(DB_PATH) as conn:
        conn.row_factory = sqlite3.Row
        rows = conn.execute("SELECT * FROM maps WHERE profile=?", (profile,)).fetchall()
    return json.dumps([dict(r) for r in rows], indent=2)


init_db()

# ── Gemini ────────────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """You are a CS education expert. When given a CS concept and student level, return ONLY a valid JSON object — no prose, no markdown fences, no explanation.

JSON schema:
{
  "central": "<concept name>",
  "nodes": [
    {"id": "1", "label": "<name>", "type": "central",       "description": "<max 20 words>"},
    {"id": "2", "label": "<name>", "type": "prerequisite",  "description": "<max 20 words>"},
    ...
  ],
  "edges": [
    {"from": "1", "to": "2", "label": "requires"},
    ...
  ]
}

Node types  : central | prerequisite | related | advanced | application
Edge labels : requires | extends | uses | leads_to | implements | related_to

Rules:
- 1 central node; distribute remaining nodes across types.
- Adjust depth to student level (beginner → fundamentals; advanced → deeper theory).
- Every node must appear in at least one edge.
- Keep every description under 20 words.
- Always close all arrays and the root object — JSON must be complete and valid.
- Be accurate and educational."""


def repair_json(raw: str) -> str:
    raw = raw.strip()
    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]
        raw = raw.rsplit("```", 1)[0].strip()
    return raw


def call_gemini(concept: str, level: str, node_count: int) -> dict:
    api_key = os.getenv("GEMINI_API_KEY") or st.session_state.get("api_key", "")
    if not api_key:
        raise ValueError("No API key configured.")

    client = genai.Client(api_key=api_key)
    prompt = (
        f"Concept: {concept}\n"
        f"Student level: {level}\n"
        f"Total nodes: {node_count} (1 central + {node_count - 1} others)\n"
        f"End the JSON object properly."
    )

    last_exc = None
    for attempt in range(2):
        try:
            response = client.models.generate_content(
                model=MODEL_NAME,
                contents=prompt,
                config=types.GenerateContentConfig(
                    system_instruction=SYSTEM_PROMPT,
                    temperature=0.4,
                    max_output_tokens=32000,
                    response_mime_type="application/json",
                ),
            )
            raw = response.text.strip()
            try:
                return json.loads(raw)
            except json.JSONDecodeError:
                repaired = repair_json(raw)
                try:
                    return json.loads(repaired)
                except json.JSONDecodeError as exc:
                    exc.raw_response = raw
                    raise
        except json.JSONDecodeError:
            raise
        except Exception as exc:
            last_exc = exc
            if attempt == 0:
                err = str(exc).lower()
                if "timeout" in err or "503" in err or "unavailable" in err:
                    time.sleep(2)
                    continue
            raise

    raise last_exc


def get_concept_map(concept: str, level: str, node_count: int, use_cache: bool = True):
    profile = profile_key(st.session_state.get("profile_name", ""))
    topic = normalize_topic(concept)
    if use_cache:
        cached_id, cached_data = get_cached_map(profile, topic, level, node_count)
        if cached_data is not None:
            return cached_id, cached_data, True
    graph_data = call_gemini(topic, level, node_count)
    map_id = save_map(profile, topic, level, graph_data)
    return map_id, graph_data, False


# ── Vis.js renderer ───────────────────────────────────────────────────────────

NODE_COLORS = {
    "central":      "#FF6B6B",
    "prerequisite": "#4ECDC4",
    "related":      "#45B7D1",
    "advanced":     "#96CEB4",
    "application":  "#F7DC6F",
}

LEGEND = [
    ("Central",      "#FF6B6B"),
    ("Prerequisite", "#4ECDC4"),
    ("Related",      "#45B7D1"),
    ("Advanced",     "#96CEB4"),
    ("Application",  "#F7DC6F"),
]


def _esc(s: str) -> str:
    return s.replace("\\", "\\\\").replace('"', '\\"').replace("\n", " ")


def render_vis_graph(graph_data: dict, height: int = 560,
                     quiz_mode: bool = False, annotations: dict = None) -> str:
    if annotations is None:
        annotations = {}

    nodes_js = []
    for n in graph_data["nodes"]:
        ntype  = n.get("type", "related")
        color  = NODE_COLORS.get(ntype, "#45B7D1")
        size   = 44 if ntype == "central" else 26
        fsize  = 16 if ntype == "central" else 12
        label  = "" if (quiz_mode and ntype != "central") else _esc(n["label"])
        desc   = _esc(n.get("description", ""))
        ann    = _esc(annotations.get(n["id"], ""))
        title  = ("???" if quiz_mode
                  else desc + ("  |  Note: " + ann if ann else ""))
        nodes_js.append(
            f'{{"id":"{_esc(n["id"])}","label":"{label}","title":"{title}",'
            f'"color":{{"background":"{color}","border":"#222",'
            f'"highlight":{{"background":"{color}","border":"#000"}}}},'
            f'"size":{size},"font":{{"size":{fsize},"color":"#f0f0f0"}}}}'
        )

    edges_js = []
    for e in graph_data["edges"]:
        edges_js.append(
            f'{{"from":"{_esc(e["from"])}","to":"{_esc(e["to"])}","label":"{_esc(e.get("label",""))}",'
            f'"arrows":"to","font":{{"size":9,"color":"#ccc","align":"middle"}},'
            f'"color":{{"color":"#888","highlight":"#fff"}}}}'
        )

    return f"""<!DOCTYPE html>
<html>
<head>
  <script src="https://unpkg.com/vis-network@9.1.9/standalone/umd/vis-network.min.js"></script>
  <script src="https://html2canvas.hertzen.com/dist/html2canvas.min.js"></script>
  <style>
    body  {{ margin:0; padding:0; background:transparent; }}
    #net  {{ width:100%; height:{height}px; background:#0f0f1a; border-radius:8px; border:1px solid #333; }}
    #wrap {{ position:relative; }}
    #dl   {{ position:absolute; top:8px; right:8px; background:#444; color:#fff;
             border:none; border-radius:4px; padding:4px 12px; cursor:pointer;
             font-size:12px; z-index:10; }}
    #dl:hover {{ background:#666; }}
  </style>
</head>
<body>
  <div id="wrap">
    <div id="net"></div>
    <button id="dl" onclick="downloadPNG()">⬇ PNG</button>
  </div>
  <script>
    var nodes = new vis.DataSet([{",".join(nodes_js)}]);
    var edges = new vis.DataSet([{",".join(edges_js)}]);
    new vis.Network(
      document.getElementById("net"),
      {{nodes, edges}},
      {{
        physics: {{
          solver: "forceAtlas2Based",
          forceAtlas2Based: {{ gravitationalConstant:-55, springLength:130, springConstant:0.05 }},
          stabilization: {{ iterations:200 }}
        }},
        interaction: {{ hover:true, tooltipDelay:150 }},
        nodes: {{ shape:"dot", borderWidth:2 }},
        edges: {{ smooth:{{ type:"continuous" }} }}
      }}
    );
    function downloadPNG() {{
      html2canvas(document.getElementById("net")).then(function(canvas) {{
        var a = document.createElement("a");
        a.download = "concept_map.png";
        a.href = canvas.toDataURL();
        a.click();
      }});
    }}
  </script>
</body>
</html>"""


# ── Shared render helpers ─────────────────────────────────────────────────────

def render_legend():
    cols = st.columns(len(LEGEND))
    for col, (name, color) in zip(cols, LEGEND):
        col.markdown(
            f'<span style="background:{color};color:#111;padding:3px 10px;'
            f'border-radius:12px;font-size:12px;font-weight:600">{name}</span>',
            unsafe_allow_html=True,
        )


def render_node_details(graph_data: dict, map_id: int = None, annotations: dict = None):
    if annotations is None:
        annotations = {}
    with st.expander("📋 Node Details & Annotations"):
        non_central = [n for n in graph_data["nodes"] if n.get("type") != "central"]
        cols = st.columns(2)
        half = (len(non_central) + 1) // 2
        for i, node in enumerate(non_central):
            col = cols[0] if i < half else cols[1]
            badge = NODE_COLORS.get(node.get("type", "related"), "#45B7D1")
            col.markdown(
                f'<span style="background:{badge};color:#111;padding:1px 6px;'
                f'border-radius:8px;font-size:11px">{node.get("type","")}</span> '
                f'**{node["label"]}** — {node.get("description","")}',
                unsafe_allow_html=True,
            )
            if map_id:
                existing = annotations.get(node["id"], "")
                note = col.text_input(
                    "Note", value=existing,
                    key=f"ann_{map_id}_{node['id']}",
                    placeholder="Add a personal note…",
                    label_visibility="collapsed",
                )
                if note != existing:
                    save_annotation(map_id, node["id"], note)
                    st.session_state[f"ann_cache_{map_id}"] = load_annotations(map_id)


def render_map(graph_data: dict, label: str, map_id: int = None):
    st.subheader(label)
    render_legend()

    annotations = {}
    if map_id:
        cache_key = f"ann_cache_{map_id}"
        if cache_key not in st.session_state:
            st.session_state[cache_key] = load_annotations(map_id)
        annotations = st.session_state[cache_key]

    st.components.v1.html(
        render_vis_graph(graph_data, annotations=annotations),
        height=575, scrolling=False,
    )
    render_node_details(graph_data, map_id, annotations)

    # Click-to-expand buttons
    st.markdown("**Expand a concept:**")
    non_central = [n for n in graph_data["nodes"] if n.get("type") != "central"]
    btn_cols = st.columns(min(len(non_central), 6))
    for i, node in enumerate(non_central):
        if btn_cols[i % 6].button(node["label"], key=f"expand_{map_id}_{i}_{node['label']}"):
            st.session_state["_concept_input"] = node["label"]
            st.rerun()


# ── Quiz ──────────────────────────────────────────────────────────────────────

def render_quiz(graph_data: dict):
    st.subheader("🧠 Quiz Mode")
    st.caption("Labels are hidden. Read each description and type the concept name.")
    render_legend()
    st.components.v1.html(render_vis_graph(graph_data, quiz_mode=True), height=575, scrolling=False)
    st.markdown("---")

    non_central = [n for n in graph_data["nodes"] if n.get("type") != "central"]
    answers = {}
    for node in non_central:
        answers[node["id"]] = st.text_input(
            f'*{node.get("description", "No description")}* ({node.get("type","")})',
            key=f"quiz_{node['id']}",
            placeholder="Your answer…",
        )

    if st.button("Check Answers ✔", type="primary"):
        correct = 0
        for node in non_central:
            user   = answers.get(node["id"], "").strip().lower()
            actual = node["label"].strip().lower()
            if user and (user == actual or user in actual or actual in user):
                st.success(f"✅ **{node['label']}** — correct!")
                correct += 1
            else:
                st.error(f"❌ **{node['label']}** — you said: *{answers.get(node['id'], '(blank)')}*")
        pct = int(correct / len(non_central) * 100) if non_central else 0
        st.info(f"Score: **{correct}/{len(non_central)}** ({pct}%)")


# ── Sidebar ───────────────────────────────────────────────────────────────────

with st.sidebar:
    st.header("⚙️ Settings")

    # Profile (keyed widget — never show internal "default" label)
    st.text_input("👤 Your name", key="profile_name", placeholder="e.g. Alex")
    profile = profile_key(st.session_state.get("profile_name", ""))

    # API key
    if not os.getenv("GEMINI_API_KEY"):
        key_input = st.text_input("Gemini API Key", type="password",
                                  help="Or set GEMINI_API_KEY in a .env file.")
        if key_input:
            st.session_state["api_key"] = key_input
    else:
        st.success("API key loaded from .env", icon="✅")

    st.divider()
    st.subheader("💡 Try these topics")
    examples = [
        "Binary Search Tree", "Dynamic Programming", "TCP/IP Stack",
        "Neural Networks", "OS Process Scheduling", "SQL Joins",
        "Recursion", "Hash Tables", "Dijkstra's Algorithm", "Public Key Cryptography",
    ]
    for ex in examples:
        if st.button(ex, key=f"ex_{ex}", use_container_width=True):
            st.session_state["_concept_input"] = ex

    st.divider()
    st.subheader("🕘 History")

    search = st.text_input("🔍 Search history", placeholder="Filter by topic or name…", key="hist_search")
    profile_key(st.session_state.get("profile_name", ""))
    page    = st.session_state.get("hist_page", 0)
    total   = count_history(profile, search)
    history = load_history(profile, search, page)

    if not history:
        st.caption("No saved maps yet." if not search else "No results found.")
    else:
        for row in history:
            display = row["name"] or row["topic"]
            label   = f"{display} · {row['level']} · {row['timestamp']}"
            ca, cb, cc = st.columns([4, 1, 1])
            if ca.button(label, key=f"hist_{row['id']}", use_container_width=True):
                m = load_map(row["id"])
                st.session_state["_loaded_map"]    = m
                st.session_state["_loaded_map_id"] = row["id"]
                st.session_state["_last_graph"]    = m["graph_data"]
            if cb.button("✏️", key=f"ren_{row['id']}", help="Rename"):
                st.session_state["_renaming"] = row["id"]
            if cc.button("✕", key=f"del_{row['id']}", help="Delete"):
                delete_map(row["id"])
                st.rerun()

        # Inline rename form
        if st.session_state.get("_renaming"):
            rid      = st.session_state["_renaming"]
            new_name = st.text_input("New name", key="rename_input", placeholder="Enter display name…")
            rc1, rc2 = st.columns(2)
            if rc1.button("Save", use_container_width=True):
                rename_map(rid, new_name)
                st.session_state.pop("_renaming")
                st.rerun()
            if rc2.button("Cancel", use_container_width=True):
                st.session_state.pop("_renaming")
                st.rerun()

        # Pagination
        total_pages = max(1, (total + HISTORY_PAGE_SIZE - 1) // HISTORY_PAGE_SIZE)
        if total_pages > 1:
            pc1, pc2, pc3 = st.columns([1, 2, 1])
            if pc1.button("◀", disabled=page == 0, key="pg_prev"):
                st.session_state["hist_page"] = page - 1
                st.rerun()
            pc2.caption(f"Page {page + 1} / {total_pages}")
            if pc3.button("▶", disabled=page >= total_pages - 1, key="pg_next"):
                st.session_state["hist_page"] = page + 1
                st.rerun()

    st.divider()
    # Export
    export_data = export_all_maps(profile)
    st.download_button(
        "⬇ Export my maps (JSON)",
        data=export_data,
        file_name=f"maps_{profile}.json",
        mime="application/json",
        use_container_width=True,
    )

    st.divider()
    st.caption(f"Powered by **{MODEL_NAME}**")


# ── Main area — tabs ──────────────────────────────────────────────────────────

st.title("🗺️ CS Concept Cartographer")
st.caption("Enter any CS topic and see how it connects to the broader knowledge map — powered by Gemini.")

tab_map, tab_compare, tab_quiz = st.tabs(["🗺️ Map", "⚖️ Compare", "🧠 Quiz"])

# ── Tab: Map ──────────────────────────────────────────────────────────────────

with tab_map:
    # Reload from history
    if "_loaded_map" in st.session_state:
        loaded    = st.session_state["_loaded_map"]
        loaded_id = st.session_state.get("_loaded_map_id")
        st.session_state["_last_graph"] = loaded["graph_data"]
        render_map(
            loaded["graph_data"],
            f"Concept Map: {loaded['topic']}  ·  {loaded['level']}  *(from history)*",
            map_id=loaded_id,
        )
        if st.button("✕ Close history view", key="close_history"):
            st.session_state.pop("_loaded_map", None)
            st.session_state.pop("_loaded_map_id", None)
            st.rerun()
        st.divider()

    # Input row
    col1, col2, col3, col4 = st.columns([3, 2, 2, 1])
    with col1:
        if "_concept_input" in st.session_state:
            st.session_state["concept_input"] = st.session_state.pop("_concept_input")
        concept = st.text_input(
            "CS Concept",
            key="concept_input",
            placeholder="e.g. Binary Search Tree…",
            label_visibility="collapsed",
        )
    with col2:
        level = st.selectbox(
            "Level", ["Beginner", "Intermediate", "Advanced"],
            key="level_select",
            label_visibility="collapsed",
        )
    with col3:
        if "node_count" not in st.session_state:
            st.session_state["node_count"] = DEFAULT_NODES["Beginner"]
        if st.session_state.get("_prev_level") != level:
            st.session_state["node_count"] = DEFAULT_NODES.get(level, 10)
            st.session_state["_prev_level"] = level
        node_count = st.slider(
            "Nodes", 6, 20,
            key="node_count",
            label_visibility="collapsed",
            help="Number of concept nodes to generate",
        )
    with col4:
        go = st.button("Map It ➜", type="primary", use_container_width=True)

    use_cache = st.checkbox("Use cached result if available", value=True,
                            help="Skip the API call if this topic+level was already generated.")

    if go and concept:
        with st.status(f"Mapping **{concept}**…", expanded=True) as status:
            try:
                st.write("Checking cache…" if use_cache else "Calling Gemini…")
                map_id, graph_data, from_cache = get_concept_map(
                    concept.strip(), level, node_count, use_cache
                )
                if from_cache:
                    st.write("✅ Loaded from cache — no API call needed.")
                else:
                    st.write("✅ Response received and saved.")
                status.update(
                    label=f"Done{'  (cached)' if from_cache else ''}!",
                    state="complete",
                )
                st.session_state["_last_graph"] = graph_data
                render_map(graph_data, f"Concept Map: {graph_data['central']}  ·  {level}", map_id=map_id)

            except json.JSONDecodeError as exc:
                status.update(label="Parse error", state="error")
                st.error(f"JSON parse error: {exc}")
                raw = getattr(exc, "raw_response", None)
                if raw:
                    st.subheader("Raw response from Gemini:")
                    st.code(raw, language="text")
            except ValueError as exc:
                status.update(label="Configuration error", state="error")
                st.error(str(exc))
                st.info("Add your Gemini API key in the sidebar or set GEMINI_API_KEY in .env")
            except Exception as exc:
                status.update(label="Error", state="error")
                st.error(f"Unexpected error: {exc}")

    elif go and not concept:
        st.warning("Please enter a CS concept first.")

# ── Tab: Compare ──────────────────────────────────────────────────────────────

with tab_compare:
    st.subheader("Compare Two Concepts Side by Side")
    cc1, cc2 = st.columns(2)
    with cc1:
        concept_a = st.text_input("Concept A", placeholder="e.g. Stack", key="cmp_a")
        level_a   = st.selectbox("Level A", ["Beginner", "Intermediate", "Advanced"], key="lvl_a")
    with cc2:
        concept_b = st.text_input("Concept B", placeholder="e.g. Queue", key="cmp_b")
        level_b   = st.selectbox("Level B", ["Beginner", "Intermediate", "Advanced"], key="lvl_b")

    cmp_go = st.button("Compare ➜", type="primary", key="cmp_go")

    if cmp_go and concept_a and concept_b:
        left, right = st.columns(2)
        with left:
            with st.spinner(f"Mapping {concept_a}…"):
                try:
                    mid_a, data_a, _ = get_concept_map(concept_a.strip(), level_a, 10)
                    st.session_state["_last_graph"] = data_a
                    render_map(data_a, f"{concept_a}  ·  {level_a}", map_id=mid_a)
                except Exception as exc:
                    st.error(f"{concept_a}: {exc}")
        with right:
            with st.spinner(f"Mapping {concept_b}…"):
                try:
                    mid_b, data_b, _ = get_concept_map(concept_b.strip(), level_b, 10)
                    render_map(data_b, f"{concept_b}  ·  {level_b}", map_id=mid_b)
                except Exception as exc:
                    st.error(f"{concept_b}: {exc}")
    elif cmp_go:
        st.warning("Please enter both concepts.")

# ── Tab: Quiz ─────────────────────────────────────────────────────────────────

with tab_quiz:
    if "_last_graph" in st.session_state:
        render_quiz(st.session_state["_last_graph"])
    else:
        st.info("Generate a concept map in the **Map** tab first, then come here to quiz yourself on it.")
