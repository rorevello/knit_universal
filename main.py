import gradio as gr
from rdflib import Graph, URIRef, Literal, RDF, RDFS, OWL
from collections import defaultdict, Counter
import requests
import os
import tempfile
import traceback
from difflib import SequenceMatcher
from urllib.parse import urlparse
from time import sleep
from owlready2 import get_ontology, ThingClass
from search_order_centroid import distancia_al_centroide
import pandas as pd
from pyvis.network import Network


def convert_obo_to_owl_url(url):
    if "obolibrary.org/obo/" in url and url.endswith(".obo"):
        path = urlparse(url).path
        base_name = os.path.basename(path).replace(".obo", "")
        return f"http://purl.obolibrary.org/obo/{base_name}.owl"
    return url


def get_keyword_uris_from_decarat(keyword: str, DECARAT_API, source):
    k = 10
    response = requests.get(
        f"{DECARAT_API}/search?query={keyword}&k={k}&source={source}"
    )
    if response.status_code != 200:
        raise ValueError(f"Error for '{keyword}': {response.status_code}")
    results = response.json().get("results", [])
    uri_onto_pairs = []
    ontos = []
    for result in results:
        for entry in result["entries"]:
            uri = entry["uri"]

            if (
                "http" in uri
                and entry["ontology"] != "https://chideba.github.io/ontove/ontology.owl"
            ):
                if entry["source"] == "bioportal":
                    api_key = "apikey bioportal"
                    ontology_url = (
                        str(entry["ontology"]) + "/download?apikey=" + api_key
                    )
                else:
                    ontology_url = entry["ontology"]

                if ontology_url not in ontos:
                    ontos.append(ontology_url)
                    uri_onto_pairs.append((keyword, uri, ontology_url))

    return uri_onto_pairs


def parse_ontology_from_url(url):
    g = Graph()
    for fmt in ["xml", "ttl", "nt", "json-ld", "trig", "n3", "obo", "rdf", "o"]:
        try:
            u = url
            if u.endswith(".obo"):
                u = convert_obo_to_owl_url(u)
            g.parse(u, format=fmt)
            return g
        except:
            print(f"The ontology could not be loaded from {url} with format {fmt}")
            continue
    raise ValueError(f"The ontology could not be loaded from {url}")


def get_parent_class_local(graph, class_uri, visited=None):
    if visited is None:
        visited = set()
    parents = set()
    for _, _, parent in graph.triples((URIRef(class_uri), RDFS.subClassOf, None)):
        if parent not in visited:
            visited.add(parent)
            parents.add(parent)
            parents |= get_parent_class_local(graph, str(parent), visited)
    return parents


def get_parent_class_local_and_enrich(
    class_uri, source_graph, result_graph, visited=None
):
    if visited is None:
        visited = set()
    class_ref = URIRef(class_uri)
    if class_ref in visited:
        return
    visited.add(class_ref)
    result_graph += get_class_info_local(source_graph, class_ref)
    for _, _, parent in source_graph.triples((class_ref, RDFS.subClassOf, None)):
        if isinstance(parent, URIRef):
            get_parent_class_local_and_enrich(
                parent, source_graph, result_graph, visited
            )


def get_class_info_local(graph, class_uri):
    result_graph = Graph()
    class_ref = URIRef(class_uri)
    count = 0
    warnings = []
    label = next(graph.objects(class_ref, RDFS.label), None)
    if label:
        result_graph.add((class_ref, RDFS.label, label))
    if (class_ref, RDF.type, OWL.Class) in graph:
        result_graph.add((class_ref, RDF.type, OWL.Class))
    for p, o in graph.predicate_objects(subject=class_ref):
        result_graph.add((class_ref, p, o))
        count += 1
        if isinstance(o, URIRef):
            for pp, oo in graph.predicate_objects(subject=o):
                result_graph.add((o, pp, oo))
            if (o, RDF.type, OWL.Class) in graph:
                result_graph.add((o, RDF.type, OWL.Class))
            lbl = next(graph.objects(o, RDFS.label), None)
            if lbl:
                result_graph.add((o, RDFS.label, lbl))
    for s, p in graph.subject_predicates(object=class_ref):
        if isinstance(s, URIRef) and (s, RDF.type, OWL.ObjectProperty) in graph:
            result_graph.add((s, p, class_ref))
            count += 1
            for pp, oo in graph.predicate_objects(subject=s):
                result_graph.add((s, pp, oo))
        elif isinstance(s, URIRef):
            warnings.append(f"⚠ Clase {class_ref} usada sin tipado en {s}")
    for rel in [OWL.equivalentClass, OWL.disjointWith, OWL.complementOf]:
        for _, _, related in graph.triples((class_ref, rel, None)):
            if not isinstance(related, URIRef):
                warnings.append(f"⚠ {rel} apunta a no-URI: {related}")
                continue
            uri_str = str(related).strip()
            if (
                uri_str.startswith("http://org.semanticweb.owlapi/error#")
                or not uri_str
            ):
                warnings.append(f"⚠ {rel} apunta a error o vacío: {related}")
                continue
            result_graph.add((class_ref, rel, related))
            result_graph.add((related, RDF.type, OWL.Class))
            lbl = next(graph.objects(related, RDFS.label), None)
            if lbl:
                result_graph.add((related, RDFS.label, lbl))
            for pp, oo in graph.predicate_objects(subject=related):
                result_graph.add((related, pp, oo))
    if warnings:
        with open("errors_detected.log", "a", encoding="utf-8") as logf:
            for w in warnings:
                logf.write(w + "\n")
    return result_graph if count > 0 else Graph()


def remove_empty_property_chains(graph):
    from rdflib.namespace import RDF

    to_remove = []
    for s, p, o in graph.triples((None, OWL.propertyChainAxiom, None)):
        if o == RDF.nil or not list(graph.objects(o, RDF.first)):
            to_remove.append((s, p, o))
    for s, p, o in to_remove:
        graph.remove((s, p, o))
        current = o
        while current and current != RDF.nil:
            for pred, obj in graph.predicate_objects(current):
                graph.remove((current, pred, obj))
            rest = list(graph.objects(current, RDF.rest))
            current = rest[0] if rest else None
    return graph


def remove_all_blank_nodes(graph):
    from rdflib.term import BNode

    blanks = {n for n in graph.all_nodes() if isinstance(n, BNode)}
    for b in blanks:
        for t in list(graph.triples((b, None, None))) + list(
            graph.triples((None, None, b))
        ):
            graph.remove(t)
    return graph


def clean_ontology(graph):
    to_rm_cls = set()
    to_rm_prop = set()
    for cls in graph.subjects(RDF.type, OWL.Class):
        if str(cls).startswith("http://org.semanticweb.owlapi/error#") or not list(
            graph.predicate_objects(cls)
        ):
            to_rm_cls.add(cls)
    for cls in to_rm_cls:
        graph.remove((cls, None, None))
    for prop in graph.subjects(RDF.type, RDF.Property):
        if not list(graph.predicate_objects(prop)):
            to_rm_prop.add(prop)
    for p in to_rm_prop:
        graph.remove((p, None, None))
    return graph


def enrich_missing_classes(graph, source_graph):
    enriched = 0
    for cls in set(graph.subjects(RDF.type, OWL.Class)):
        if not any(graph.objects(cls, RDFS.label)) or not any(
            graph.predicate_objects(cls)
        ):
            for p, o in source_graph.predicate_objects(cls):
                graph.add((cls, p, o))
            lbl = next(source_graph.objects(cls, RDFS.label), None)
            if lbl:
                graph.add((cls, RDFS.label, lbl))
            if (cls, RDF.type, OWL.Class) in source_graph:
                graph.add((cls, RDF.type, OWL.Class))
            enriched += 1
    return enriched


def enrich_class_and_ancestors(class_uri, source_graph, result_graph, visited=None):
    if visited is None:
        visited = set()
    ref = URIRef(class_uri)
    if ref in visited:
        return
    visited.add(ref)
    result_graph += get_class_info_local(source_graph, class_uri)
    for _, _, parent in source_graph.triples((ref, RDFS.subClassOf, None)):
        if isinstance(parent, URIRef):
            enrich_class_and_ancestors(parent, source_graph, result_graph, visited)


# -----------------------------
# Wrapper Gradio
# -----------------------------


def run_pipeline(keywords_str, DECARAT_API, source_onto, search_order):
    # 1) parse keywords
    keywords = [k.strip() for k in keywords_str.splitlines() if k.strip()]

    # 2) search keywords in api
    ontology_uri_map = defaultdict(list)

    frequency_counter = Counter()
    list_ontos_uris = []

    dicc_kw_uris = {}
    for kw in keywords:
        try:
            pairs = get_keyword_uris_from_decarat(kw, DECARAT_API, source_onto)
            sleep(0.2)
            list_ontos_uris.append(pairs)
        except:
            continue
    for pairs in list_ontos_uris:
        for kw, uri, onto in pairs:
            if uri not in [u for _, u in ontology_uri_map[onto]]:
                ontology_uri_map[onto].append((kw, uri))
                frequency_counter[onto] += 1

    # 3) order ontologies
    if search_order == "knit":
        sorted_ontos = sorted(
            frequency_counter.items(), key=lambda x: x[1], reverse=True
        )

    if search_order == "Nc" or search_order == "DSRC":
        dicc_nc, dicc_dsrc = {}, {}

        for onto in ontology_uri_map:
            # api_key = "apikey bioportal", e.g. download?apikey=8c9zi0cd-66-49667-b9f2-956677b0528b
            onto1 = onto.replace("/download?apikey=...", "")
            r = requests.get(
                f"{DECARAT_API}/dsr_class?source={source_onto}&ontology={onto1}"
            )
            data = r.json()
            dicc_nc[onto] = data.get("Nc")
            dicc_dsrc[onto] = data.get("DSRC")

        if search_order == "Nc":
            sorted_ontos = sorted(
                dicc_nc.items(),
                key=lambda x: (x[1] is None, -x[1] if x[1] is not None else 0),
            )
        if search_order == "DSRC":  # DSRC
            sorted_ontos = sorted(
                dicc_dsrc.items(),
                key=lambda x: (
                    x[1] is None,
                    x[1] if x[1] is not None else float("inf"),
                ),
            )

    if search_order == "centroid":
        cent_d = {}
        for onto in ontology_uri_map:
            r = requests.get(
                f"{DECARAT_API}/centroid?source={source_onto}&ontology={onto}"
            )
            cent = r.json().get("centroid")
            cent_d[onto] = (
                distancia_al_centroide(keywords, cent)[0] if cent is not None else None
            )
        sorted_ontos = sorted(
            cent_d.items(), key=lambda x: (x[1] is None, x[1]), reverse=True
        )

    # 4) selection minimum keywords
    selected = defaultdict(list)
    covered = set()
    for onto, _ in sorted_ontos:
        for kw, uri in ontology_uri_map[onto]:
            if kw not in covered:
                selected[onto].append((kw, uri))
                covered.add(kw)
        if covered == set(keywords):
            break

    # 5) download knit_graph
    knit_graph = Graph()
    for onto, entries in selected.items():
        try:
            g = parse_ontology_from_url(onto)
            for kw, uri in entries:
                enrich_class_and_ancestors(uri, g, knit_graph)
                enrich_missing_classes(knit_graph, g)

        except:
            continue

    # 6) clear
    knit_graph = remove_empty_property_chains(knit_graph)
    knit_graph = remove_all_blank_nodes(knit_graph)
    knit_graph = clean_ontology(knit_graph)

    # 7)  triples / DataFrame
    triples = [(str(s), str(p), str(o)) for s, p, o in knit_graph]

    df = pd.DataFrame(triples, columns=["subject", "predicate", "object"])

    # 8) save OWL
    print(f"[DEBUG] triples in knit_graph: {len(knit_graph)}")

    owl_tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".owl")
    owl_tmp.write(knit_graph.serialize(format="xml").encode("utf-8"))
    owl_tmp.close()

    net = Network(
        height="700px",
        width="100%",
        bgcolor="#ffffff",
        font_color="black",
        directed=True,
        notebook=False,
    )

    net.barnes_hut(
        gravity=-8000,
        central_gravity=0.2,
        spring_length=200,
        spring_strength=0.001,
        damping=0.09,
    )

    net.set_options(
        """
        var options = {
        "physics": {
            "enabled": true,
            "barnesHut": {
            "gravitationalConstant": -20000,
            "centralGravity": 0.3,
            "springLength": 150,
            "springConstant": 0.005,
            "damping": 0.09
            },
            "minVelocity": 0.75
        },
        "nodes": {
            "borderWidth": 2,
            "shape": "dot",
            "size": 16,
            "color": {
            "border": "#222222",
            "background": "#666666"
            },
            "font": {
            "color": "#ffffff"
            }
        },
        "edges": {
            "color": {
            "color":"#aaaaaa",
            "highlight":"#ff0000"
            },
            "width":2,
            "smooth": {
            "type": "dynamic"
            }
        }
        }
        """
    )

    seen_nodes = set()
    for s, p, o in triples:
        label_s = s.rsplit("/", 1)[-1].rsplit("#", 1)[-1]
        label_o = o.rsplit("/", 1)[-1].rsplit("#", 1)[-1]

        if s not in seen_nodes:
            net.add_node(s, label=label_s, title=s)
            seen_nodes.add(s)
        if o not in seen_nodes:
            net.add_node(o, label=label_o, title=o)
            seen_nodes.add(o)

        net.add_edge(s, o, label=p, title=p, width=1.5)

    # 5) Write the HTML to a temporary file
    html_tmp = tempfile.NamedTemporaryFile(
        delete=False, suffix=".html", mode="w", encoding="utf-8"
    )
    net.write_html(html_tmp.name, notebook=False)
    html_tmp.close()

    return df, owl_tmp.name, html_tmp.name


# -----------------------------
# Interfaz Gradio
# -----------------------------

try:
    resp = requests.get("http://0.0.0.0:8019/sources")
    resp.raise_for_status()
    ontology_sources = resp.json()
except Exception as e:
    print(f" Could not get API sources: {e}")
    ontology_sources = ["obo_fundry", "bioportal", "w3id"]


iface = gr.Interface(
    fn=run_pipeline,
    inputs=[
        gr.Textbox(
            label="Keywords (one per line)",
            value="Williams syndrome\nMarfan syndrome\ncongenital contractural arachnodactyly\nacute myeloid leukaemia",
        ),
        gr.Textbox(label="Storage component URL", value="http://0.0.0.0:8019"),
        gr.Dropdown(
            label="Ontology source",
            choices=ontology_sources,
            value=ontology_sources[0] if ontology_sources else None,
        ),
        gr.Dropdown(
            label="Search strategy",
            choices=["knit", "Nc", "DSRC", "centroid"],
            value="knit",
        ),
    ],
    outputs=[
        gr.Dataframe(label="Resulting triples"),
        gr.File(label="Download OWL"),
        gr.File(label="Download HTML graph"),
    ],
    title="Ontology Generator",
    description="Provide your parameters, run the pipeline, and download your ontology and interactive graph.",
)


if __name__ == "__main__":
    iface.launch(server_name="0.0.0.0", server_port=8504)
