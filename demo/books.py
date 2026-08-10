"""Write three small PDFs, so the `ask` demo in the README is reproducible.

Real books cannot be committed to a repo, and a demo that needs a 40 MB
textbook nobody has is a demo nobody runs. These are tiny, but they have the
shape that matters: one has a proper outline, one has only headings, and their
subjects overlap enough that routing to the wrong book is a real possibility.

    python demo/books.py ./books
    graphroute ask ./books "what is leukemia" --passages 3
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir))

from demo.synthetic_pdf import write_pdf, write_pdf_with_outline  # noqa: E402

HEMATOLOGY = [
    ["Chapter 87 Erythrocyte Production",
     "The bone marrow generates red cells through erythropoiesis, a process",
     "regulated by erythropoietin released from the renal cortex in response",
     "to tissue hypoxia. Iron availability limits haemoglobin synthesis and",
     "is the commonest cause of a hypoproliferative anaemia worldwide.",
     "Reticulocyte counts measure the marrow response to anaemia directly."],
    ["Reticulocyte kinetics",
     "A reticulocyte matures over roughly four days, one of which is spent",
     "in the circulation. Counting them separates a marrow that is failing",
     "from a marrow that is responding to peripheral destruction."],
    ["Chapter 88 Acute Lymphoblastic Leukemia",
     "Leukemia is a malignant proliferation of haematopoietic precursor cells",
     "that accumulate in the bone marrow and displace normal haematopoiesis.",
     "Acute lymphoblastic leukemia arises from lymphoid progenitors and is the",
     "commonest malignancy of childhood, with a peak incidence between two and",
     "five years of age. Patients present with fatigue, pallor, bruising and",
     "recurrent infection, all consequences of marrow replacement by blasts."],
    ["Diagnosis of acute lymphoblastic leukemia",
     "Diagnosis requires bone marrow aspiration demonstrating at least twenty",
     "percent lymphoblasts. Immunophenotyping by flow cytometry distinguishes",
     "B lineage from T lineage disease, and cytogenetic analysis identifies the",
     "translocations that determine risk group and therefore treatment intensity."],
    ["Treatment of acute lymphoblastic leukemia",
     "Therapy proceeds in phases: induction, consolidation and a prolonged",
     "maintenance period lasting two to three years. Central nervous system",
     "prophylaxis with intrathecal methotrexate is mandatory because systemic",
     "chemotherapy does not reliably cross the blood brain barrier."],
    ["Chapter 89 Chronic Myeloid Leukemia",
     "Chronic myeloid leukemia is defined by the Philadelphia chromosome, a",
     "translocation between chromosomes nine and twenty two producing the BCR",
     "ABL fusion kinase. Tyrosine kinase inhibitors transformed this disease",
     "from uniformly fatal to a chronic condition managed with a daily tablet."],
]

HEMATOLOGY_OUTLINE = [
    ("Ch 87: Erythrocyte Production", 0, 0),
    ("Reticulocyte kinetics", 1, 1),
    ("Ch 88: Acute Lymphoblastic Leukemia", 2, 0),
    ("Diagnosis", 3, 1),
    ("Treatment", 4, 1),
    ("Ch 89: Chronic Myeloid Leukemia", 5, 0),
]

PHARMACOLOGY = [
    ["Chapter 12 Antimetabolites",
     "Methotrexate inhibits dihydrofolate reductase and thereby blocks",
     "thymidylate synthesis, killing cells in S phase. It remains a backbone",
     "of maintenance therapy in lymphoblastic disease and in rheumatoid",
     "arthritis at far lower weekly doses.",
     "Leucovorin rescue permits high dose administration without lethal",
     "myelosuppression, and serum level monitoring guides its duration."],
    ["12.1 Purine analogues",
     "Mercaptopurine and thioguanine are activated by hypoxanthine guanine",
     "phosphoribosyltransferase. TPMT genotype predicts which patients will",
     "suffer profound myelosuppression at conventional doses, and testing",
     "before treatment is now standard practice."],
    ["Chapter 13 Kinase inhibitors",
     "Imatinib binds the inactive conformation of the ABL kinase domain and",
     "was the first targeted agent to convert a lethal leukemia into a",
     "manageable chronic illness. Resistance arises through point mutations",
     "in the kinase domain, of which T315I resists all first generation drugs."],
]

NETWORKS = [
    ["Chapter 4 Congestion control",
     "TCP infers congestion from loss and halves its window on detecting it.",
     "Modern algorithms such as BBR model the bottleneck bandwidth and round",
     "trip propagation time instead, avoiding the standing queues that loss",
     "based control necessarily builds in deep buffers."],
    ["4.1 Queue management",
     "Active queue management drops or marks packets before a buffer fills.",
     "CoDel measures the sojourn time of packets through the queue and acts",
     "on persistent delay rather than on queue length in bytes."],
]


def write_books(out_dir: str) -> list:
    os.makedirs(out_dir, exist_ok=True)
    paths = [
        write_pdf_with_outline(
            os.path.join(out_dir, "williams-hematology.pdf"),
            HEMATOLOGY, HEMATOLOGY_OUTLINE, title="Williams Hematology",
        ),
        write_pdf(
            os.path.join(out_dir, "clinical-pharmacology.pdf"),
            PHARMACOLOGY, title="Clinical Pharmacology",
        ),
        write_pdf(
            os.path.join(out_dir, "computer-networks.pdf"),
            NETWORKS, title="Computer Networks",
        ),
    ]
    return paths


if __name__ == "__main__":
    target = sys.argv[1] if len(sys.argv) > 1 else "./books"
    for path in write_books(target):
        print(path)
