#!/usr/bin/env python3
"""Render GPC > TPC > SM nested ASCII boxes from sm_gpc_mapping output.

Reads the structured stdout of bin/sm_gpc_mapping (or a saved log via
--input) and prints one nested-box rendering per cluster shape, with each
SM cell labeled by the cluster_id and rank_in_cluster of the first block
that landed on it.
"""

import argparse
import re
import sys
from collections import defaultdict


GPC_LINE = re.compile(r"^#\s+GPC\s+(\d+):\s+(\d+)\s+SMs\s+\[([\d,]+)\]")
SHAPE_HDR = re.compile(
    r"^#\s+===\s+cluster\s+shape\s+(\S+)\s+"
    r"\(size=(\d+),\s+nclusters=(\d+),\s+"
    r"grid=(\d+)x(\d+)x(\d+),\s+blocks=(\d+)\)\s+==="
)
CSV_HDR = re.compile(
    r"^#\s+CSV:\s+linear,blockIdx_x,blockIdx_y,blockIdx_z,"
    r"cluster_id,rank_in_cluster,smid,gpcId,tpcId"
)
ANSI_RE = re.compile(r"\033\[[0-9;]*m")


def vislen(s):
    return len(ANSI_RE.sub("", s))


PALETTE = [9, 10, 11, 12, 13, 14, 51, 87, 123, 159, 195, 219, 226, 208, 99, 250]


def colored(text, cid, enabled):
    if not enabled or cid is None:
        return text
    code = PALETTE[cid % len(PALETTE)]
    return f"\033[38;5;{code}m{text}\033[0m"


def parse(text):
    gpc_to_smids = {}
    shapes = []
    cur = None
    in_csv = False
    for line in text.splitlines():
        m = GPC_LINE.match(line)
        if m:
            gpc_to_smids[int(m.group(1))] = [int(x) for x in m.group(3).split(",")]
            continue
        m = SHAPE_HDR.match(line)
        if m:
            cur = {
                "name": m.group(1),
                "size": int(m.group(2)),
                "nclusters": int(m.group(3)),
                "grid": (int(m.group(4)), int(m.group(5)), int(m.group(6))),
                "blocks": int(m.group(7)),
                # smid -> list of (cluster_id, rank, tpc, gpc, linear) tuples,
                # sorted by linear (the dispatch order on the SM)
                "occ": defaultdict(list),
            }
            shapes.append(cur)
            in_csv = False
            continue
        if cur is not None and CSV_HDR.match(line):
            in_csv = True
            continue
        if in_csv and line and not line.startswith("#"):
            try:
                p = line.split(",")
                linear, cid, rank, smid, gpc, tpc = (
                    int(p[0]),
                    int(p[4]),
                    int(p[5]),
                    int(p[6]),
                    int(p[7]),
                    int(p[8]),
                )
            except (IndexError, ValueError):
                in_csv = False
                continue
            cur["occ"][smid].append((cid, rank, tpc, gpc, linear))
    # sort each SM's occupant list by linear so the visualizer can render
    # them in dispatch order
    for s in shapes:
        for smid in s["occ"]:
            s["occ"][smid].sort(key=lambda t: t[4])
    return gpc_to_smids, shapes


def build_smid_to_tpc(shapes):
    out = {}
    for s in shapes:
        for smid, occs in s["occ"].items():
            if occs:
                out[smid] = occs[0][2]  # tpc is identical for every occupant
    return out


def gpc_tpc_layout(gpc_to_smids, smid_to_tpc):
    """gpc -> list of (tpc_id, [smids in tpc]), ordered by the lowest smid in
    each TPC so reading L-to-R matches ascending physical SM ids. Hardware
    tpc_id is preserved as the label, which makes the permutation visible."""
    layout = {}
    for gpc, smids in gpc_to_smids.items():
        d = defaultdict(list)
        for sm in smids:
            tpc = smid_to_tpc.get(sm)
            if tpc is None:
                d[-1].append(sm)
            else:
                d[tpc].append(sm)
        for k in d:
            d[k].sort()
        layout[gpc] = sorted(d.items(), key=lambda kv: kv[1][0])
    return layout


# ---------- rendering ----------

TPC_WIDTH = 11      # outer
TPC_INNER = TPC_WIDTH - 2  # 9
TPCS_PER_ROW = 4
GAP = " "


def pad_to(s, vis_text, width):
    extra = width - len(vis_text)
    if extra <= 0:
        return s
    left = extra // 2
    right = extra - left
    return " " * left + s + " " * right


def render_sm_lines(smid, occ, color, occ_rows):
    """occ_rows = number of occupant lines to emit per SM cell (uniform within
    a shape, computed from the max occupants/SM observed in the shape).
    Returns 1 sm-id line + occ_rows occupant lines."""
    if smid is None:
        return ["|" + " " * TPC_INNER + "|"] * (1 + occ_rows)
    sm_label = f"sm{smid:>3d}"
    head = "|" + sm_label.center(TPC_INNER) + "|"
    occs = occ.get(smid, []) if isinstance(occ, dict) else occ.get(smid, [])
    if not occs:
        # SM not touched in this shape's launch.
        body = ["|" + pad_to("-", "-", TPC_INNER) + "|"]
        body += ["|" + " " * TPC_INNER + "|"] * (occ_rows - 1)
        return [head] + body
    body = []
    for i in range(occ_rows):
        if i < len(occs):
            cid, rank = occs[i][0], occs[i][1]
            cr_raw = f"c{cid}:{rank}"
            cr = colored(cr_raw, cid, color)
            body.append("|" + pad_to(cr, cr_raw, TPC_INNER) + "|")
        else:
            body.append("|" + " " * TPC_INNER + "|")
    return [head] + body


def render_tpc(tpc_id, smids, occ, color, occ_rows):
    """Return list of strings, each width TPC_WIDTH (visible). Height is
    1 (top) + 2 * (1 + occ_rows) + 1 (sep) + 1 (bottom)."""
    label = f" TPC {tpc_id} " if tpc_id >= 0 else " TPC ? "
    extra = TPC_INNER - len(label)
    left = extra // 2
    right = extra - left
    top = "+" + "-" * left + label + "-" * right + "+"
    sep = "|" + "-" * TPC_INNER + "|"
    bot = "+" + "-" * TPC_INNER + "+"
    sm0 = smids[0] if len(smids) >= 1 else None
    sm1 = smids[1] if len(smids) >= 2 else None
    return [
        top,
        *render_sm_lines(sm0, occ, color, occ_rows),
        sep,
        *render_sm_lines(sm1, occ, color, occ_rows),
        bot,
    ]


def empty_tpc_block(occ_rows):
    h = 1 + 2 * (1 + occ_rows) + 1 + 1  # top + 2 SMs + sep + bot
    return [" " * TPC_WIDTH] * h


def hjoin(boxes, gap=GAP):
    if not boxes:
        return []
    return [gap.join(b[r] for b in boxes) for r in range(len(boxes[0]))]


def render_gpc_boxed(gpc_id, tpcs, occ, color, occ_rows):
    sm_count = sum(len(smids) for _, smids in tpcs)
    n_tpcs = len(tpcs)
    inner_w = TPCS_PER_ROW * TPC_WIDTH + (TPCS_PER_ROW - 1) * len(GAP)
    margin = "| "
    rmargin = " |"
    width = inner_w + len(margin) + len(rmargin)

    label = f"GPC {gpc_id} [{sm_count} SMs]"
    head = f"+-- {label} "
    top = head + "-" * (width - len(head) - 1) + "+"
    bot = "+" + "-" * (width - 2) + "+"

    out = [top]
    for i in range(0, n_tpcs, TPCS_PER_ROW):
        chunk = tpcs[i : i + TPCS_PER_ROW]
        boxes = [render_tpc(tid, smids, occ, color, occ_rows) for tid, smids in chunk]
        while len(boxes) < TPCS_PER_ROW:
            boxes.append(empty_tpc_block(occ_rows))
        for joined in hjoin(boxes):
            pad = inner_w - vislen(joined)
            out.append(margin + joined + " " * pad + rmargin)
    out.append(bot)
    return "\n".join(out)


def render_gpc_compact(gpc_id, tpcs, occ, color):
    sm_count = sum(len(smids) for _, smids in tpcs)
    # Each line: "TPC N: [ smX c0:0,c5:1 | smX c1:0 ]"
    body_lines = []
    for tpc_id, smids in tpcs:
        cells = []
        for sm in smids[:2]:
            sm_label = f"sm{sm:>3d}"
            occs = occ.get(sm, [])
            if not occs:
                cr_raw = "-"
                cr = cr_raw
            else:
                cr_raw = ",".join(f"c{c}:{r}" for c, r, *_ in occs)
                cr = ",".join(colored(f"c{c}:{r}", c, color) for c, r, *_ in occs)
            cell_vis = f"{sm_label} {cr_raw}"
            cell = f"{sm_label} {cr}"
            cells.append((cell, cell_vis))
        while len(cells) < 2:
            cells.append(("", ""))
        # equalize cell widths
        cell_w = max(len(cv) for _, cv in cells) if cells else 0
        padded = []
        for c, cv in cells:
            padded.append(c + " " * (cell_w - len(cv)))
        body_vis = f"TPC {tpc_id}: [ {padded[0]} | {padded[1]} ]"
        body_lines.append(body_vis)
    inner_w = max(vislen(l) for l in body_lines) if body_lines else 20
    width = inner_w + 4

    label = f"GPC {gpc_id} [{sm_count} SMs]"
    head = f"+-- {label} "
    top = head + "-" * (width - len(head) - 1) + "+"
    bot = "+" + "-" * (width - 2) + "+"

    out = [top]
    for line in body_lines:
        pad = inner_w - vislen(line)
        out.append("| " + line + " " * pad + " |")
    out.append(bot)
    return "\n".join(out)


def render_topology_header(gpc_to_smids, gpc_tpc_layout_, total_smids):
    n_gpc = len(gpc_to_smids)
    out = [f"=== GPU topology ({total_smids} SMs, {n_gpc} GPCs) ==="]
    for gpc, tpcs in sorted(gpc_tpc_layout_.items()):
        sm_count = sum(len(s) for _, s in tpcs)
        parts = []
        for tid, smids in tpcs:
            tag = f"TPC{tid}" if tid >= 0 else "TPC?"
            parts.append(f"{tag}={{{','.join(str(s) for s in smids)}}}")
        out.append(f"GPC {gpc} [{sm_count} SMs]: " + "  ".join(parts))
    return "\n".join(out)


def render_shape(shape, gpc_tpc_layout_, style, color, total_smids):
    occ = shape["occ"]
    occ_rows = max((len(v) for v in occ.values()), default=1)
    occ_rows = max(occ_rows, 1)
    active_sms = sum(1 for v in occ.values() if v)

    out = []
    out.append("")
    out.append(
        f"=== Cluster shape {shape['name']} "
        f"(size={shape['size']}, {shape['nclusters']} clusters, {shape['blocks']} blocks) ==="
    )
    out.append(
        f"--- active SMs (touched by >=1 block) = {active_sms} / {total_smids}; "
        f"max occupants per SM in this launch = {occ_rows} ---"
    )
    out.append("")
    if style == "boxed":
        for gpc, tpcs in sorted(gpc_tpc_layout_.items()):
            out.append(render_gpc_boxed(gpc, tpcs, occ, color, occ_rows))
            out.append("")
    else:
        for gpc, tpcs in sorted(gpc_tpc_layout_.items()):
            out.append(render_gpc_compact(gpc, tpcs, occ, color))
            out.append("")
    return "\n".join(out)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--input", "-i", help="Read from file instead of stdin")
    ap.add_argument("--shape", "-s", help="Render only this cluster shape (e.g., 2x2x1)")
    ap.add_argument("--style", choices=("boxed", "compact"), default="boxed")
    ap.add_argument("--color", action="store_true", help="ANSI color cluster ids")
    args = ap.parse_args()

    text = open(args.input).read() if args.input else sys.stdin.read()
    gpc_to_smids, shapes = parse(text)
    if not gpc_to_smids:
        print("error: no GPC histogram lines found in input", file=sys.stderr)
        sys.exit(1)
    if not shapes:
        print("error: no cluster shape sections found in input", file=sys.stderr)
        sys.exit(1)

    smid_to_tpc = build_smid_to_tpc(shapes)
    layout = gpc_tpc_layout(gpc_to_smids, smid_to_tpc)
    total_smids = sum(len(v) for v in gpc_to_smids.values())

    print(render_topology_header(gpc_to_smids, layout, total_smids))

    target_shapes = (
        [s for s in shapes if s["name"] == args.shape] if args.shape else shapes
    )
    if args.shape and not target_shapes:
        print(f"error: cluster shape '{args.shape}' not found in input", file=sys.stderr)
        sys.exit(1)
    for s in target_shapes:
        print(render_shape(s, layout, args.style, args.color, total_smids))


if __name__ == "__main__":
    main()
