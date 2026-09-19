#!/usr/bin/env python3
"""Render one problems/*.json checkpoint as a standalone, offline HTML viewer.

Usage: python tools/visualize_problem.py runs/omega_small/problems/ID.json -o tree.html
No third-party packages, web server, CDN or network connection are required.
"""
import argparse
import html
from collections import defaultdict, deque
import json
import math
from pathlib import Path
import sys


def prepare_graph(data):
    if not isinstance(data, dict) or not isinstance(data.get('states'), list) or not isinstance(data.get('edges'), list):
        raise ValueError('Expected a v2 problem checkpoint with states and edges arrays, not samples.jsonl or screening JSON.')
    nodes = []
    by_id = {}
    for raw in data['states']:
        if not isinstance(raw, dict) or isinstance(raw.get('id'), bool) or not isinstance(raw.get('id'), (str, int)):
            raise ValueError('Every state needs a string/integer id.')
        key = str(raw['id'])
        if key in by_id:
            raise ValueError('Duplicate state id: ' + key)
        prefix = raw.get('prefix')
        if not isinstance(prefix, str):
            raise ValueError('Every state needs a text prefix.')
        mc = raw.get('mc')
        if mc is not None and (isinstance(mc, bool) or not isinstance(mc, (int, float)) or not math.isfinite(mc) or not 0 <= mc <= 1):
            raise ValueError('MC must be null or a finite number between 0 and 1.')
        if not isinstance(raw.get('rollouts', []), list):
            raise ValueError('State rollouts must be an array.')
        node = dict(raw, key=key, mc=mc)
        nodes.append(node)
        by_id[key] = node
    children, parents = defaultdict(list), defaultdict(list)
    edges = []
    seen = set()
    for raw in data['edges']:
        if not isinstance(raw, dict):
            raise ValueError('Every edge must be an object.')
        source, target = str(raw.get('parent_id')), str(raw.get('child_id'))
        if source not in by_id or target not in by_id:
            raise ValueError('Edge refers to a missing state.')
        if (source, target) in seen:
            raise ValueError('Duplicate parent-child edge.')
        seen.add((source, target))
        action = raw.get('action')
        if not isinstance(action, str) or not action or by_id[source]['prefix'] + action != by_id[target]['prefix']:
            raise ValueError('Edge action must append nonempty text to reconstruct the child prefix.')
        edge = dict(raw, source=source, target=target, key=str(len(edges)))
        edges.append(edge)
        children[source].append(target)
        parents[target].append(source)
    # Longest-path layers support multiple parents; IDs need not be contiguous.
    indegree = {key: len(parents[key]) for key in by_id}
    queue = deque(key for key in by_id if not indegree[key])
    depth = {key: 0 for key in by_id}
    order = []
    while queue:
        key = queue.popleft()
        order.append(key)
        for child in children[key]:
            depth[child] = max(depth[child], depth[key] + 1)
            indegree[child] -= 1
            if indegree[child] == 0:
                queue.append(child)
    if len(order) != len(nodes):
        raise ValueError('The state graph contains a cycle.')
    active = {key for key, n in by_id.items() if parents[key] or children[key] or n.get('in_tree') or n['prefix'] == ''}
    # Show isolated sampling probes separately, rather than inventing connections.
    layers = defaultdict(list)
    for key in order:
        if key in active:
            layers[depth[key]].append(key)
    rank = {}
    for level in sorted(layers):
        layers[level].sort(key=lambda key: sum(rank.get(p, 0) for p in parents[key]) / max(1, len(parents[key])))
        for index, key in enumerate(layers[level]):
            rank[key] = index
    widest = max([len(v) for v in layers.values()] + [1])
    width = max(720, widest * 240 + 80)
    for level, keys in layers.items():
        for index, key in enumerate(keys):
            by_id[key].update(x=width/2 + (index-(len(keys)-1)/2)*240 - 96, y=40 + level*150, probe=False)
    tree_height = max(200, (max(layers, default=0) + 1)*150 + 40)
    probes = [n for n in nodes if n['key'] not in active]
    columns = max(1, int((width-80)//240))
    for index, node in enumerate(probes):
        node.update(x=40+(index % columns)*240, y=tree_height+80+(index//columns)*125, probe=True)
    full_height = tree_height + (130 + math.ceil(len(probes)/columns)*125 if probes else 0)
    return {'problem_id': str(data.get('problem_id', 'unknown')), 'question': data.get('question', ''),
            'gold_answer': data.get('gold_answer', ''), 'status': data.get('status', ''),
            'nodes': nodes, 'edges': edges, 'samples': data.get('samples', []), 'events': data.get('events', []),
            'model_calls': data.get('model_calls'), 'searches': data.get('searches'),
            'width': width, 'tree_height': tree_height, 'full_height': full_height,
            'probe_count': len(probes)}


def render_html(data):
    graph = prepare_graph(data)
    # Prevent model-generated text from terminating the JSON script element.
    payload = json.dumps(graph, ensure_ascii=True, allow_nan=False).replace('<', '\\u003c').replace('>', '\\u003e').replace('&', '\\u0026')
    return HTML.replace('__QUESTION_HTML__', html.escape(str(graph['question'] or 'Question text is missing from this checkpoint.'))).replace('__GRAPH_JSON__', payload)


HTML = r'''<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>OmegaPRM · Search graph</title>
<style>
:root{font-family:Inter,system-ui,-apple-system,sans-serif;color:#1e293b;background:#f4f7fb}*{box-sizing:border-box}body{margin:0}header{padding:20px 26px;background:#fff;border-bottom:1px solid #dbe3ed}h1{font-size:23px;margin:0 0 5px}#question-heading{margin:16px 0 6px;font-size:13px;color:#64748b}#question-text{white-space:pre-wrap;overflow-wrap:anywhere;font-size:15px;line-height:1.6;max-height:25vh;overflow:auto}header p{margin:0;color:#64748b;font-size:13px}.toolbar{display:flex;flex-wrap:wrap;align-items:center;gap:9px;padding:12px 24px;background:#fff;border-bottom:1px solid #dbe3ed}button,input{font:inherit}button{border:1px solid #cbd5e1;border-radius:7px;background:white;padding:7px 11px;cursor:pointer}button:hover{background:#eef2ff}button:focus-visible,input:focus-visible{outline:3px solid #818cf8}.toolbar label{font-size:13px}.toolbar input[type=text]{width:170px;padding:7px;border:1px solid #cbd5e1;border-radius:7px}.layout{display:grid;grid-template-columns:minmax(0,1fr) 370px;height:65vh;min-height:480px}#canvas{width:100%;height:100%;touch-action:none;cursor:grab;background:#f7f9fc}#canvas.dragging{cursor:grabbing}.node{cursor:pointer}.node:focus{outline:none}.node:focus rect{stroke:#4f46e5;stroke-width:4}.edge{cursor:pointer}.selected rect{stroke:#4f46e5!important;stroke-width:4!important}.dim{opacity:.2}.highlight path{stroke:#4f46e5!important;stroke-width:3!important}aside{background:white;border-left:1px solid #dbe3ed;overflow:auto;padding:20px}h2{font-size:17px;margin:0 0 12px}h3{font-size:13px;color:#64748b;text-transform:uppercase;letter-spacing:.06em;margin:22px 0 8px}.muted{color:#64748b;font-size:13px;line-height:1.6}pre{font:12px/1.6 ui-monospace,SFMono-Regular,monospace;white-space:pre-wrap;overflow-wrap:anywhere;background:#f8fafc;padding:12px;border:1px solid #e2e8f0;border-radius:8px;max-height:400px;overflow:auto}dl{display:grid;grid-template-columns:1fr 1fr;font-size:13px;gap:9px;margin:12px 0}dt{color:#64748b}dd{margin:0;overflow-wrap:anywhere}details{border:1px solid #e2e8f0;border-radius:6px;margin:7px 0;padding:9px;font-size:13px}summary{cursor:pointer}.legend{display:flex;gap:15px;flex-wrap:wrap;padding:9px 24px;font-size:12px;background:white;border-bottom:1px solid #dbe3ed}.dot{display:inline-block;width:9px;height:9px;border-radius:50%;margin-right:5px}.footer{font-size:11px;color:#64748b;margin-top:20px}@media(max-width:800px){.layout{grid-template-columns:1fr;height:auto}#canvas{height:65vh}aside{border-left:0;border-top:1px solid #dbe3ed;max-height:70vh}.toolbar{padding:10px}.legend{padding:9px}}
</style></head><body>
<header><h1>OmegaPRM · Search graph</h1><p id="subtitle"></p><section aria-labelledby="question-heading"><h2 id="question-heading">Question</h2><div id="question-text">__QUESTION_HTML__</div></section></header>
<div class="toolbar"><button id="fit">Fit graph</button><button id="zoomin" aria-label="Zoom in">＋</button><button id="zoomout" aria-label="Zoom out">−</button><button id="overview">Overview</button><label><input id="probes" type="checkbox"> Show disconnected probes</label><label><input id="labels" type="checkbox" checked> Edge labels</label><input id="lookup" type="text" placeholder="Find state ID" aria-label="Find state ID"><button id="find">Find</button><button id="download">Export SVG</button><span id="notice" class="muted" role="status"></span></div>
<div class="legend"><span><i class="dot" style="background:#16a34a"></i>0 &lt; MC &lt; 1</span><span><i class="dot" style="background:#0284c7"></i>MC = 1</span><span><i class="dot" style="background:#dc2626"></i>MC = 0</span><span>Solid edge: exported sample · Dashed edge: multi-step/not exported</span></div>
<main class="layout"><svg id="canvas" xmlns="http://www.w3.org/2000/svg" aria-label="Interactive search graph"><defs><marker id="arrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse"><path d="M 0 0 L 10 5 L 0 10 z" fill="#94a3b8"/></marker></defs><g id="viewport"></g></svg><aside id="details"></aside></main>
<script id="graph-data" type="application/json">__GRAPH_JSON__</script>
<script>
'use strict';
const G=JSON.parse(document.getElementById('graph-data').textContent);
const $=id=>document.getElementById(id),NS='http://www.w3.org/2000/svg',svg=$('canvas'),vp=$('viewport');
const nodes=new Map(G.nodes.map(n=>[n.key,n])),nodeEls=new Map(),edgeEls=new Map();
const exported=new Set(G.samples.map(s=>String(s.parent_id)+'\u0000'+String(s.child_id)));
let scale=1,tx=0,ty=0,drag=null,moved=false;
function sv(tag,attrs={},text){const e=document.createElementNS(NS,tag);for(const[k,v]of Object.entries(attrs))e.setAttribute(k,v);if(text!==undefined)e.textContent=text;return e;}
function el(tag,text,parent=$('details')){const e=document.createElement(tag);if(text!==undefined)e.textContent=String(text);parent.appendChild(e);return e;}
function block(title,text){el('h3',title);el('pre',text??'');}
function stats(obj){const d=el('dl');for(const[k,v]of Object.entries(obj)){el('dt',k,d);el('dd',v??'—',d);}}
function clear(){ $('details').replaceChildren();nodeEls.forEach(e=>e.classList.remove('selected','dim'));edgeEls.forEach(e=>e.classList.remove('highlight','dim'));}
function overview(){clear();el('h2','Problem '+G.problem_id);stats({'States':G.nodes.length,'Saved edges':G.edges.length,'Exported samples':G.samples.length,'Disconnected probes':G.probe_count,'Search iterations':G.searches,'Generation calls':G.model_calls,'Status':G.status});block('Question',G.question);block('Gold answer',G.gold_answer);el('p','Click a node to inspect its full prefix and sampled continuations. Click an edge to inspect the added action. Drag the canvas to pan; scroll to zoom.',el('div')).className='muted';el('p','MC is a sampled success signal, not a proof of logical correctness. Only saved edges are drawn: complete rollouts are available in the node inspector, not expanded into invented paths. Disconnected probes are shown separately when enabled.').className='footer';}
function inspectNode(n){clear();nodeEls.get(n.key).classList.add('selected');edgeEls.forEach((e,k)=>{const a=G.edges[Number(k)];e.classList.toggle('highlight',a.source===n.key||a.target===n.key);});el('h2','State '+n.key);stats({'MC':n.mc===null?'Unknown':n.mc.toFixed(3),'Visits':n.visits??0,'Value source':n.mc_source??'—','Rollouts':(n.rollouts||[]).length,'In saved tree':n.in_tree?'Yes':'No','Sampling seed':n.sampling_seed,'Incoming edges':G.edges.filter(e=>e.target===n.key).length,'Outgoing edges':G.edges.filter(e=>e.source===n.key).length});if(n.probe)el('p','Disconnected probe: no saved edge establishes its parent.').className='muted';block('Full reasoning prefix',n.prefix||'(empty root prefix)');el('h3','Sampled continuations');if(!(n.rollouts||[]).length)el('p',n.mc_source==='terminal_answer'?'Terminal answer value; no additional continuations were sampled.':'No continuations stored.').className='muted';(n.rollouts||[]).forEach((r,i)=>{const d=el('details');el('summary','#'+(i+1)+' · '+(r.correct===true?'Correct':r.correct===false?'Incorrect':'Unknown')+' · '+r.finish_reason,d);el('p','Reason: '+r.reason+' | tokens: '+r.token_count+' | visited: '+Boolean(r.visited)+(r.unknown_mapped_to_incorrect?' | unknown mapped to incorrect':''),d);el('pre',r.text,d);});}
function inspectEdge(a){clear();edgeEls.get(a.key).classList.add('highlight');el('h2','Edge '+a.source+' → '+a.target);stats({'Action tokens':a.action_tokens,'Single-step':a.single_step?'Yes':'No','Exported sample':exported.has(a.source+'\u0000'+a.target)?'Yes':'No','Child MC':nodes.get(a.target).mc});block('Action / added text',a.action);block('Parent prefix',nodes.get(a.source).prefix);block('Child prefix',nodes.get(a.target).prefix);}
function transform(){vp.setAttribute('transform',`translate(${tx} ${ty}) scale(${scale})`);}
function fit(){const box=svg.getBoundingClientRect(),h=$('probes').checked?G.full_height:G.tree_height;scale=Math.min(box.width/G.width,box.height/h)*.9;scale=Math.max(.005,scale);tx=(box.width-G.width*scale)/2;ty=(box.height-h*scale)/2;transform();}
function zoom(f,x=svg.clientWidth/2,y=svg.clientHeight/2){const ns=Math.min(5,Math.max(.005,scale*f)),ratio=ns/scale;tx=x-(x-tx)*ratio;ty=y-(y-ty)*ratio;scale=ns;transform();}
for(const a of G.edges){const p=nodes.get(a.source),c=nodes.get(a.target),x1=p.x+96,y1=p.y+82,x2=c.x+96,y2=c.y;const d=`M${x1},${y1} C${x1},${(y1+y2)/2} ${x2},${(y1+y2)/2} ${x2},${y2}`;const group=sv('g',{'class':'edge','tabindex':0,'role':'button','aria-label':`Edge ${a.source} to ${a.target}`});group.appendChild(sv('path',{d,fill:'none',stroke:'transparent','stroke-width':16}));group.appendChild(sv('path',{d,fill:'none',stroke:'#94a3b8','stroke-width':1.7,'stroke-dasharray':exported.has(a.source+'\u0000'+a.target)?'none':'6 4','marker-end':'url(#arrow)'}));group.appendChild(sv('text',{x:(x1+x2)/2+8,y:(y1+y2)/2,'font-size':11,fill:'#64748b','font-family':'system-ui','class':'edge-label'},a.action_tokens+' tokens'));group.addEventListener('click',e=>{e.stopPropagation();if(!moved)inspectEdge(a);});group.addEventListener('keydown',e=>{if(e.key==='Enter'){inspectEdge(a);}});vp.appendChild(group);edgeEls.set(a.key,group);}
const probeTitle=sv('text',{x:40,y:G.tree_height+40,'font-size':18,'font-family':'system-ui',fill:'#64748b'},'Disconnected sampling probes — no inferred edges');vp.appendChild(probeTitle);
for(const n of G.nodes){const color=n.mc===null?'#64748b':n.mc===0?'#dc2626':n.mc===1?'#0284c7':'#16a34a',fill=n.mc===null?'#f1f5f9':n.mc===0?'#fff1f2':n.mc===1?'#eff6ff':'#f0fdf4';const group=sv('g',{'class':'node','transform':`translate(${n.x},${n.y})`,'tabindex':0,'role':'button','aria-label':`State ${n.key}, MC ${n.mc}`});group.appendChild(sv('rect',{width:192,height:82,rx:10,fill,stroke:color,'stroke-width':1.5}));group.appendChild(sv('text',{x:13,y:23,'font-size':14,'font-family':'system-ui','font-weight':600,fill:'#1e293b'},'State '+n.key+(n.prefix===''?' · ROOT':'')));group.appendChild(sv('text',{x:13,y:44,'font-size':12,'font-family':'system-ui',fill:color},'MC '+(n.mc===null?'?':n.mc.toFixed(3))+' · visits '+(n.visits??0)));const preview=n.prefix===''?'(empty prefix)':n.prefix.replace(/\s+/g,' ').slice(-25);group.appendChild(sv('text',{x:13,y:65,'font-size':11,'font-family':'monospace',fill:'#64748b'},preview));group.addEventListener('click',e=>{e.stopPropagation();if(!moved)inspectNode(n);});group.addEventListener('keydown',e=>{if(e.key==='Enter'){inspectNode(n);}});vp.appendChild(group);nodeEls.set(n.key,group);}
function probes(){nodeEls.forEach((e,k)=>{e.style.display=nodes.get(k).probe&&!$('probes').checked?'none':'';});probeTitle.style.display=$('probes').checked&&G.probe_count?'':'none';fit();}
$('subtitle').textContent='Problem '+G.problem_id+' · '+G.nodes.length+' states · '+G.edges.length+' saved edges · '+G.samples.length+' samples';
$('fit').onclick=fit;$('zoomin').onclick=()=>zoom(1.25);$('zoomout').onclick=()=>zoom(.8);$('overview').onclick=overview;$('probes').onchange=probes;$('labels').onchange=()=>vp.querySelectorAll('.edge-label').forEach(e=>e.style.display=$('labels').checked?'':'none');
function find(){const key=$('lookup').value.trim(),n=nodes.get(key);if(!n){$('notice').textContent='State not found';return;}$('notice').textContent='';if(n.probe){$('probes').checked=true;probes();}scale=1;tx=svg.clientWidth/2-(n.x+96);ty=svg.clientHeight/2-(n.y+41);transform();inspectNode(n);}$('find').onclick=find;$('lookup').onkeydown=e=>{if(e.key==='Enter')find();};
svg.addEventListener('wheel',e=>{e.preventDefault();const b=svg.getBoundingClientRect();zoom(e.deltaY<0?1.12:1/1.12,e.clientX-b.left,e.clientY-b.top);},{passive:false});
svg.addEventListener('pointerdown',e=>{if(e.button!==0)return;drag={x:e.clientX,y:e.clientY,tx,ty};moved=false;svg.classList.add('dragging');});
window.addEventListener('pointermove',e=>{if(!drag)return;const dx=e.clientX-drag.x,dy=e.clientY-drag.y;if(Math.abs(dx)+Math.abs(dy)>4)moved=true;if(moved){tx=drag.tx+dx;ty=drag.ty+dy;transform();}});
window.addEventListener('pointerup',()=>{drag=null;svg.classList.remove('dragging');setTimeout(()=>moved=false,0);});window.addEventListener('pointercancel',()=>{drag=null;svg.classList.remove('dragging');});
$('download').onclick=()=>{const clone=svg.cloneNode(true),h=$('probes').checked?G.full_height:G.tree_height;clone.setAttribute('viewBox',`0 0 ${G.width} ${h}`);clone.setAttribute('width',G.width);clone.setAttribute('height',h);clone.querySelector('#viewport').removeAttribute('transform');const blob=new Blob([new XMLSerializer().serializeToString(clone)],{type:'image/svg+xml'});const url=URL.createObjectURL(blob),a=document.createElement('a');a.href=url;a.download='omegaprm-graph.svg';a.click();setTimeout(()=>URL.revokeObjectURL(url),1000);};
window.addEventListener('resize',fit);overview();probes();
</script></body></html>'''


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('problem', type=Path, help='one v2 problems/<hash>.json checkpoint')
    parser.add_argument('-o', '--output', type=Path, help='output HTML (default: input basename with .html)')
    parser.add_argument('--force', action='store_true', help='replace an existing HTML output')
    args = parser.parse_args(argv)
    output = args.output or args.problem.with_suffix('.html')
    if output.resolve() == args.problem.resolve():
        parser.error('output must not overwrite the input checkpoint')
    if output.suffix.lower() != '.html':
        parser.error('output filename must end in .html')
    if output.exists() and not args.force:
        parser.error('output exists; use --force to replace it')
    try:
        html = render_html(json.loads(args.problem.read_text(encoding='utf-8')))
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(html, encoding='utf-8')
    except (OSError, ValueError) as exc:
        parser.exit(1, f'Error: {exc}\n')
    print(f'Open in your browser: {output.resolve()}')


if __name__ == '__main__':
    main()
