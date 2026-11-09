/* Ink UI kit v1 · shell + figures from vix HQ. Plain JS, no dependencies. Exposes window.Ink:
     Ink.shell(opts)              wire the header tabs, side rail, pages, "On this page", search (⌘K / Ctrl K) and theme
     Ink.icon(name)               an inline line icon (24px grid, stroke) - see Ink.ICONS for names
     Ink.line(el, data, opts)     line chart: data = {x:[...labels], series:[{name, values:[...]}]}
     Ink.bar(el, data, opts)      column chart: data = {x:[...], series:[{name, values:[...]}]} (first series)
     Ink.radar(el, data, opts)    radar: data = {spokes:[{key,label,group}], series:[{name, values:{key:0..100}, shown:{key:text}}]}
   Figures follow the Inkling style: lead series heavy cobalt with filled white-ringed dots, others thinner with their own
   dash and hollow markers, light grid in a thin frame, centred legend. Colours come from --c1..--c8 in ink.css. */
(function(){
const E=s=>String(s??'').replace(/[&<>"]/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;'}[c]));
const css=v=>getComputedStyle(document.documentElement).getPropertyValue(v).trim();
const PAL=()=>[1,2,3,4,5,6,7,8].map(i=>css('--c'+i)||'#2f5fd0');
const DASH=['','7 4','2 4','9 4 2 4','4 3','1 3','6 2','3 6'];
const ls={get:(k,d)=>{try{const v=localStorage.getItem(k);return v==null?d:JSON.parse(v)}catch(_){return d}},set:(k,v)=>{try{localStorage.setItem(k,JSON.stringify(v))}catch(_){}}};

const ICONS={overview:'<path d="M3 10.5 12 3l9 7.5"/><path d="M5 9.5V21h14V9.5"/><path d="M10 21v-6h4v6"/>',
 decisions:'<rect x="3" y="3" width="18" height="18" rx="3"/><path d="m8 12 3 3 5-6"/>',
 reports:'<path d="M14 3H6a2 2 0 0 0-2 2v14a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V9z"/><path d="M14 3v6h6M8 13h8M8 17h5"/>',
 people:'<circle cx="9" cy="8" r="3"/><path d="M3.5 20c.8-3.2 3-5 5.5-5s4.7 1.8 5.5 5"/><circle cx="17" cy="9" r="2.4"/><path d="M16.5 14.2c2.2.3 3.6 2 4 4.8"/>',
 schedules:'<rect x="3" y="5" width="18" height="16" rx="2"/><path d="M3 10h18M8 3v4M16 3v4M12 14v3l2 1"/>',
 finance:'<path d="M3 21h18M5 21V10M10 21V10M14 21V10M19 21V10M2.5 10 12 4l9.5 6z"/>',
 growth:'<path d="m3 17 6-6 4 4 8-8"/><path d="M15 7h6v6"/>',chart:'<path d="M4 20V10M10 20V4M16 20v-7M22 20H2"/>',
 trend:'<path d="M3 3v18h18"/><path d="m7 15 4-4 3 3 6-7"/>',world:'<circle cx="12" cy="12" r="9"/><path d="M3 12h18M12 3c2.5 2.5 3.8 5.5 3.8 9s-1.3 6.5-3.8 9c-2.5-2.5-3.8-5.5-3.8-9S9.5 5.5 12 3z"/>',
 network:'<rect x="9" y="2.5" width="6" height="5" rx="1"/><rect x="2.5" y="16.5" width="6" height="5" rx="1"/><rect x="15.5" y="16.5" width="6" height="5" rx="1"/><path d="M12 7.5V12M5.5 16.5V12h13v4.5"/>',
 shield:'<path d="M12 3 4.5 6v5.5c0 4.5 3.1 8 7.5 9.5 4.4-1.5 7.5-5 7.5-9.5V6z"/><path d="m9 12 2 2 4-4"/>',
 idea:'<path d="M9 18h6M10 21h4"/><path d="M12 3a6 6 0 0 0-3.6 10.8c.7.6 1.1 1.3 1.1 2.2h5c0-.9.4-1.6 1.1-2.2A6 6 0 0 0 12 3z"/>',
 grid:'<rect x="3" y="3" width="7" height="9" rx="1.2"/><rect x="14" y="3" width="7" height="5" rx="1.2"/><rect x="14" y="12" width="7" height="9" rx="1.2"/><rect x="3" y="16" width="7" height="5" rx="1.2"/>',
 plug:'<path d="M9 2v5M15 2v5"/><path d="M6 7h12v4a6 6 0 0 1-12 0z"/><path d="M12 17v5"/>',inbox:'<path d="M3 13h5l1.5 3h5L16 13h5"/><path d="M5.5 5h13L21 13v6a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-6z"/>',
 edit:'<path d="M12 20h9"/><path d="M16.5 3.5a2.1 2.1 0 0 1 3 3L7 19l-4 1 1-4z"/>',graph:'<circle cx="6" cy="6" r="2.5"/><circle cx="18" cy="6" r="2.5"/><circle cx="12" cy="18" r="2.5"/><path d="M8.3 7.2 10.8 16M15.7 7.2 13.2 16M8.5 6h7"/>',
 lab:'<path d="M9 3h6M10 3v6L4.5 18.5A1.7 1.7 0 0 0 6 21h12a1.7 1.7 0 0 0 1.5-2.5L14 9V3"/><path d="M7 15h10"/>',
 settings:'<path d="M4 6h10M18 6h2M4 12h4M12 12h8M4 18h12"/><circle cx="16" cy="6" r="2"/><circle cx="10" cy="12" r="2"/><circle cx="18" cy="18" r="2"/>',
 radar:'<path d="M12 3 20.5 9.2 17.3 19H6.7L3.5 9.2z"/><path d="M12 7.5 16 10.6 14.5 15.5h-5L8 10.6z"/>',layers:'<path d="m12 3 9 5-9 5-9-5z"/><path d="m3 13 9 5 9-5"/>',
 brain:'<path d="M9 4a3 3 0 0 0-3 3 3 3 0 0 0-2 5 3 3 0 0 0 2 5 3 3 0 0 0 6 1V5a2 2 0 0 0-3-1z"/><path d="M15 4a3 3 0 0 1 3 3 3 3 0 0 1 2 5 3 3 0 0 1-2 5 3 3 0 0 1-6 1"/>',
 play:'<path d="M7 4.5v15l12-7.5z"/>',search:'<circle cx="11" cy="11" r="7"/><path d="m20 20-3.5-3.5"/>',sun:'<circle cx="12" cy="12" r="4"/><path d="M12 2v2M12 20v2M4.9 4.9l1.4 1.4M17.7 17.7l1.4 1.4M2 12h2M20 12h2M4.9 19.1l1.4-1.4M17.7 6.3l1.4-1.4"/>',
 moon:'<path d="M20 14.5A8 8 0 1 1 9.5 4 6.5 6.5 0 0 0 20 14.5z"/>',dot:'<circle cx="12" cy="12" r="3"/>'};
const icon=(n,cls='ink-i')=>`<svg class="${cls}" viewBox="0 0 24 24" aria-hidden="true">${ICONS[n]||ICONS.dot}</svg>`;

// ------------------------------------------------------------ shell
function shell(o={}){
  const KEY=(o.storageKey||'ink')+'.page';
  const nav=document.querySelector('.ink-nav'),main=document.querySelector('.ink-main'),toc=document.querySelector('.ink-toc'),
        hdr=document.querySelector('.ink-header'),sh=document.querySelector('.ink-shell');
  if(!nav||!main)throw new Error('Ink.shell needs .ink-nav and .ink-main');
  const items=[...nav.querySelectorAll('[data-page]')];
  items.forEach(b=>{if(!b.querySelector('.ink-i'))b.insertAdjacentHTML('afterbegin',icon(b.dataset.icon||'dot'))});
  // section tabs from the rail's .grp headings
  const groups=[];let cur=null;[...nav.children].forEach(el=>{if(el.classList.contains('grp')){cur={name:el.textContent.trim(),el,items:[]};groups.push(cur)}
    else if(el.dataset.page){if(!cur){cur={name:o.firstGroup||'Home',el:null,items:[]};groups.push(cur)}cur.items.push(el);el.dataset.grp=groups.length-1}});
  let tabs=hdr&&hdr.querySelector('.ink-tabs');
  if(hdr&&groups.length>1&&!tabs){tabs=document.createElement('nav');tabs.className='ink-tabs';tabs.setAttribute('role','tablist');hdr.appendChild(tabs)}
  if(tabs)tabs.innerHTML=groups.map((g,i)=>`<button role="tab" data-g="${i}">${E(g.name)}</button>`).join('');
  const setHdr=()=>{if(hdr)document.documentElement.style.setProperty('--hdr',hdr.offsetHeight+'px')};setHdr();addEventListener('resize',setHdr);
  // theme
  const tb=document.querySelector('[data-ink=theme]');
  const applyTheme=t=>{if(t)document.documentElement.dataset.theme=t;else delete document.documentElement.dataset.theme;
    if(tb){const dark=t?t==='dark':matchMedia('(prefers-color-scheme: dark)').matches;tb.innerHTML=icon(dark?'sun':'moon')+(dark?' Light':' Dark')}};
  applyTheme(ls.get((o.storageKey||'ink')+'.theme',null));
  if(tb)tb.onclick=()=>{const dark=document.documentElement.dataset.theme?document.documentElement.dataset.theme==='dark':matchMedia('(prefers-color-scheme: dark)').matches;
    const t=dark?'light':'dark';ls.set((o.storageKey||'ink')+'.theme',t);applyTheme(t);if(o.onTheme)o.onTheme(t);redrawFigures()};
  // pages
  function show(page,push=true){
    const b=items.find(x=>x.dataset.page===page)||items[0];if(!b)return;page=b.dataset.page;
    items.forEach(x=>x.classList.toggle('on',x===b));
    const g=+b.dataset.grp;groups.forEach((G,i)=>{G.items.forEach(x=>x.classList.toggle('hide',i!==g));if(G.el)G.el.classList.toggle('cur',i===g)});
    if(tabs)tabs.querySelectorAll('button').forEach(x=>x.setAttribute('aria-selected',+x.dataset.g===g));
    main.querySelectorAll('section[data-page]').forEach(s=>s.hidden=s.dataset.page!==page);
    ls.set(KEY,page);if(push&&location.hash!=='#'+page)history.replaceState(null,'','#'+page);
    if(o.onShow)o.onShow(page);buildToc();scrollTo({top:0})}
  items.forEach(b=>b.addEventListener('click',e=>{e.preventDefault();show(b.dataset.page)}));
  if(tabs)tabs.querySelectorAll('button').forEach(t=>t.onclick=()=>{const G=groups[+t.dataset.g];if(G.items[0])show(G.items[0].dataset.page)});
  // on this page
  function buildToc(){if(!toc)return;const s=main.querySelector('section[data-page]:not([hidden])')||main;
    const hs=[...s.querySelectorAll('h2,h3')].filter(h=>h.offsetParent!==null&&!h.classList.contains('title'));
    hs.forEach((h,i)=>{if(!h.id)h.id='s-'+(s.dataset.page||'p')+'-'+i});
    if(sh)sh.classList.toggle('notoc',hs.length<2);
    toc.innerHTML=hs.length<2?'':`<b>On this page</b><ul>${hs.map(h=>`<li class="${h.tagName==='H3'?'sub':''}"><a href="#${h.id}">${E([...h.childNodes].filter(n=>n.nodeType===3).map(n=>n.textContent).join('').trim()||h.textContent)}</a></li>`).join('')}</ul>`;
    toc.querySelectorAll('a').forEach(a=>a.onclick=e=>{e.preventDefault();document.getElementById(a.getAttribute('href').slice(1)).scrollIntoView({behavior:'smooth'})})}
  addEventListener('scroll',()=>{if(!toc)return;const as=[...toc.querySelectorAll('a')];let on=null;as.forEach(a=>{const h=document.getElementById(a.getAttribute('href').slice(1));if(h&&h.getBoundingClientRect().top<innerHeight*.3)on=a});as.forEach(a=>a.classList.toggle('on',a===on))},{passive:true});
  // search: jump to a page (and your own results via o.search(q) -> [{title, sub, go()}])
  const sb=document.querySelector('.ink-search');
  if(sb){if(!sb.querySelector('input'))sb.innerHTML=`${icon('search','')}<input type="search" placeholder="${E(o.searchPlaceholder||'Search')}" aria-label="Search"><kbd>${/Mac/.test(navigator.platform)?'⌘':'Ctrl'} K</kbd><div class="ink-results"></div>`;
    const inp=sb.querySelector('input'),res=sb.querySelector('.ink-results');let hits=[],k=0;
    const draw=()=>{res.innerHTML=hits.map((h,i)=>`<div class="${i===k?'on':''}" data-i="${i}"><b>${E(h.title)}</b>${h.sub?`<small>${E(h.sub)}</small>`:''}</div>`).join('');res.querySelectorAll('[data-i]').forEach(d=>d.onmousedown=()=>{hits[+d.dataset.i].go();inp.value='';hits=[];draw()})};
    inp.oninput=async()=>{const q=inp.value.trim().toLowerCase();k=0;if(!q){hits=[];return draw()}
      hits=items.filter(b=>b.textContent.toLowerCase().includes(q)).slice(0,6).map(b=>({title:b.textContent.trim(),sub:groups[+b.dataset.grp].name,go:()=>show(b.dataset.page)}));
      if(o.search)try{hits=hits.concat(await o.search(q)||[])}catch(_){};draw()};
    inp.onkeydown=e=>{if(e.key==='ArrowDown'){k=Math.min(hits.length-1,k+1);draw();e.preventDefault()}else if(e.key==='ArrowUp'){k=Math.max(0,k-1);draw();e.preventDefault()}
      else if(e.key==='Enter'&&hits[k]){hits[k].go();inp.value='';hits=[];draw();inp.blur()}else if(e.key==='Escape'){inp.value='';hits=[];draw();inp.blur()}};
    inp.onblur=()=>setTimeout(()=>{hits=[];draw()},150);
    addEventListener('keydown',e=>{if((e.metaKey||e.ctrlKey)&&e.key.toLowerCase()==='k'){e.preventDefault();inp.focus()}})}
  addEventListener('hashchange',()=>{const h=location.hash.slice(1);if(items.some(b=>b.dataset.page===h))show(h,false)});
  show((location.hash||'').slice(1)||ls.get(KEY,null)||(items[0]&&items[0].dataset.page),false);
  return {show,refreshToc:buildToc}}

// ------------------------------------------------------------ figures
const FIGS=new Set();function redrawFigures(){FIGS.forEach(f=>{if(f.el.isConnected)f.draw();else FIGS.delete(f)})}
function register(el,draw){el.classList.add('ink-fig');const f={el,draw};FIGS.add(f);draw();
  if(window.ResizeObserver&&!el._inkRO){let w=el.clientWidth;el._inkRO=new ResizeObserver(()=>{if(Math.abs(el.clientWidth-w)>20){w=el.clientWidth;draw()}});el._inkRO.observe(el)}}
const fmtDef=v=>v==null?'–':Math.abs(v)>=1e9?(v/1e9).toFixed(1)+'B':Math.abs(v)>=1e6?(v/1e6).toFixed(1)+'M':Math.abs(v)>=1e4?(v/1e3).toFixed(0)+'k':+(+v).toFixed(2);
function nice(lo,hi,n=4){if(hi===lo){hi=lo+(Math.abs(lo)||1)}const span=hi-lo,raw=span/n,mag=Math.pow(10,Math.floor(Math.log10(raw))),r=raw/mag,step=(r<1.5?1:r<3?2:r<7?5:10)*mag;
  const a=Math.floor(lo/step)*step,b=Math.ceil(hi/step)*step,t=[];for(let x=a;x<=b+step/2;x+=step)t.push(+x.toPrecision(12));return {lo:a,hi:b,ticks:t}}
function legend(series,P,cb){return `<div class="ink-lg">${series.map((s,i)=>`<button data-li="${i}" class="${s.hidden?'off':''}"><svg viewBox="0 0 24 12" aria-hidden="true"><line x1="1" x2="23" y1="6" y2="6" stroke="${P[i%8]}" stroke-width="${i?2:3}"${i&&DASH[i%8]?` stroke-dasharray="${DASH[i%8]}"`:''}/><circle cx="12" cy="6" r="4" fill="${i?'var(--panel)':P[0]}" stroke="${i?P[i%8]:'var(--panel)'}" stroke-width="2"/></svg>${E(s.name)}${s.score!=null?` <b>${Math.round(s.score)}</b>`:''}</button>`).join('')}</div>`}
function wireLegend(el,series,draw){el.querySelectorAll('[data-li]').forEach(b=>b.onclick=()=>{const s=series[+b.dataset.li];s.hidden=!s.hidden;draw()})}

function line(el,data,o={}){const draw=()=>{const P=PAL(),W=Math.max(320,el.clientWidth||700),H=o.height||300,L=o.left||62,R=16,T=o.title?30:12,B=34,fmt=o.format||fmtDef;
  const xs=data.x||[],ser=data.series||[],vis=ser.filter(s=>!s.hidden),vals=vis.flatMap(s=>s.values).filter(v=>v!=null&&isFinite(v));
  if(!xs.length||!vals.length){el.innerHTML='<p class="note">No data.</p>';return}
  let lo=Math.min(...vals),hi=Math.max(...vals);if(o.zero)lo=Math.min(0,lo);const pad=(hi-lo)*.08||1;if(!o.zero)lo-=pad;hi+=pad;const N=nice(lo,hi);lo=N.lo;hi=N.hi;
  const px=i=>L+(W-L-R)*(xs.length<2?.5:i/(xs.length-1)),py=v=>T+(H-T-B)*(1-(v-lo)/(hi-lo));
  let g=o.title?`<text x="${L}" y="18" class="ttl">${E(o.title)}</text>`:'';
  g+=`<rect class="fr" x="${L}" y="${T}" width="${W-L-R}" height="${H-T-B}"/>`;
  N.ticks.forEach(t=>{const y=py(t);if(y<T-1||y>H-B+1)return;g+=`<line class="gl" x1="${L}" x2="${W-R}" y1="${y}" y2="${y}"/><text x="${L-8}" y="${y+4}" text-anchor="end">${E(fmt(t))}</text>`});
  const step=Math.max(1,Math.ceil(xs.length/Math.max(2,Math.floor((W-L)/86))));
  xs.forEach((x,i)=>{if(i%step===0||i===xs.length-1)g+=`<text x="${px(i)}" y="${H-10}" text-anchor="${xs.length>1&&i===0?'start':xs.length>1&&i===xs.length-1?'end':'middle'}">${E(x)}</text>`});
  ser.map((s,i)=>[s,i]).reverse().forEach(([s,i])=>{if(s.hidden)return;const c=P[i%8],lead=i===0;let d='';const dots=[];
    s.values.forEach((v,j)=>{if(v==null||!isFinite(v)){return}d+=(d?'L':'M')+px(j).toFixed(1)+','+py(v).toFixed(1);
      if(xs.length<=60)dots.push(`<circle cx="${px(j)}" cy="${py(v)}" r="${lead?4.6:4}" fill="${lead?c:'var(--panel)'}" stroke="${lead?'var(--panel)':c}" stroke-width="2"><title>${E(s.name)} · ${E(xs[j])}: ${E(fmt(v))}</title></circle>`)});
    g+=`<path d="${d}" fill="none" stroke="${c}" stroke-width="${lead?2.8:2}" stroke-linejoin="round" stroke-linecap="round"${!lead&&DASH[i%8]?` stroke-dasharray="${DASH[i%8]}"`:''}/>`+dots.join('')});
  el.innerHTML=`<svg viewBox="0 0 ${W} ${H}" role="img" aria-label="${E(o.title||'Line chart')}">${g}</svg>`+(ser.length>1?legend(ser,P):'');wireLegend(el,ser,draw)};
  register(el,draw)}

function bar(el,data,o={}){const draw=()=>{const P=PAL(),W=Math.max(320,el.clientWidth||700),H=o.height||300,L=o.left||62,R=16,T=o.title?30:12,B=40,fmt=o.format||fmtDef;
  const xs=data.x||[],s=(data.series||[])[0];if(!s||!xs.length){el.innerHTML='<p class="note">No data.</p>';return}
  const vs=s.values.map(v=>+v||0),N=nice(Math.min(0,...vs),Math.max(0,...vs)),bw=(W-L-R)/xs.length,py=v=>T+(H-T-B)*(1-(v-N.lo)/(N.hi-N.lo));
  let g=o.title?`<text x="${L}" y="18" class="ttl">${E(o.title)}</text>`:'';g+=`<rect class="fr" x="${L}" y="${T}" width="${W-L-R}" height="${H-T-B}"/>`;
  N.ticks.forEach(t=>{g+=`<line class="gl" x1="${L}" x2="${W-R}" y1="${py(t)}" y2="${py(t)}"/><text x="${L-8}" y="${py(t)+4}" text-anchor="end">${E(fmt(t))}</text>`});
  const every=Math.max(1,Math.ceil(xs.length/Math.floor((W-L)/70)));
  vs.forEach((v,i)=>{const x=L+i*bw+bw*.16,w=bw*.68,y0=py(0),y1=py(v);g+=`<rect x="${x}" y="${Math.min(y0,y1)}" width="${w}" height="${Math.max(1,Math.abs(y1-y0))}" rx="2" fill="${o.color||P[0]}"><title>${E(xs[i])}: ${E(fmt(v))}</title></rect>`;
    if(i%every===0)g+=`<text x="${x+w/2}" y="${H-B+18}" text-anchor="middle">${E(String(xs[i]).slice(0,Math.max(4,Math.floor(bw*every/7.4))))}</text>`});
  el.innerHTML=`<svg viewBox="0 0 ${W} ${H}" role="img" aria-label="${E(o.title||'Bar chart')}">${g}</svg>`};register(el,draw)}

function radar(el,data,o={}){const draw=()=>{const P=PAL(),narrow=(el.clientWidth||900)<640;
  const spokes=(data.spokes||[]).map(s=>narrow&&s.short?{...s,label:s.short,group:''}:narrow?{...s,group:''}:s),series=data.series||[],n=spokes.length;
  if(n<3||!series.length){el.innerHTML='<p class="note">A radar needs three or more spokes.</p>';return}
  const S=o.size||640,pad=o.pad||175,cx=S/2+pad,cy=S/2+56,R=S/2-24,W=S+2*pad,H=S+112;
  const ang=k=>-Math.PI/2+2*Math.PI*k/n,pt=(k,v)=>[cx+R*(v/100)*Math.cos(ang(k)),cy+R*(v/100)*Math.sin(ang(k))];
  let g='';[20,40,60,80,100].forEach(r=>{g+=`<polygon class="rg" points="${spokes.map((_,k)=>pt(k,r).map(x=>x.toFixed(1)).join(',')).join(' ')}"/>`});
  spokes.forEach((s,k)=>{const [x,y]=pt(k,100);g+=`<line class="rs" x1="${cx}" y1="${cy}" x2="${x.toFixed(1)}" y2="${y.toFixed(1)}"/>`;
    const [lx,ly]=pt(k,113),a=Math.cos(ang(k)),an=Math.abs(a)<.2?'middle':a>0?'start':'end',dy=Math.sin(ang(k))<-.5?-20:Math.sin(ang(k))>.5?6:-6;
    g+=`<g class="rl" data-k="${k}"><rect x="${(an==='start'?lx-6:an==='end'?lx-156:lx-80).toFixed(1)}" y="${(ly+dy-16).toFixed(1)}" width="160" height="42" fill="transparent"/><text x="${lx.toFixed(1)}" y="${(ly+dy).toFixed(1)}" text-anchor="${an}" class="rt">${E(s.label)}</text>${s.group?`<text x="${lx.toFixed(1)}" y="${(ly+dy+18).toFixed(1)}" text-anchor="${an}" class="rgp">${E(s.group)}</text>`:''}</g>`});
  series.map((s,i)=>i).reverse().forEach(i=>{const s=series[i];if(s.hidden)return;const c=P[i%8],lead=i===0,ps=spokes.map((sp,k)=>pt(k,s.values[sp.key]??0));
    g+=`<g class="rser" data-i="${i}"><polygon points="${ps.map(p=>p.map(x=>x.toFixed(1)).join(',')).join(' ')}" fill="${c}" fill-opacity="${lead?.10:.05}" stroke="${c}" stroke-width="${lead?3.2:2.2}" stroke-linejoin="round"${!lead&&DASH[i%8]?` stroke-dasharray="${DASH[i%8]}"`:''}/>`+
      ps.map((p,k)=>{const v=s.values[spokes[k].key];return `<circle cx="${p[0].toFixed(1)}" cy="${p[1].toFixed(1)}" r="${lead?6:5.2}" fill="${lead?c:'var(--panel)'}" stroke="${lead?'var(--panel)':c}" stroke-width="2.4"${v==null?' opacity=".35"':''}><title>${E(s.name)} · ${E(spokes[k].label)}: ${v==null?'no data':Math.round(v)}${s.shown&&s.shown[spokes[k].key]?' ('+E(s.shown[spokes[k].key])+')':''}</title></circle>`}).join('')+'</g>'});
  g+=`<line class="rhot" x1="${cx}" y1="${cy}" x2="${cx}" y2="${cy}" style="display:none"/>`;
  el.innerHTML=`<div class="ink-tip"></div><svg viewBox="0 0 ${W} ${H}" role="img" aria-label="${E(o.title||'Radar chart')}" style="max-width:900px;margin:0 auto">${g}</svg>`+legend(series,P);
  const svg=el.querySelector('svg'),tip=el.querySelector('.ink-tip'),hot=svg.querySelector('.rhot');
  svg.querySelectorAll('.rl').forEach(l=>{const k=+l.dataset.k;l.onmouseenter=()=>{const [x,y]=pt(k,100);hot.setAttribute('x2',x);hot.setAttribute('y2',y);hot.style.display='';l.classList.add('on');
      tip.innerHTML=`<b>${E(spokes[k].label)}</b>${spokes[k].how?`<div class="note">${E(spokes[k].how)}</div>`:''}<table>${series.map((s,i)=>s.hidden?'':`<tr><td><i style="background:${P[i%8]}"></i>${E(s.name)}</td><td class="n">${s.values[spokes[k].key]==null?'no data':Math.round(s.values[spokes[k].key])}</td><td class="note">${E(s.shown?.[spokes[k].key]||'')}</td></tr>`).join('')}</table>`;
      tip.style.display='block';const r=l.getBoundingClientRect(),h=el.getBoundingClientRect();tip.style.left=Math.max(4,Math.min(h.width-tip.offsetWidth-4,r.left-h.left+r.width/2-tip.offsetWidth/2))+'px';tip.style.top=(r.bottom-h.top+6)+'px'};
    l.onmouseleave=()=>{hot.style.display='none';l.classList.remove('on');tip.style.display='none'}});
  wireLegend(el,series,draw)};register(el,draw)}

window.Ink={shell,icon,ICONS,line,bar,radar,redraw:redrawFigures,esc:E};
})();
