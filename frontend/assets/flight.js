export default function ({ data, parentElement }) {
  const root = parentElement.querySelector('.flight-root');
  const points = data.points;
  root.innerHTML = `
    <style>
      .flight-root {font:14px system-ui,sans-serif;color:#dcecff;width:100%}
      .toolbar {display:flex;align-items:center;gap:18px;flex-wrap:wrap;margin:0 0 6px}
      .legend {display:flex;gap:18px;flex-wrap:wrap;font-size:13px}
      .legend span:first-child {color:#75dfff}.legend span:last-child {color:#ffce70}
      button {background:#071c31;color:#eef5ff;border:1px solid #698bad;border-radius:8px;
        padding:9px 14px;font:inherit;line-height:20px;cursor:pointer;margin-left:auto}
      button:hover {border-color:#ffd17e} button:focus-visible {outline:3px solid #ffd17e}
      button:disabled {opacity:.65;cursor:default}
      svg {display:block;width:100%;overflow:hidden;font:12px system-ui,sans-serif}
      svg:focus-visible {outline:2px solid #75dfff;outline-offset:3px}
      svg text {paint-order:stroke;stroke:#0a213b;stroke-width:2px;stroke-linejoin:round}
      .readout {min-height:24px;color:#e2edf8;font-size:12px;padding-top:4px}
    </style>
    <div class="toolbar"><div class="legend"><span>━ 포트폴리오</span>
      <span>┄ 벤치마크 (합성)</span></div><button type="button">▷ 비행 경로 재생</button></div>
    <svg role="img" tabindex="0" aria-label="날짜별 누적 수익률 비교. 좌우 방향키로 관측일 탐색."></svg>
    <div class="readout" aria-live="polite"></div>`;
  const svg = root.querySelector('svg'), button = root.querySelector('button');
  const readout = root.querySelector('.readout');
  const ns = 'http://www.w3.org/2000/svg';
  const motion = window.matchMedia('(prefers-reduced-motion: reduce)');
  let frame = 0, previousWidth = 0, selected = points.length - 1;
  let x, y, width, portfolio, benchmark, plane, cursor, endLabel, dates;
  function el(tag, attrs, content) {
    const node = document.createElementNS(ns, tag);
    Object.entries(attrs).forEach(([k,v]) => node.setAttribute(k,v));
    if (content !== undefined) node.textContent = content;
    svg.appendChild(node); return node;
  }
  const pct = n => `${n >= 0 ? '+' : ''}${(n*100).toFixed(2)}%`;
  function show(index) {
    selected = index;
    const p = points[index];
    readout.textContent = `${p.date} · 포트폴리오 ${pct(p.portfolio)} · 벤치마크 ${pct(p.benchmark)}`;
    cursor.setAttribute('x1',x(index));cursor.setAttribute('x2',x(index));
  }
  function position(progress) {
    progress = Math.max(0, Math.min(points.length-1, progress));
    const i = Math.min(points.length-2, Math.floor(progress));
    const t = progress-i;
    const px=x(i)+(x(i+1)-x(i))*t;
    const py=y(points[i].portfolio)+(y(points[i+1].portfolio)-y(points[i].portfolio))*t;
    plane.setAttribute('transform',`translate(${px},${py})`);
    const fraction=(px-52)/(width-52-88);
    [portfolio,benchmark].forEach(path => {
      // Clip against the same date coordinate used by the aircraft.
      path.style.clipPath=`inset(0 ${Math.max(0,width-px-1)}px 0 0)`;
    });
    if (fraction >= .9999) [portfolio,benchmark].forEach(p => p.style.clipPath='none');
  }
  function finish() {
    cancelAnimationFrame(frame); frame=0;
    position(points.length-1); endLabel.style.opacity='1';
    button.disabled=motion.matches;
    button.textContent=motion.matches ? '동작 감소 · 정적 표시' : '▷ 비행 경로 재생';
    show(points.length-1);
  }
  function draw() {
    cancelAnimationFrame(frame);
    width=Math.max(root.clientWidth,280); previousWidth=width;
    const height=310, left=52, right=88, top=26, bottom=39;
    svg.replaceChildren(); svg.setAttribute('viewBox',`0 0 ${width} ${height}`);
    svg.style.height=`${height}px`;
    dates=points.map(p=>Date.parse(p.date+'T00:00:00Z'));
    const values=points.flatMap(p=>[p.portfolio,p.benchmark]);
    const low=Math.min(0,...values)-.008, high=Math.max(.01,...values)+.018;
    x=i=>left+(dates[i]-dates[0])/(dates.at(-1)-dates[0])*(width-left-right);
    y=v=>top+(high-v)/(high-low)*(height-top-bottom);
    // The continuous sky belongs to the outer window; SVG stays transparent.
    for(let i=0;i<5;i++) {
      const v=low+(high-low)*i/4, py=y(v);
      el('line',{x1:left,y1:py,x2:width-right,y2:py,stroke:'#6a94b5','stroke-opacity':'.3'});
      el('text',{x:left-10,y:py+4,'text-anchor':'end',fill:'#d6e8fa'},`${(v*100).toFixed(0)}%`);
    }
    const ticks=width<500?3:6;
    for(let i=0;i<ticks;i++) {
      const index=Math.round(i*(points.length-1)/(ticks-1));
      el('text',{x:x(index),y:height-10,'text-anchor':'middle',fill:'#d6e8fa'},points[index].date.slice(5));
    }
    el('text',{x:left,y:14,fill:'#aac8e7'},'누적 수익률 (%)');
    const pathFor=field=>points.map((p,i)=>`${i?'L':'M'}${x(i)},${y(p[field])}`).join(' ');
    benchmark=el('path',{d:pathFor('benchmark'),fill:'none',stroke:'#ffce70','stroke-width':2,'stroke-dasharray':'6 3'});
    portfolio=el('path',{d:pathFor('portfolio'),fill:'none',stroke:'#69dcff','stroke-width':3});
    cursor=el('line',{x1:0,y1:top,x2:0,y2:height-bottom,stroke:'#cce6ff','stroke-opacity':'.25','stroke-dasharray':'3 4'});
    plane=el('g',{'aria-label':'포트폴리오 수익률 선의 끝점'});
    const shape=document.createElementNS(ns,'path');
    shape.setAttribute('d','M 14 0 L 4 -2 L -2 -10 L -5 -10 L -1 -2 L -8 -1 L -11 -5 L -13 -5 L -11 0 L -13 5 L -11 5 L -8 1 L -1 2 L -5 10 L -2 10 L 4 2 Z');
    shape.setAttribute('fill','#e7faff');shape.setAttribute('stroke','#39baff');
    shape.style.filter='drop-shadow(0 0 4px #29baff)';plane.appendChild(shape);
    endLabel=el('text',{x:width-right+18,y:y(points.at(-1).portfolio)+4,fill:'#8de8ff','font-weight':'bold'},pct(points.at(-1).portfolio));
    finish();
  }
  button.onclick=()=>{
    if(motion.matches) return finish();
    cancelAnimationFrame(frame);endLabel.style.opacity='0';button.textContent='↻ 다시 재생';
    position(0);
    let start;
    const tick=now=>{
      // Frame timestamps can precede the click handler's performance.now().
      if(start === undefined) start=now;
      const t=Math.max(0,Math.min(1,(now-start)/3200));
      position(t*(points.length-1));
      if(t<1) frame=requestAnimationFrame(tick); else finish();
    };
    frame=requestAnimationFrame(tick);
  };
  svg.onpointermove=e=>{
    const px=e.clientX-svg.getBoundingClientRect().left;
    let closest=0;
    for(let i=1;i<points.length;i++) if(Math.abs(x(i)-px)<Math.abs(x(closest)-px)) closest=i;
    show(closest);
  };
  svg.onkeydown=e=>{
    if(e.key==='ArrowLeft'||e.key==='ArrowRight'){
      e.preventDefault();show(Math.max(0,Math.min(points.length-1,selected+(e.key==='ArrowLeft'?-1:1))));
    }
  };
  const onMotion=()=>finish();motion.addEventListener('change',onMotion);
  const resize=new ResizeObserver(()=>{if(Math.abs(root.clientWidth-previousWidth)>1) draw();});
  resize.observe(root);draw();
  return ()=>{cancelAnimationFrame(frame);resize.disconnect();motion.removeEventListener('change',onMotion);};
}
