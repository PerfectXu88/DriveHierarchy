'use strict';
const $ = selector => document.querySelector(selector);
const $$ = selector => [...document.querySelectorAll(selector)];
const reducedMotion = window.matchMedia('(prefers-reduced-motion: reduce)');
let pausedByUser = false;
function updateMotion(){
 const paused = pausedByUser || reducedMotion.matches || document.hidden;
 document.documentElement.setAttribute('data-motion',paused?'paused':'playing');
 const control=$('#motion-toggle');
 control.disabled=reducedMotion.matches;
 control.setAttribute('aria-pressed',String(paused));
 const label=reducedMotion.matches?'Visual effects disabled by reduced-motion preference':paused?'Resume visual effects':'Pause visual effects';
 control.setAttribute('aria-label',label);control.title=label;
}
$('#motion-toggle').addEventListener('click',()=>{pausedByUser=!pausedByUser;updateMotion();});
reducedMotion.addEventListener?.('change',updateMotion);
document.addEventListener('visibilitychange',updateMotion);
updateMotion();

const menu=$('.menu-toggle');
function closeMenu(){menu.setAttribute('aria-expanded','false');$('#navigation').classList.remove('is-open');}
menu.addEventListener('click',()=>{
 const open=menu.getAttribute('aria-expanded')!=='true';
 menu.setAttribute('aria-expanded',String(open));$('#navigation').classList.toggle('is-open',open);
});
$('#navigation').addEventListener('click',event=>{if(event.target.closest('a'))closeMenu();});
document.addEventListener('keydown',event=>{if(event.key==='Escape'&&menu.getAttribute('aria-expanded')==='true'){closeMenu();menu.focus();}});
window.matchMedia('(min-width: 1001px)').addEventListener?.('change',event=>{if(event.matches)closeMenu();});
$$('[data-paper-link]').forEach(link=>{
 const placeholder=()=>!link.getAttribute('href')?.trim()||link.getAttribute('href').trim()==='#';
 if(!placeholder()){link.removeAttribute('aria-disabled');link.removeAttribute('title');}
 link.addEventListener('click',event=>{if(placeholder())event.preventDefault();});
});

const scopes={
 refined:{qa:'14,000',frames:'31,945',note:'14 tasks. 1,000 questions per task.'},
 full:{qa:'76,798',frames:'84,279',note:'A unified corpus drawn from multiple autonomous-driving datasets.'}
};
$$('[data-scope]').forEach(button=>button.addEventListener('click',()=>{
 const value=scopes[button.dataset.scope];$('#qa-count').textContent=value.qa;$('#frame-count').textContent=value.frames;$('#scope-note').textContent=value.note;
 $$('[data-scope]').forEach(item=>item.setAttribute('aria-pressed',String(item===button)));
}));
const ranks = [
  {meta:'R1 · 8 OPEN-LOOP TASKS', title:'Ground the scene.', description:'Recognize traffic participants and hazards, estimate distances, localize referred objects, and describe the current traffic situation.',tasks:['Object existence, counting, and attributes','Nearest-object, referred-object, and bucketed distances','Visual grounding and situation description'],image:'perception',alt:"Front camera observation from the paper’s open-loop examples.",label:'OPEN-LOOP / VISUAL OBSERVATION'},
  {meta:'R2 · 4 OPEN-LOOP TASKS', title:'Connect views and time.', description:'Integrate context across camera views, temporal sequences, and spatial relationships to form a coherent scene representation.',tasks:['Multi-view memory across synchronized cameras','Temporal counting and status recognition','Spatial relationships between traffic participants'],image:'memory',alt:'One front-camera view from a multi-view memory example in the paper.',label:'OPEN-LOOP / ONE VIEW OF A MULTI-VIEW EXAMPLE'},
  {meta:'R3 · 2 OPEN-LOOP TASKS', title:'Reason about what comes next.',description:'Move beyond the current observation: predict future behavior and recover the correct temporal order of driving observations.',tasks:['Outcome prediction for a target traffic participant','Sequential planning through temporal ordering'],image:'reasoning',alt:'One observation from a sequential planning example in the paper.',label:'OPEN-LOOP / ONE FRAME OF A SEQUENCE'},
  {meta:'R4 · 100 CLOSED-LOOP SCENARIOS',title:'Put understanding into action.',description:'Evaluate vehicle control under continuous traffic interaction in CARLA–SUMO co-simulation on a real-world road layout.',tasks:['10 families of urban driving interactions','Route completion, safety, and efficiency','Simulation logging, replay, and failure analysis'],image:'scenario-4',alt:'Ego vehicle approaching an intersection in the closed-loop simulator.',label:'CLOSED-LOOP / INTERACTIVE SIMULATION'}
];
const rankButtons = $$('[data-rank]');
function showRank(index) {
  const rank = ranks[index];
  $('#rank-panel').dataset.activeRank=String(index);
  $('#memory-exercise').hidden=index!==1;$('#sequence-exercise').hidden=index!==2;
  rankButtons.forEach((button, i) => {button.setAttribute('aria-selected',String(i===index));button.tabIndex=i===index?0:-1;});
  $('#rank-panel').setAttribute('aria-labelledby',rankButtons[index].id);
  $('#rank-meta').textContent=rank.meta; $('#rank-title').textContent=rank.title; $('#rank-description').textContent=rank.description;
  $('#rank-image').src=`assets/${rank.image}.webp`;$('#rank-zoom').href=`assets/${rank.image}.webp`;$('#rank-zoom').dataset.caption=rank.alt; $('#rank-image').alt=rank.alt; $('#rank-image-label').textContent=rank.label;
  $('#rank-tasks').replaceChildren(...rank.tasks.map(text=>{const li=document.createElement('li');li.textContent=text;return li;}));
  $('#rank-link').textContent=index===3?'Explore closed-loop scenarios ↗':'Explore task records ↗';
  $('#rank-link').href=index===3?'https://github.com/PerfectXu88/DriveHierarchy/tree/main/Close_Loop_Evaluation/Scenario_Onsite':'https://github.com/PerfectXu88/DriveHierarchy/tree/main/Open_Loop_Evaluation';
}
rankButtons.forEach((button,index)=>{
  button.addEventListener('click',()=>showRank(index));
  button.addEventListener('keydown',event=>{
    let next=index;
    if(event.key==='ArrowRight')next=(index+1)%4;else if(event.key==='ArrowLeft')next=(index+3)%4;else if(event.key==='Home')next=0;else if(event.key==='End')next=3;else return;
    event.preventDefault();showRank(next);rankButtons[next].focus();
  });
});

const scenarios=[
 ['Pedestrian encounters','Detect crossing intent, decelerate in time, yield when required, and resume only after the conflict region is clear.'],
 ['Obstacle avoidance','Select a safe maneuver around static obstacles while preserving route continuity and avoiding collisions.'],
 ['Turning maneuver','Coordinate speed control, path tracking, and conflict-aware execution under constrained road geometry.'],
 ['Intersection','Resolve multi-agent interactions and select an appropriate passage strategy as traffic enters from different directions.'],
 ['T-intersection','Infer right-of-way and negotiate crossing or joining traffic without blocking or unsafe encroachment.'],
 ['Traffic disturbances','Remain stable under irregular surrounding motion while preserving safety margins and route progress.'],
 ['Sudden braking','React to abrupt deceleration of the leading vehicle with safe longitudinal control.'],
 ['Merging','Coordinate with surrounding traffic and complete a merge while maintaining safe spacing.'],
 ['Yielding','Recognize priority conflicts and time the ego vehicle’s passage appropriately.'],
 ['Roundabout navigation','Negotiate circulating traffic and follow the route through a roundabout.']
];
let scenarioIndex=0;
function showScenario(index){
 scenarioIndex=(index+scenarios.length)%scenarios.length;
 $$('#scenario-thumbs button').forEach((b,i)=>b.setAttribute('aria-pressed',String(i===scenarioIndex)));
 const [title,description]=scenarios[scenarioIndex], src=`assets/scenario-${scenarioIndex+1}.webp`;
 $('#scenario-select').value=String(scenarioIndex);$('#scenario-image').src=src;$('#scenario-image').alt=`Simulation example: ${title.toLowerCase()}.`;
 $('#scenario-title').textContent=title;$('#scenario-description').textContent=description;$('#scenario-number').textContent=`${String(scenarioIndex+1).padStart(2,'0')} / 10`;
 $('#scenario-zoom').href=src;$('#scenario-zoom').dataset.caption=`${title}: a representative simulation scene from the paper.`;
}
$('#scenario-select').addEventListener('change',event=>showScenario(Number(event.target.value)));
$('#scenario-prev').addEventListener('click',()=>showScenario(scenarioIndex-1));
$('#scenario-next').addEventListener('click',()=>showScenario(scenarioIndex+1));


scenarios.forEach(([title],index)=>{
 const button=document.createElement('button');button.type='button';
 button.setAttribute('aria-label','Show '+title.toLowerCase());
 button.setAttribute('aria-pressed',String(index===0));
 const img=document.createElement('img');img.src='assets/scenario-'+(index+1)+'.webp';img.alt='';img.loading='lazy';img.width=95;img.height=54;
 const label=document.createElement('span');label.textContent=String(index+1).padStart(2,'0')+' / '+title;
 button.append(img,label);button.addEventListener('click',()=>showScenario(index));$('#scenario-thumbs').append(button);
});
const models=[
 {name:'Qwen3-VL',size:'2B',open:43.91,closed:0.377},
 {name:'InternVL3.5',size:'2B',open:39.28,closed:1.218},
 {name:'Qwen3-VL',size:'8B',open:45.09,closed:0.902},
 {name:'InternVL3.5',size:'8B',open:43.49,closed:0.702},
 {name:'ZwZ',size:'8B',open:45.09,closed:0.809},
 {name:'MiniCPM-V-4.5',size:'9B',open:45.50,closed:5.405},
 {name:'Gemma-3-it',size:'12B',open:46.27,closed:0.719},
 {name:'Pixtral',size:'12B',open:42.31,closed:0.704},
 {name:'Gemma-3-it',size:'27B',open:49.53,closed:12.619},
 {name:'Qwen3-VL',size:'32B',open:49.64,closed:3.452},
 {name:'InternVL3.5',size:'38B',open:46.61,closed:1.112},
 {name:'Qwen2.5-VL',size:'72B',open:54.47,closed:9.366},
 {name:'InternVL3',size:'78B',open:46.93,closed:3.142},
 {name:'DA-DriveLM',size:'4B',open:38.31,closed:0.017,specialized:true},
 {name:'ReasonDrive',size:'7B',open:43.13,closed:4.201,specialized:true}
];
let metric='open';
function renderResults(){
 const query=$('#model-search').value.trim().toLowerCase();
 const ordered=[...models].sort((a,b)=>b[metric]-a[metric]);
 let priorScore=null,rank=0;
 const ranked=ordered.map((model,index)=>{if(model[metric]!==priorScore)rank=index+1;priorScore=model[metric];return {...model,rank};});
 const visible=ranked.filter(model=>`${model.name} ${model.size}`.toLowerCase().includes(query));
 const body=$('#results-body');body.replaceChildren();
 visible.forEach(model=>{
  const row=document.createElement('tr');if(model.rank===1)row.className='best-row';
  [String(model.rank).padStart(2,'0'),model.name,model.size,model.open.toFixed(2),model.closed.toFixed(3)].forEach((value,index)=>{
   const cell=document.createElement('td');cell.textContent=value;
   if(index===(metric==='open'?3:4))cell.className='active-score';
   if(index===1&&model.specialized){const tag=document.createElement('span');tag.className='model-tag';tag.textContent='DRIVING';cell.append(tag);}
   row.append(cell);
  });body.append(row);
 });
 if(!visible.length){const row=document.createElement('tr'),cell=document.createElement('td');cell.colSpan=5;cell.textContent='No matching models. Try another name or parameter size.';row.append(cell);body.append(row);}
 $('#result-caption').textContent=`Ranked by ${metric==='open'?'open-loop':'closed-loop'} score`;
 $('#result-count').textContent=`${visible.length} of 15 models`;
}
$$('[data-metric]').forEach(button=>button.addEventListener('click',()=>{metric=button.dataset.metric;$$('[data-metric]').forEach(b=>b.setAttribute('aria-pressed',String(b===button)));renderResults();}));
$('#model-search').addEventListener('input',renderResults);renderResults();

const profileCaptions={midscale:'Capability profiles of representative 8B–12B models.',qwen:'Capability profiles within the Qwen model family.',internvl:'Capability profiles within the InternVL model family.'};
$$('[data-profile]').forEach(button=>button.addEventListener('click',()=>{
 const key=button.dataset.profile;$$('[data-profile]').forEach(b=>b.setAttribute('aria-pressed',String(b===button)));
 $('#profile-image').src=`assets/radar-${key}.webp`;$('#profile-image').alt=profileCaptions[key];$('#profile-zoom').href=`assets/radar-${key}.webp`;$('#profile-zoom').dataset.caption=profileCaptions[key];
}));

// These ground truths are transcribed from QAs.tex, not inferred from images.
$$('[data-preview-rank]').forEach(link=>link.addEventListener('click',()=>showRank(Number(link.dataset.previewRank))));
$('#check-memory').addEventListener('click',()=>{
 const raw=$('#vehicle-count').value,answer=Number(raw);
 $('#memory-feedback').textContent=raw===''||!Number.isInteger(answer)||answer<0?'Enter a non-negative whole number.':answer===10?'Correct — 10 unique vehicles across all six cameras.':'Not quite. Check vehicles that appear in overlapping views, or reveal the ground truth.';
});
$('#reveal-memory').addEventListener('click',()=>{$('#memory-feedback').textContent='Answer: 10 unique vehicles.';});
const correctOrder=['E','D','B','A','C'];
let sequenceOrder=['A','B','C','D','E'];
function renderSequence(focusLetter=null){
 const items=sequenceOrder.map((letter,index)=>{
  const li=document.createElement('li');li.tabIndex=-1;li.dataset.letter=letter;
  li.setAttribute('aria-label','Frame '+letter+', position '+(index+1));
  const link=document.createElement('a');link.className='zoomable';link.href='assets/sequence-'+letter.toLowerCase()+'.webp';link.dataset.caption='Frame '+letter+' from the temporal-ordering benchmark example.';
  const image=document.createElement('img');image.src=link.href;image.alt='Observation '+letter+' from the driving sequence.';image.width=700;image.height=394;image.loading='lazy';link.append(image);
  const controls=document.createElement('div');controls.className='sequence-controls';
  const earlier=document.createElement('button');earlier.type='button';earlier.textContent='←';earlier.disabled=index===0;earlier.setAttribute('aria-label','Move frame '+letter+' earlier');
  const later=document.createElement('button');later.type='button';later.textContent='→';later.disabled=index===4;later.setAttribute('aria-label','Move frame '+letter+' later');
  const label=document.createElement('strong');label.textContent=letter;
  earlier.addEventListener('click',()=>moveFrame(index,-1));later.addEventListener('click',()=>moveFrame(index,1));controls.append(earlier,label,later);
  const position=document.createElement('span');position.className='sequence-position';position.textContent='POSITION '+(index+1);
  li.append(link,controls,position);return li;
 });
 $('#sequence-grid').replaceChildren(...items);
 if(focusLetter)items.find(item=>item.dataset.letter===focusLetter)?.focus({preventScroll:true});
}
function moveFrame(index,delta){
 const next=index+delta;if(next<0||next>=sequenceOrder.length)return;
 const letter=sequenceOrder[index];[sequenceOrder[index],sequenceOrder[next]]=[sequenceOrder[next],sequenceOrder[index]];
 $('#sequence-feedback').textContent='Frame '+letter+' moved to position '+(next+1)+'.';renderSequence(letter);
}
function sequenceConsistency(order){
 let correct=0;
 for(let i=0;i<order.length;i++)for(let j=i+1;j<order.length;j++)if(correctOrder.indexOf(order[i])<correctOrder.indexOf(order[j]))correct++;
 return correct;
}
$('#check-sequence').addEventListener('click',()=>{
 const pairs=sequenceConsistency(sequenceOrder);
 $('#sequence-feedback').textContent=pairs===10?'Correct — all 10 pairwise relations match the chronology.':pairs+' of 10 pairwise relations are correct ('+(pairs*10)+'%). Continue arranging the frames or reveal the chronology.';
});
$('#reveal-sequence').addEventListener('click',()=>{
 sequenceOrder=[...correctOrder];renderSequence();$('#sequence-feedback').textContent='Ground-truth chronology: E → D → B → A → C.';
});
$('#reset-sequence').addEventListener('click',()=>{sequenceOrder=['A','B','C','D','E'];renderSequence();$('#sequence-feedback').textContent='Reset to alphabetical order.';});
renderSequence();

const dialog=$('#figure-dialog');
let figureTrigger=null,zoom=1;
function setZoom(value){
 zoom=Math.max(1,Math.min(3,value));$('#dialog-image').style.width=(zoom*100)+'%';
 $('#zoom-out').disabled=zoom===1;$('#zoom-in').disabled=zoom===3;
}
document.addEventListener('click',event=>{
 const link=event.target.closest('.zoomable');if(!link||!dialog.showModal||event.ctrlKey||event.metaKey||event.shiftKey||event.altKey)return;
 event.preventDefault();figureTrigger=link;$('#dialog-image').src=link.href;$('#dialog-image').alt=link.querySelector('img')?.alt||'Expanded research figure';
 $('#dialog-caption').textContent=link.dataset.caption||'';$('#download-figure').href=link.href;setZoom(1);
 dialog.showModal();document.body.classList.add('modal-open');$('.dialog-image-wrap').scrollTo(0,0);$('.dialog-close').focus();
});
$('.dialog-close').addEventListener('click',()=>dialog.close());
$('#zoom-in').addEventListener('click',()=>setZoom(zoom+.5));$('#zoom-out').addEventListener('click',()=>setZoom(zoom-.5));
dialog.addEventListener('keydown',event=>{
 if(event.key==='+'||event.key==='='){event.preventDefault();setZoom(zoom+.5);}
 if(event.key==='-'){event.preventDefault();setZoom(zoom-.5);}
});
dialog.addEventListener('click',event=>{
 if(event.target===dialog){const r=dialog.getBoundingClientRect();if(event.clientX<r.left||event.clientX>r.right||event.clientY<r.top||event.clientY>r.bottom)dialog.close();}
});
dialog.addEventListener('close',()=>{document.body.classList.remove('modal-open');figureTrigger?.focus({preventScroll:true});});
$('#copy-citation').addEventListener('click',async()=>{
 try{if(!navigator.clipboard||!window.isSecureContext)throw new Error('Clipboard unavailable');
 await navigator.clipboard.writeText($('#bibtex').textContent.trim());$('#copy-status').textContent='Citation copied to clipboard.';
 }catch{
 const range=document.createRange();range.selectNodeContents($('#bibtex'));const selection=window.getSelection();selection.removeAllRanges();selection.addRange(range);
 let copied=false;try{copied=document.execCommand('copy');}catch{}
 if(copied){selection.removeAllRanges();$('#copy-status').textContent='Citation copied to clipboard.';}
 else $('#copy-status').textContent='Citation selected. Press Ctrl+C or ⌘C to copy.';
 }
});

const navigation=$$('#navigation a[href^="#"]');
let scrollFrame=0;
function updateScroll(){
 scrollFrame=0;const range=document.documentElement.scrollHeight-window.innerHeight;
 $('.reading-progress').style.width=(range>0?Math.max(0,Math.min(100,window.scrollY/range*100)):0)+'%';
 let active=-1;
 navigation.forEach((link,index)=>{if($(link.getAttribute('href')).getBoundingClientRect().top<=170)active=index;});
 navigation.forEach((link,index)=>{link.classList.toggle('active',index===active);if(index===active)link.setAttribute('aria-current','location');else link.removeAttribute('aria-current');});
}
window.addEventListener('scroll',()=>{if(!scrollFrame)scrollFrame=window.requestAnimationFrame(updateScroll);},{passive:true});
window.addEventListener('resize',updateScroll,{passive:true});updateScroll();
if(!reducedMotion.matches&&'IntersectionObserver' in window){
 const observer=new IntersectionObserver(entries=>entries.forEach(entry=>{if(entry.isIntersecting){entry.target.classList.remove('pending');observer.unobserve(entry.target);}}),{threshold:.04,rootMargin:'0px 0px 35px 0px'});
 $$('.section-heading,.intro-grid,.dataset-strip,.rank-panel,.editor-figure,.editor-interface,.scenario-stage,.result-leaders,.profiles-grid,.improvement-grid,.citation-box').forEach(element=>{
  element.classList.add('reveal');if(element.getBoundingClientRect().top>window.innerHeight)element.classList.add('pending');observer.observe(element);
 });document.body.classList.add('motion-ready');
}
(() => {
 const fine=window.matchMedia('(hover:hover) and (pointer:fine)');
 let active=null,frame=0,point={x:0,y:0};
 const enabled=()=>fine.matches&&!reducedMotion.matches&&!document.hidden&&document.documentElement.getAttribute('data-motion')==='playing';
 function reset(){if(frame)window.cancelAnimationFrame(frame);frame=0;active?.removeAttribute('data-light');active=null;}
 function paint(){frame=0;if(!active||!enabled()){reset();return;}const r=active.getBoundingClientRect();active.style.setProperty('--light-x',(point.x-r.left)+'px');active.style.setProperty('--light-y',(point.y-r.top)+'px');active.setAttribute('data-light','on');}
 $$('.overview-teaser,.paper-figure,.rank-panel,.editor-figure,.scenario-stage,.result-leaders,.citation-box').forEach(surface=>{
  surface.classList.add('light-surface');surface.addEventListener('pointermove',event=>{
   if(event.pointerType==='touch'||!enabled())return;if(active!==surface){reset();active=surface;}point={x:event.clientX,y:event.clientY};if(!frame)frame=window.requestAnimationFrame(paint);
  },{passive:true});surface.addEventListener('pointerleave',()=>{if(active===surface)reset();});surface.addEventListener('pointercancel',reset);
 });
 window.addEventListener('scroll',reset,{passive:true});window.addEventListener('blur',reset);document.addEventListener('visibilitychange',reset);reducedMotion.addEventListener?.('change',reset);fine.addEventListener?.('change',reset);$('#motion-toggle').addEventListener('click',reset);
})();

// A short pointer tail and a one-shot response on the actual SVG layer faces.
(() => {
 const cover=$('#home'),trail=$('.pointer-trail'),tail=$('#pointer-trail-path'),ink=$('#pointer-trail-ink');
 const face=$('#hierarchy-plane .stratum-face');
 const fine=window.matchMedia('(hover:hover) and (pointer:fine)');
 const forcedColors=window.matchMedia('(forced-colors: active)');
 const narrow=window.matchMedia('(max-width:720px)');
 const wideLayers=$$('.scene-wide .interactive-layer').reverse();
 const narrowLayers=$$('.scene-narrow .interactive-layer').reverse();
 const reactions=new Map();
 const lifetime=170,maxLength=76;
 let frame=0,pending=null,points=[],activeLayer=null;
 const allowed=()=>fine.matches&&!forcedColors.matches&&!reducedMotion.matches&&!document.hidden&&document.documentElement.getAttribute('data-motion')==='playing';
 function reset(){
  if(frame)window.cancelAnimationFrame(frame);
  frame=0;pending=null;points=[];activeLayer=null;
  tail.setAttribute('d','');trail.style.opacity='0';
  reactions.forEach((animation,layer)=>{animation.cancel();layer.classList.remove('is-reacting');});
  reactions.clear();
 }
 function findLayer(x,y){
  if(typeof DOMPoint!=='function'||typeof face.isPointInFill!=='function')return null;
  const cursor=new DOMPoint(x,y);
  for(const layer of narrow.matches?narrowLayers:wideLayers){
   // The stationary wrapper keeps hit testing stable while its child bounces.
   const matrix=layer.getScreenCTM();
   if(!matrix)continue;
   const local=cursor.matrixTransform(matrix.inverse());
   if(face.isPointInFill(local))return layer;
  }
  return null;
 }
 function react(layer){
  if(layer===activeLayer)return;
  activeLayer=layer;
  if(!layer)return;
  const response=layer.querySelector('.layer-response');
  if(typeof response.animate!=='function')return;
  reactions.get(layer)?.cancel();
  layer.classList.add('is-reacting');
  const animation=response.animate([
   {transform:'translateY(0) scale(1)',offset:0},
   {transform:'translateY(-12px) scale(1.025)',offset:.3},
   {transform:'translateY(2px) scale(.998)',offset:.62},
   {transform:'translateY(-2px) scale(1.005)',offset:.82},
   {transform:'translateY(0) scale(1)',offset:1}
  ],{duration:620,easing:'cubic-bezier(.2,.65,.3,1)'});
  reactions.set(layer,animation);
  animation.onfinish=()=>{if(reactions.get(layer)===animation){reactions.delete(layer);layer.classList.remove('is-reacting');}};
 }
 function clippedTrail(){
  if(points.length<2)return [];
  const result=[points[points.length-1]];
  let remaining=maxLength;
  for(let i=points.length-2;i>=0&&remaining>0;i--){
   const newer=result[result.length-1],older=points[i];
   const distance=Math.hypot(newer.x-older.x,newer.y-older.y);
   if(distance>remaining){
    const t=remaining/distance;
    result.push({x:newer.x+(older.x-newer.x)*t,y:newer.y+(older.y-newer.y)*t});break;
   }
   result.push(older);remaining-=distance;
  }
  return result.reverse();
 }
 function drawTrail(time){
  points=points.filter(point=>time-point.time<lifetime);
  const segment=clippedTrail();
  if(segment.length<2){tail.setAttribute('d','');trail.style.opacity='0';return;}
  const first=segment[0],last=segment[segment.length-1];
  let d='M'+first.x.toFixed(1)+' '+first.y.toFixed(1);
  for(let i=1;i<segment.length-1;i++){
   const p=segment[i],next=segment[i+1];
   d+=' Q'+p.x.toFixed(1)+' '+p.y.toFixed(1)+' '+((p.x+next.x)/2).toFixed(1)+' '+((p.y+next.y)/2).toFixed(1);
  }
  d+=' L'+last.x.toFixed(1)+' '+last.y.toFixed(1);
  tail.setAttribute('d',d);
  ink.setAttribute('x1',String(first.x));ink.setAttribute('y1',String(first.y));
  ink.setAttribute('x2',String(last.x));ink.setAttribute('y2',String(last.y));
  trail.style.opacity=String(.68*Math.max(0,1-(time-points[points.length-1].time)/lifetime));
 }
 function paint(time){
  frame=0;
  if(!allowed()){reset();return;}
  if(pending){
   const cursor=pending;pending=null;
   const rect=cover.getBoundingClientRect();
   const x=cursor.x-rect.left,y=cursor.y-rect.top;
   if(x<0||y<0||x>rect.width||y>rect.height){reset();return;}
   const last=points[points.length-1];
   if(!last||Math.hypot(x-last.x,y-last.y)>1.5){
    points.push({x,y,time});if(points.length>14)points.shift();
   }
   react(findLayer(cursor.x,cursor.y));
  }
  drawTrail(time);
  if(points.length)frame=window.requestAnimationFrame(paint);
 }
 cover.addEventListener('pointermove',event=>{
  if(event.pointerType==='touch'||event.buttons||!allowed()){reset();return;}
  pending={x:event.clientX,y:event.clientY};
  if(!frame)frame=window.requestAnimationFrame(paint);
 },{passive:true});
 cover.addEventListener('pointerleave',reset);
 cover.addEventListener('pointercancel',reset);
 cover.addEventListener('pointerdown',reset,{passive:true});
 window.addEventListener('scroll',reset,{passive:true});
 window.addEventListener('resize',reset,{passive:true});
 window.addEventListener('blur',reset);
 document.addEventListener('visibilitychange',reset);
 fine.addEventListener?.('change',reset);
 forcedColors.addEventListener?.('change',reset);
 narrow.addEventListener?.('change',reset);
 reducedMotion.addEventListener?.('change',reset);
 $('#motion-toggle').addEventListener('click',reset);
})();
