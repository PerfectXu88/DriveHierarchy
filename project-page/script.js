'use strict';
const $ = (selector) => document.querySelector(selector);
const $$ = (selector) => [...document.querySelectorAll(selector)];
const menu = $('.menu-toggle');
const updateHeader = () => $('.site-header').classList.toggle('is-scrolled', window.scrollY > 35);
window.addEventListener('scroll', updateHeader, {passive:true});
updateHeader();

// Native scrolling drives the cover; no wheel interception or scroll locking.
const cover = $('.hero-scroll');
const reducedMotion = window.matchMedia('(prefers-reduced-motion: reduce)');
let coverFrame = 0;
function updateCover() {
  coverFrame = 0;
  const progress = reducedMotion.matches ? 0 : Math.max(0, Math.min(1, -cover.getBoundingClientRect().top / window.innerHeight));
  const fade = Math.min(1, progress / 0.7);
  $('#overview').style.setProperty('--page-width', `${document.documentElement.clientWidth}px`);
  $('#overview').style.setProperty('--edge-opacity', String(Math.min(1, progress * 5)));
  cover.style.setProperty('--cover-content-opacity', String(1 - fade));
  cover.style.setProperty('--cover-content-y', `${-65 * progress}px`);
  cover.style.setProperty('--cover-scale', String(1 + 0.08 * progress));
  cover.style.setProperty('--cover-veil', String(progress * progress * 0.9));
  cover.style.setProperty('--cover-cue-opacity', String(Math.max(0, 1 - progress * 4)));
  // Invisible cover controls must not retain keyboard focus targets.
  $('.hero-content').inert = !reducedMotion.matches && fade >= 1;
  $('.hero-bottom').inert = !reducedMotion.matches && progress >= 0.25;
}
function queueCover() { if (!coverFrame) coverFrame = requestAnimationFrame(updateCover); }
window.addEventListener('scroll', queueCover, {passive:true});
window.addEventListener('resize', queueCover, {passive:true});
reducedMotion.addEventListener('change', queueCover);
updateCover();
menu.addEventListener('click', () => {
  const open = menu.getAttribute('aria-expanded') !== 'true';
  menu.setAttribute('aria-expanded', String(open));
  $('#navigation').classList.toggle('is-open', open);
});
$('#navigation').addEventListener('click', (event) => {
  if (event.target.closest('a')) { menu.setAttribute('aria-expanded', 'false'); $('#navigation').classList.remove('is-open'); }
});
document.addEventListener('keydown', (event) => {
  if (event.key === 'Escape' && menu.getAttribute('aria-expanded') === 'true') {
    menu.setAttribute('aria-expanded', 'false'); $('#navigation').classList.remove('is-open'); menu.focus();
  }
});

const ranks = [
  {meta:'R1 · 8 OPEN-LOOP TASKS', title:'Ground the scene.', description:'Recognize traffic participants and hazards, estimate distances, localize referred objects, and describe the current traffic situation.',tasks:['Object existence, counting, and attributes','Nearest-object, referred-object, and bucketed distances','Visual grounding and situation description'],image:'perception',alt:"Front camera observation from the paper’s open-loop examples.",label:'OPEN-LOOP / VISUAL OBSERVATION'},
  {meta:'R2 · 4 OPEN-LOOP TASKS', title:'Connect views and time.', description:'Integrate context across camera views, temporal sequences, and spatial relationships to form a coherent scene representation.',tasks:['Multi-view memory across synchronized cameras','Temporal counting and status recognition','Spatial relationships between traffic participants'],image:'memory',alt:'One front-camera view from a multi-view memory example in the paper.',label:'OPEN-LOOP / ONE VIEW OF A MULTI-VIEW EXAMPLE'},
  {meta:'R3 · 2 OPEN-LOOP TASKS', title:'Reason about what comes next.',description:'Move beyond the current observation: predict future behavior and recover the correct temporal order of driving observations.',tasks:['Outcome prediction for a target traffic participant','Sequential planning through temporal ordering'],image:'reasoning',alt:'One observation from a sequential planning example in the paper.',label:'OPEN-LOOP / ONE FRAME OF A SEQUENCE'},
  {meta:'R4 · 100 CLOSED-LOOP SCENARIOS',title:'Put understanding into action.',description:'Evaluate vehicle control under continuous traffic interaction in CARLA–SUMO co-simulation on a real-world road layout.',tasks:['10 families of urban driving interactions','Route completion, safety, and efficiency','Simulation logging, replay, and failure analysis'],image:'scenario-4',alt:'Ego vehicle approaching an intersection in the closed-loop simulator.',label:'CLOSED-LOOP / INTERACTIVE SIMULATION'}
];
const rankButtons = $$('[data-rank]');
function showRank(index) {
  const rank = ranks[index];
  rankButtons.forEach((button, i) => {button.setAttribute('aria-selected',String(i===index));button.tabIndex=i===index?0:-1;});
  $('#rank-panel').setAttribute('aria-labelledby',rankButtons[index].id);
  $('#rank-meta').textContent=rank.meta; $('#rank-title').textContent=rank.title; $('#rank-description').textContent=rank.description;
  $('#rank-image').src=`assets/${rank.image}.webp`; $('#rank-image').alt=rank.alt; $('#rank-image-label').textContent=rank.label;
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
 const [title,description]=scenarios[scenarioIndex], src=`assets/scenario-${scenarioIndex+1}.webp`;
 $('#scenario-select').value=String(scenarioIndex);$('#scenario-image').src=src;$('#scenario-image').alt=`Simulation example: ${title.toLowerCase()}.`;
 $('#scenario-title').textContent=title;$('#scenario-description').textContent=description;$('#scenario-number').textContent=`${String(scenarioIndex+1).padStart(2,'0')} / 10`;
 $('#scenario-zoom').href=src;$('#scenario-zoom').dataset.caption=`${title}: a representative simulation scene from the paper.`;
}
$('#scenario-select').addEventListener('change',event=>showScenario(Number(event.target.value)));
$('#scenario-prev').addEventListener('click',()=>showScenario(scenarioIndex-1));
$('#scenario-next').addEventListener('click',()=>showScenario(scenarioIndex+1));

// Paper Tables 1 and 2. Open-loop and closed-loop metrics have different protocols.
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
 $('#result-caption').textContent=`Paper results · sorted by ${metric==='open'?'open-loop':'closed-loop'} score · higher is better`;
 $('#result-count').textContent=`${visible.length} of 15 models · Tables 1 & 2 in the paper. The two scores use different evaluation protocols and should not be compared numerically across columns. Ties share a rank.`;
}
$$('[data-metric]').forEach(button=>button.addEventListener('click',()=>{metric=button.dataset.metric;$$('[data-metric]').forEach(b=>b.setAttribute('aria-pressed',String(b===button)));renderResults();}));
$('#model-search').addEventListener('input',renderResults);renderResults();

const profileCaptions={midscale:'Capability profiles of representative 8B–12B models.',qwen:'Capability profiles within the Qwen model family.',internvl:'Capability profiles within the InternVL model family.'};
$$('[data-profile]').forEach(button=>button.addEventListener('click',()=>{
 const key=button.dataset.profile;$$('[data-profile]').forEach(b=>b.setAttribute('aria-pressed',String(b===button)));
 $('#profile-image').src=`assets/radar-${key}.webp`;$('#profile-image').alt=profileCaptions[key];$('#profile-zoom').href=`assets/radar-${key}.webp`;$('#profile-zoom').dataset.caption=profileCaptions[key];
}));
const dialog=$('#figure-dialog');
let figureTrigger=null;
$$('.zoomable').forEach(link=>link.addEventListener('click',event=>{
 if(!dialog.showModal||event.ctrlKey||event.metaKey||event.shiftKey||event.altKey)return;
 event.preventDefault();figureTrigger=link;$('#dialog-image').src=link.href;$('#dialog-image').alt=link.querySelector('img').alt;$('#dialog-caption').textContent=link.dataset.caption||'';
 dialog.showModal();document.body.classList.add('modal-open');
}));
$('.dialog-close').addEventListener('click',()=>dialog.close());
dialog.addEventListener('click',event=>{if(event.target===dialog){const r=dialog.getBoundingClientRect();if(event.clientX<r.left||event.clientX>r.right||event.clientY<r.top||event.clientY>r.bottom)dialog.close();}});
dialog.addEventListener('close',()=>{document.body.classList.remove('modal-open');figureTrigger?.focus();});
$('#copy-citation').addEventListener('click',async()=>{
 try{
  if(!navigator.clipboard)throw new Error('Clipboard unavailable');
  await navigator.clipboard.writeText($('#bibtex').textContent);$('#copy-status').textContent='Citation copied to clipboard.';
 }catch{
  const range=document.createRange();range.selectNodeContents($('#bibtex'));const selection=window.getSelection();selection.removeAllRanges();selection.addRange(range);
  $('#copy-status').textContent='Citation selected. Press Ctrl+C or ⌘C to copy.';
 }
});
