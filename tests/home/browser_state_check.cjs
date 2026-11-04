/* node tests/home/browser_state_check.cjs — 외부 패키지 없는 비동기 UI 회귀 검사 */
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');
const source = fs.readFileSync(path.join(__dirname, '../../src/docagent/home/assets/app.js'), 'utf8');
const clone = value => JSON.parse(JSON.stringify(value));
const settle = () => new Promise(resolve => setImmediate(resolve));
function deferred() {let resolve, reject; const promise = new Promise((yes,no)=>{resolve=yes;reject=no;});return {promise,resolve,reject};}
function snapshot(complete=true) {
  const group={id:'g1',source_text:'검토된 원문',options:[{id:'yes',label:'동의함'},{id:'no',label:'동의하지 않음'}],
    max_selected:1,min_selected:1,pending_option_ids:null,selected_option_ids:['no'],confirmed:true,
    verification:{status:complete?'matched':'uncertain',message:complete?'선택과 일치합니다.':'다시 확인해주세요.'}};
  return {title:'합성 문서',purpose:'demo',document_ready:true,has_capture:true,can_verify:true,
    session:{current_index:0,current_group:clone(group),groups:[group],complete,preferences:{speech_rate:1.5}}};
}
class Element {
  constructor(tag='div') {this.tagName=tag;this.children=[];this.disabled=false;this.hidden=false;this.value='';this.files=[];this.textContent='';this.src='';this.videoWidth=1000;this.videoHeight=1400;this.classList={add(){},remove(){}};}
  replaceChildren(...items) {this.children=items;}
  append(...items) {this.children.push(...items);}
  removeAttribute(name) {delete this[name];}
  get innerText() {return this.textContent + this.children.map(x=>x.innerText||x.textContent||'').join('\n');}
  play() {return Promise.resolve();}
}
async function harness() {
  const elements=new Map(), calls=[], readers=[], routes=new Map();
  const get=id=>{if(!elements.has(id))elements.set(id,new Element());return elements.get(id);};
  let mediaCalls=0, mediaResult=null, intervalCount=0;
  const document={getElementById:get,querySelector:()=>({content:'test-session'}),
    createElement:tag=>new Element(tag),createTextNode:text=>Object.assign(new Element('text'),{textContent:text}),
    querySelectorAll:selector=>{
      const inputs=[]; const visit=e=>{if(e.tagName==='input')inputs.push(e);for(const child of e.children)visit(child);};visit(get('choices'));
      return selector.endsWith(':checked')?inputs.filter(x=>x.checked):inputs;
    }};
  class Reader {constructor(){readers.push(this);}readAsDataURL(file){this.file=file;}}
  const context=vm.createContext({document,window:{addEventListener(){}},FileReader:Reader,
    navigator:{mediaDevices:{getUserMedia(){mediaCalls++;return mediaResult?mediaResult.promise:Promise.reject(new Error('unconfigured'));}}},
    fetch:async(url,options)=>{
      const action=url.replace('/api/','');calls.push({action,options});
      const handler=routes.get(action);
      if(handler)return handler(options);
      return {ok:true,json:async()=>clone(action==='reference'?{image:'reference'}:snapshot())};
    },setInterval(){intervalCount++;return intervalCount;},clearInterval(){},console});
  vm.runInContext(source,context); await settle();await settle();
  return {elements,get,calls,readers,routes,context,
    setMedia(value){mediaResult=value;},get mediaCalls(){return mediaCalls;},get intervalCount(){return intervalCount;},
    value(expression){return vm.runInContext(expression,context);},
    respond(action,value){routes.set(action,async()=>({ok:true,json:async()=>clone(value)}));},
    fail(action){routes.set(action,async()=>({ok:false,json:async()=>({error:'요청 실패'})}));}};
}
async function testBusyControlsCannotStartAnotherUpload() {
  const h=await harness(), pending=deferred();
  h.routes.set('verify',()=>pending.promise);
  const verifying=h.get('verify').onclick();
  assert.equal(h.get('file').disabled,true);assert.equal(h.get('camera').disabled,true);
  h.get('file').files=[{size:100}];h.get('file').onchange();h.get('camera').onclick();
  assert.equal(h.calls.filter(c=>c.action==='reset_capture').length,0);assert.equal(h.mediaCalls,0);assert.equal(h.readers.length,0);
  pending.resolve({ok:true,json:async()=>snapshot()});await verifying;
  assert.equal(h.get('file').disabled,false);
}
async function testFailedResetCannotReuseCompletedStateOrPrivacyImage() {
  const h=await harness();h.fail('reset_capture');h.get('masked').src='old-private-image';h.get('masked').hidden=false;h.get('privacy-status').textContent='old mask';
  h.get('file').files=[{size:100}];await h.get('file').onchange();
  assert.equal(h.readers.length,0);assert.equal(h.calls.some(c=>c.action==='capture'),false);
  assert.equal(h.value('state.session.complete'),false);assert.equal(h.value('state.has_capture'),false);
  assert.equal(h.get('masked').src,undefined);assert.equal(h.get('masked').hidden,true);assert.equal(h.get('privacy-status').textContent,'');
  assert.equal(h.get('verify').disabled,true);
}
async function testFileReaderIsPartOfTheLockedOperation() {
  const h=await harness();h.respond('reset_capture',{...snapshot(false),has_capture:false,document_ready:false,can_verify:false});h.respond('capture',snapshot(false));
  h.get('file').files=[{size:100}];const uploading=h.get('file').onchange();await settle();
  assert.equal(h.readers.length,1);assert.equal(h.get('next').disabled,true);assert.equal(h.get('scan').disabled,true);
  const before=h.calls.length;await h.get('next').onclick();assert.equal(h.calls.length,before);
  h.readers[0].result='data:image/png;base64,example';h.readers[0].onload();await uploading;
  assert.equal(h.calls.filter(c=>c.action==='capture').length,1);assert.equal(h.value('captureInvalidated'),false);
  assert.equal(h.get('file').disabled,false);assert.equal(h.get('rate').value,'1.5');
}
async function testCancellingPendingCameraStopsLateTracksAndBlocksDuplicateStarts() {
  const h=await harness(), media=deferred();h.setMedia(media);
  const starting=h.get('camera').onclick();await settle();assert.equal(h.mediaCalls,1);
  h.get('camera').onclick();assert.equal(h.mediaCalls,1);assert.equal(h.get('camera-stop').disabled,false);
  h.get('camera-stop').onclick();let stopped=0;media.resolve({getTracks:()=>[{stop(){stopped++;}}]});await starting;
  assert.equal(stopped,1);assert.equal(h.value('stream'),null);assert.equal(h.intervalCount,0);assert.equal(h.get('camera').disabled,false);
}
async function testFailedCameraResetDoesNotRequestPermission() {
  const h=await harness();h.fail('reset_capture');await h.get('camera').onclick();assert.equal(h.mediaCalls,0);assert.equal(h.value('state.session.complete'),false);
}
async function testChangedSelectionRequiresFreshCaptureButRetainsPrivacyAccess() {
  const h=await harness();h.respond('confirm',{...snapshot(false),can_verify:false});
  await h.get('confirm').onclick();
  assert.equal(h.get('verify').disabled,true);assert.equal(h.get('privacy').disabled,false);
  assert.equal(h.value('state.has_capture'),true);assert.equal(h.value('state.can_verify'),false);
  h.value('invalidateCaptureView()');assert.equal(h.value('state.can_verify'),false);
}
async function testFailedCaptureKeepsPriorSuccessInvalidated() {
  const h=await harness();h.fail('capture');h.get('file').files=[{size:100}];const uploading=h.get('file').onchange();await settle();
  h.readers[0].result='invalid';h.readers[0].onload();await uploading;
  assert.equal(h.value('state.session.complete'),false);assert.equal(h.value('state.document_ready'),false);assert.equal(h.get('read-group').disabled,true);
}
(async()=>{
  const tests=[testBusyControlsCannotStartAnotherUpload,testFailedResetCannotReuseCompletedStateOrPrivacyImage,testFileReaderIsPartOfTheLockedOperation,
    testCancellingPendingCameraStopsLateTracksAndBlocksDuplicateStarts,testFailedCameraResetDoesNotRequestPermission,testFailedCaptureKeepsPriorSuccessInvalidated,testChangedSelectionRequiresFreshCaptureButRetainsPrivacyAccess];
  for(const test of tests){await test();console.log(`PASS ${test.name}`);}
  console.log(`${tests.length} browser state regressions passed`);
})().catch(error=>{console.error(error);process.exitCode=1;});
