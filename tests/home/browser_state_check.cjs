/* node tests/home/browser_state_check.cjs — 외부 패키지 없는 비동기 UI 회귀 검사 */
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');
const source = fs.readFileSync(path.join(__dirname, '../../src/docagent/home/assets/app.js'), 'utf8');
const audioSource = fs.readFileSync(path.join(__dirname, '../../src/docagent/home/assets/audio.js'), 'utf8');
const clone = value => JSON.parse(JSON.stringify(value));
const settle = () => new Promise(resolve => setImmediate(resolve));
function deferred() {let resolve, reject; const promise = new Promise((yes,no)=>{resolve=yes;reject=no;});return {promise,resolve,reject};}
function snapshot(complete=true) {
  const group={id:'g1',source_text:'검토된 원문',options:[{id:'yes',label:'동의함'},{id:'no',label:'동의하지 않음'}],
    max_selected:1,min_selected:1,pending_option_ids:null,selected_option_ids:['no'],confirmed:true,
    verification:{status:complete?'matched':'uncertain',message:complete?'선택과 일치합니다.':'다시 확인해주세요.'}};
  return {title:'합성 문서',purpose:'demo',document_ready:true,has_capture:true,can_verify:true,
    capture_id:'capture-1',document:{summary:'신청인 서명란이 1개 있습니다.'},questions_enabled:true,speech_enabled:true,
    session:{current_index:0,current_group:clone(group),groups:[group],complete,preferences:{speech_rate:1.5}}};
}
class Element {
  constructor(tag='div') {this.tagName=tag;this.children=[];this.disabled=false;this.hidden=false;this.value='';this.files=[];this.textContent='';this.src='';this.videoWidth=1000;this.videoHeight=1400;this.classList={add(){},remove(){}};}
  replaceChildren(...items) {this.children=items;}
  append(...items) {this.children.push(...items);}
  removeAttribute(name) {delete this[name];}
  get innerText() {return this.textContent + this.children.map(x=>x.innerText||x.textContent||'').join('\n');}
  play() {return Promise.resolve();}
  focus() {this.focused=true;}
  getContext() {return {drawImage(){}};}
  toDataURL() {return 'data:image/jpeg;base64,evaluated-frame';}
}
async function harness() {
  const elements=new Map(), calls=[], readers=[], routes=new Map(), intervals=new Map(), listeners=new Map();
  const get=id=>{if(!elements.has(id))elements.set(id,new Element());return elements.get(id);};
  let mediaCalls=0, mediaResult=null, intervalCount=0;
  const document={getElementById:get,querySelector:()=>({content:'test-session'}),addEventListener(name,fn){listeners.set(name,fn);},
    createElement:tag=>new Element(tag),createTextNode:text=>Object.assign(new Element('text'),{textContent:text}),
    querySelectorAll:selector=>{
      const inputs=[]; const visit=e=>{if(e.tagName==='input')inputs.push(e);for(const child of e.children)visit(child);};visit(get('choices'));
      return selector.endsWith(':checked')?inputs.filter(x=>x.checked):inputs;
    }};
  class Reader {constructor(){readers.push(this);}readAsDataURL(file){this.file=file;}}
  class Recorder {
    constructor() {this.state='inactive';this.mimeType='audio/webm';}
    start() {this.state='recording';}
    stop() {this.state='inactive';this.finish=()=>{this.ondataavailable({data:new Blob(['audio'])});this.onstop();};if(!this.delayStop)queueMicrotask(this.finish);}
  }
  const context=vm.createContext({document,window:{MediaRecorder:Recorder,addEventListener(name,fn){listeners.set(name,fn);}},FileReader:Reader,
    MediaRecorder:Recorder,Blob,
    navigator:{mediaDevices:{getUserMedia(){mediaCalls++;return mediaResult?mediaResult.promise:Promise.reject(new Error('unconfigured'));}}},
    fetch:async(url,options)=>{
      const action=url.replace('/api/','');calls.push({action,options});
      const handler=routes.get(action);
      if(handler)return handler(options);
      return {ok:true,json:async()=>clone(action==='reference'?{image:'reference'}:snapshot())};
    },setInterval(fn){intervalCount++;intervals.set(intervalCount,fn);return intervalCount;},clearInterval(id){intervals.delete(id);},
    setTimeout(){return 1;},clearTimeout(){},console});
  vm.runInContext(audioSource,context);
  vm.runInContext(source,context); await settle();await settle();
  return {elements,get,calls,readers,routes,context,intervals,listeners,
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
async function startCamera(h) {
  let stopped=0;const media=deferred();h.setMedia(media);
  const starting=h.get('camera').onclick();await settle();
  media.resolve({getTracks:()=>[{stop(){stopped++;}}]});await starting;
  return {tick:[...h.intervals.values()][0],get stopped(){return stopped;}};
}
async function testOnlyManualShutterAfterFocusTakesFullResolutionImage() {
  const h=await harness();
  h.value('let now = 5000; Date.now = () => now');
  h.get('video').videoWidth=3840;h.get('video').videoHeight=2160;
  const camera=await startCamera(h);
  h.respond('frame',{ready_to_capture:false,beep_state:'stabilizing',stability_progress:.4,instruction:'잠시 고정'});
  await camera.tick();assert.equal(h.calls.filter(c=>c.action==='capture').length,0);
  assert.equal(h.get('canvas').width,960);
  h.respond('frame',{ready_to_capture:true,focus_clear:true,beep_state:'ready',stability_progress:1,instruction:'준비'});
  await camera.tick();await camera.tick();
  assert.equal(h.get('focus').disabled,false);assert.equal(h.get('take').disabled,true);
  await h.get('take').onclick();await h.get('focus').onclick();
  assert.equal(h.get('take').disabled,true);
  h.value('now += 500');await camera.tick();
  h.value('now += 250');await camera.tick();
  assert.equal(h.get('take').disabled,false);
  assert.equal(h.calls.filter(c=>c.action==='capture').length,0);
  h.respond('frame',{ready_to_capture:true,focus_clear:false,beep_state:'ready'});
  await camera.tick();assert.equal(h.get('take').disabled,true);await h.get('take').onclick();
  h.respond('frame',{ready_to_capture:true,focus_clear:true,beep_state:'ready'});
  h.value('now += 200');await camera.tick();h.value('now += 200');await camera.tick();
  await h.get('take').onclick();
  const captures=h.calls.filter(c=>c.action==='capture');assert.equal(captures.length,1);
  assert.equal(JSON.parse(captures[0].options.body).image,'data:image/jpeg;base64,evaluated-frame');
  assert.equal(h.get('canvas').width,3840);
  assert.equal(camera.stopped,1);assert.equal(h.intervals.size,0);
}
async function testStoppedCameraDiscardsLateReadyFrame() {
  const h=await harness();
  const camera=await startCamera(h),pending=deferred();h.routes.set('frame',()=>pending.promise);
  const frame=camera.tick();h.get('camera-stop').onclick();
  pending.resolve({ok:true,json:async()=>({ready_to_capture:true,beep_state:'ready'})});await frame;
  assert.equal(h.calls.filter(c=>c.action==='capture').length,0);assert.equal(h.value('audio.active'),false);
}
async function testCancelledMicrophoneStopsLateTracks() {
  const h=await harness(),pending=deferred();h.setMedia(pending);
  const starting=h.get('record').onclick();await settle();assert.equal(h.get('record-stop').disabled,false);
  h.get('record-stop').onclick();let stopped=0;
  pending.resolve({getTracks:()=>[{stop(){stopped++;}}]});await starting;
  assert.equal(stopped,1);assert.equal(h.value('recorder'),null);assert.equal(h.value('microphone'),null);
}
async function testRecordingStopsTracksAndRequiresQuestionConfirmation() {
  const h=await harness(),pending=deferred();h.setMedia(pending);let stopped=0;
  const starting=h.get('record').onclick();await settle();pending.resolve({getTracks:()=>[{stop(){stopped++;}}]});await starting;
  assert.equal(h.get('ask').disabled,true);assert.equal(h.get('camera').disabled,true);
  h.respond('transcribe',{text:'서명란이 어디에 있나요?',capture_id:'capture-1'});
  h.get('record-stop').onclick();await settle();
  assert.ok(stopped>0);assert.equal(h.readers.length,1);
  h.readers[0].result='data:audio/webm;base64,YXVkaW8=';h.readers[0].onload();await settle();await settle();
  assert.equal(h.get('question').value,'서명란이 어디에 있나요?');
  assert.equal(h.calls.filter(c=>c.action==='ask').length,0);assert.equal(h.get('ask').disabled,false);
}
async function testHiddenPageDiscardsLateTranscription() {
  const h=await harness(),pending=deferred();h.setMedia(pending);
  const starting=h.get('record').onclick();await settle();pending.resolve({getTracks:()=>[{stop(){}}]});await starting;
  const transcription=deferred();h.routes.set('transcribe',()=>transcription.promise);
  h.get('record-stop').onclick();await settle();h.readers[0].result='data:audio/webm;base64,YXVkaW8=';h.readers[0].onload();await settle();
  h.value('document.hidden=true');h.listeners.get('visibilitychange')();
  transcription.resolve({ok:true,json:async()=>({text:'늦은 질문',capture_id:'capture-1'})});await settle();await settle();
  assert.equal(h.get('question').value,'');
}
async function testRecaptureClearsQuestionEvidence() {
  const h=await harness();h.get('question').value='이전 질문';h.get('answer').textContent='이전 답변';
  await h.get('clear').onclick();assert.equal(h.get('question').value,'');assert.equal(h.get('answer').textContent,'');
  assert.equal(h.value('state.capture_id'),null);assert.equal(h.get('ask').disabled,true);
}
async function testHiddenDuringResetCannotStartCameraLater() {
  const h=await harness(),pending=deferred();h.routes.set('reset_capture',()=>pending.promise);
  const starting=h.get('camera').onclick();await settle();
  h.value('document.hidden=true');h.listeners.get('visibilitychange')();
  pending.resolve({ok:true,json:async()=>snapshot(false)});await starting;
  assert.equal(h.mediaCalls,0);assert.equal(h.value('audio.active'),false);
}
async function testRecordingFinalizationKeepsControlsLocked() {
  const h=await harness(),pending=deferred();h.setMedia(pending);
  const starting=h.get('record').onclick();await settle();pending.resolve({getTracks:()=>[{stop(){}}]});await starting;
  const recorder=h.value('recorder');recorder.delayStop=true;
  h.get('record-stop').onclick();assert.equal(h.get('guide').disabled,true);
  const calls=h.calls.length;await h.get('guide').onclick();assert.equal(h.calls.length,calls);
  recorder.finish();await settle();assert.equal(h.readers.length,1);
}
function ocrState(ready=true) {
  return {...snapshot(), mode:'ocr', document_ready:ready, has_capture:ready, can_verify:ready,
    capture_ocr:ready?{text:'준비물은 물입니다.',lines:[{text:'준비물은 물입니다.'}]}:null,
    session:{groups:[],current_group:null,current_index:0,preferences:{speech_rate:1},complete:false}};
}
async function testOcrWorkflowOnlyRevealsActionsAfterCapture() {
  const h=await harness();h.value(`render(${JSON.stringify(ocrState(false))})`);
  assert.equal(h.get('capture-panel').hidden,false);
  assert.equal(h.get('document-panel').hidden,true);
  assert.equal(h.get('ask').disabled,true);
  h.respond('capture',ocrState(true));
  await h.value('requestAction("capture", {})');
  assert.equal(h.get('capture-panel').hidden,true);
  assert.equal(h.get('document-panel').hidden,false);
  assert.equal(h.get('document-title').focused,true);
  assert.equal(h.get('legacy-selection').hidden,true);
  assert.equal(h.get('source').textContent,'준비물은 물입니다.');
  h.get('open-question').onclick();
  assert.equal(h.get('question-panel').hidden,false);
  assert.equal(h.get('read-panel').hidden,true);
  assert.equal(h.get('question-title').focused,true);
  h.value('invalidateCaptureView()');
  assert.equal(h.get('document-panel').hidden,true);
  assert.equal(h.get('question-panel').hidden,true);
}
async function testOcrCheckboxReadoutIsVisibleAndNoChoicesRequired() {
  const h=await harness();h.value(`render(${JSON.stringify(ocrState())})`);
  h.respond('verify',{...ocrState(),instruction:'체크칸 후보: 우유, 표시 있음으로 추정'});
  await h.get('open-checks').onclick();
  assert.equal(h.get('checks-panel').hidden,false);
  assert.equal(h.get('results').textContent,'체크칸 후보: 우유, 표시 있음으로 추정');
}
(async()=>{
  const tests=[testBusyControlsCannotStartAnotherUpload,testFailedResetCannotReuseCompletedStateOrPrivacyImage,testFileReaderIsPartOfTheLockedOperation,
    testCancellingPendingCameraStopsLateTracksAndBlocksDuplicateStarts,testFailedCameraResetDoesNotRequestPermission,testFailedCaptureKeepsPriorSuccessInvalidated,testChangedSelectionRequiresFreshCaptureButRetainsPrivacyAccess,
    testOnlyManualShutterAfterFocusTakesFullResolutionImage,testStoppedCameraDiscardsLateReadyFrame,testCancelledMicrophoneStopsLateTracks,
    testRecordingStopsTracksAndRequiresQuestionConfirmation,testHiddenPageDiscardsLateTranscription,testRecaptureClearsQuestionEvidence,
    testHiddenDuringResetCannotStartCameraLater,testRecordingFinalizationKeepsControlsLocked,testOcrWorkflowOnlyRevealsActionsAfterCapture,testOcrCheckboxReadoutIsVisibleAndNoChoicesRequired];
  for(const test of tests){await test();console.log(`PASS ${test.name}`);}
  console.log(`${tests.length} browser state regressions passed`);
})().catch(error=>{console.error(error);process.exitCode=1;});
