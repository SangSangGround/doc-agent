"use strict";
const $ = id => document.getElementById(id);
const token = document.querySelector('meta[name="session-token"]').content;
let state = null, stream = null, timer = null, frameBusy = false, actionBusy = false;
let voice = null, cameraGeneration = 0, frameEpoch = 0, cameraStarting = false;
let captureInvalidated = false;
const headers = {"Content-Type":"application/json", "X-Session-Token":token};
function voices() {
  voice = (window.speechSynthesis?.getVoices() || []).find(v => v.localService && /^ko/i.test(v.lang));
  $("voice-status").textContent = voice ? "기기에 설치된 한국어 음성을 사용합니다." : "로컬 한국어 음성을 찾지 못했습니다. 화면 읽기 기능으로 안내를 확인할 수 있습니다.";
}
function speak(text) {
  if (!voice || !text) return;
  speechSynthesis.cancel();
  const speech = new SpeechSynthesisUtterance(text);
  speech.voice = voice; speech.lang = voice.lang; speech.rate = Number($("rate").value);
  speechSynthesis.speak(speech);
}
function notice(text, audio = true) { $("notice").textContent = text; if(audio) speak(text); }
async function api(path, data) {
  const response = await fetch(`/api/${path}`, data === undefined ? {headers} : {method:"POST",headers,body:JSON.stringify(data)});
  const value = await response.json();
  if (!response.ok) throw new Error(value.error || "요청을 처리하지 못했습니다.");
  return value;
}
function group() { return state?.session.current_group; }
function updateControls() {
  const busy = actionBusy || !state;
  $("camera").disabled = busy || !!stream || cameraStarting;
  $("camera-stop").disabled = !stream && !cameraStarting;
  $("take").disabled = busy || !stream;
  for(const id of ["file","scan","rate","previous","next","cancel","clear"]) $(id).disabled = busy;
  for(const id of ["read-page","read-group","propose"]) $(id).disabled = busy || !state?.document_ready;
  $("privacy").disabled = busy || !state?.has_capture;
  $("verify").disabled = busy || !state?.can_verify;
  $("confirm").disabled = busy || group()?.pending_option_ids == null;
  for(const input of document.querySelectorAll('#choices input')) input.disabled = busy || !state?.document_ready;
}
function clearPrivacyPreview() {
  $("masked").removeAttribute("src"); $("masked").hidden = true; $("privacy-status").textContent = "";
}
function invalidateCaptureView() {
  captureInvalidated = true;
  clearPrivacyPreview(); window.speechSynthesis?.cancel();
  if(state) render(state);
}
function render(value) {
  if (!value.session) return;
  // 새 파일/촬영을 시도한 뒤에는 실패 응답의 이전 서버 상태도 완료로 표시하지 않는다.
  if(captureInvalidated) {
    value = JSON.parse(JSON.stringify(value));
    value.document_ready = false; value.has_capture = false; value.can_verify = false; value.session.complete = false;
    for(const item of value.session.groups) item.verification = {status:"uncertain",message:"새 문서를 촬영한 뒤 다시 확인해주세요.",marked_option_ids:[]};
    value.session.current_group = value.session.groups[value.session.current_index];
  }
  state = value;
  $("rate").value=String(value.session.preferences.speech_rate);
  $("title").textContent = value.title;
  $("mode").textContent = value.purpose === "demo" ? "합성 문서 연습 모드 · 실제 동의서가 아닙니다" : "검토한 지원 양식";
  const current = group();
  $("position").textContent = `현재 ${value.session.current_index + 1}번째 / ${value.session.groups.length}개 항목`;
  $("source").textContent = value.document_ready ? current.source_text : "촬영한 문서가 지원 양식과 일치하는지 확인한 뒤 원문을 제공합니다.";
  const fieldset = $("choices");
  fieldset.replaceChildren(Object.assign(document.createElement("legend"), {textContent:"표시할 선택지"}));
  for(const option of current.options) {
    const label = document.createElement("label"), input = document.createElement("input");
    // 선택하지 않아도 되는 항목은 이미 고른 값을 해제할 수 있어야 한다.
    input.type = current.max_selected === 1 && current.min_selected > 0 ? "radio" : "checkbox";
    input.name = "option"; input.value = option.id;
    input.checked = (current.pending_option_ids || current.selected_option_ids || []).includes(option.id);
    label.append(input, document.createTextNode(option.label)); fieldset.append(label);
  }
  const selected = current.pending_option_ids ?? current.selected_option_ids;
  const labels = current.options.filter(o => selected?.includes(o.id)).map(o => o.label).join(", ");
  $("selection").textContent = current.pending_option_ids != null
    ? `선택 내용: ${labels || "선택 없음"}. 맞으면 ‘이 선택으로 확정’을 누르세요.`
    : current.confirmed ? `확정한 선택: ${labels || "선택 없음"}` : "아직 선택을 확정하지 않았습니다.";
  const results = $("results"); results.replaceChildren();
  for(const item of value.session.groups) {
    const p = document.createElement("p"); p.textContent = `${value.session.groups.indexOf(item)+1}번째 항목: ${item.verification.message}`; results.append(p);
  }
  if(value.session.complete) results.append(Object.assign(document.createElement("p"),{textContent:"모든 선택 항목의 표시가 확정한 선택과 일치합니다."}));
  updateControls();
}
async function requestAction(path, data = {}, fallback) {
  notice("처리하고 있습니다.", false);
  const result = await api(path,data);
  if(path === "capture") captureInvalidated = false;
  render(result);
  notice(result.instruction || fallback || "처리했습니다.");
  return result;
}
async function operation(work) {
  if(actionBusy || !state) return null;
  actionBusy = true; frameEpoch++; updateControls();
  try { return await work(); }
  catch(error) {
    try { render(await api("state")); } catch (_) {}
    notice(error.message || "처리하지 못했습니다. 다시 시도해주세요.");
    return null;
  } finally { actionBusy = false; updateControls(); }
}
async function action(path, data = {}, fallback) {
  return operation(()=>requestAction(path,data,fallback));
}
function snapshot() {
  const video=$("video"), canvas=$("canvas");
  if (!video.videoWidth) throw new Error("카메라 화면을 기다려주세요.");
  const scale=Math.min(1,1800/Math.max(video.videoWidth,video.videoHeight));
  canvas.width=Math.round(video.videoWidth*scale); canvas.height=Math.round(video.videoHeight*scale);
  canvas.getContext("2d").drawImage(video,0,0,canvas.width,canvas.height);
  return canvas.toDataURL("image/jpeg",.92);
}
function stopCamera() {
  cameraGeneration++; clearInterval(timer); timer=null; cameraStarting=false;
  stream?.getTracks().forEach(track=>track.stop()); stream=null;
  $("video").srcObject=null; $("video").classList.remove("active");
  updateControls();
}
$("camera").onclick=()=>{
  if(actionBusy || cameraStarting || stream) return;
  return operation(async()=>{
    stopCamera(); invalidateCaptureView();
    await requestAction("reset_capture"); // 실패하면 카메라 권한 요청으로 진행하지 않는다.
    const generation=++cameraGeneration; cameraStarting=true; updateControls();
    try {
      const opened=await navigator.mediaDevices.getUserMedia({video:{facingMode:"environment",width:{ideal:1920},height:{ideal:1080}},audio:false});
      if(generation!==cameraGeneration) {opened.getTracks().forEach(track=>track.stop()); return;}
      stream=opened; $("video").srcObject=stream; $("video").classList.add("active"); await $("video").play();
      if(generation!==cameraGeneration) return;
      notice("문서 네 모서리가 보이도록 카메라를 맞춰주세요.");
      timer=setInterval(async()=>{
        if(frameBusy || actionBusy || !stream) return;
        frameBusy=true; const epoch=frameEpoch;
        try {
          const result=await api("frame",{image:snapshot()});
          if(generation===cameraGeneration && epoch===frameEpoch && !actionBusy && result.should_announce) notice(result.instruction);
        } catch(error) {if(generation===cameraGeneration && epoch===frameEpoch && !actionBusy) notice(error.message,false);}
        finally {frameBusy=false;}
      },900);
    } catch(error) {
      if(generation!==cameraGeneration) return;
      stopCamera(); throw new Error("카메라를 열지 못했습니다. 브라우저 권한을 확인하거나 이미지 파일을 불러오세요.");
    } finally {cameraStarting=false;updateControls();}
  });
};
$("camera-stop").onclick=()=>{stopCamera();notice("카메라를 껐습니다.");};
$("take").onclick=()=>{
  if(actionBusy || !stream) return;
  return operation(async()=>{
    const image=snapshot(); stopCamera(); invalidateCaptureView();
    return requestAction("capture",{image,mode:"photo"});
  });
};
function readFile(file) {
  return new Promise((resolve,reject)=>{
    const reader=new FileReader(); reader.onload=()=>resolve(reader.result);
    reader.onerror=reader.onabort=()=>reject(new Error("파일을 읽지 못했습니다."));
    reader.readAsDataURL(file);
  });
}
$("file").onchange=()=>{
  if(actionBusy) {$("file").value="";notice("현재 작업이 끝난 뒤 파일을 다시 선택해주세요.",false);return;}
  const file=$("file").files[0]; if(!file)return;
  const mode=$("scan").checked?"scan":"photo";
  return operation(async()=>{
    stopCamera(); invalidateCaptureView();
    await requestAction("reset_capture");
    if(file.size>8*1024*1024) throw new Error("8MB 이하 PNG 또는 JPEG를 사용해주세요.");
    const image=await readFile(file);
    return requestAction("capture",{image,mode});
  });
};
$("read-page").onclick=()=>action("read",{scope:"page"});
$("read-group").onclick=()=>action("read",{scope:"current"});
$("stop").onclick=()=>{window.speechSynthesis?.cancel();notice("읽기를 정지했습니다.",false);};
$("rate").onchange=()=>action("preferences",{speech_rate:Number($("rate").value)},"읽기 속도를 변경했습니다.");
for(const direction of ["previous","next"]) $(direction).onclick=()=>action("move",{direction},"항목을 이동했습니다.");
$("propose").onclick=async()=>{
  if(actionBusy || !group()) return;
  const result=await action("select",{group_id:group().id,option_ids:[...document.querySelectorAll('#choices input:checked')].map(x=>x.value)});
  if(result)notice($("selection").textContent);
};
$("confirm").onclick=()=>{if(!group())return;return action("confirm",{group_id:group().id},"선택을 확정했습니다. 작성 후 다시 촬영해서 결과를 확인해주세요.");};
$("cancel").onclick=()=>{if(!group())return;return action("cancel",{group_id:group().id},"변경 중인 선택을 취소했습니다.");};
$("verify").onclick=async()=>{if(actionBusy)return;stopCamera();const result=await action("verify");if(result)notice($("results").innerText);};
$("privacy").onclick=()=>operation(async()=>{
  clearPrivacyPreview();
  const result=await requestAction("privacy");
  $("privacy-status").textContent=result.instruction; $("masked").hidden=!result.preview;
  if(result.preview)$("masked").src=result.preview;
});
$("clear").onclick=()=>operation(async()=>{
  stopCamera(); $("file").value=""; invalidateCaptureView();
  return requestAction("reset_capture",{},"촬영 이미지를 지웠습니다. 다시 촬영해야 체크 결과를 확인할 수 있습니다.");
});
window.addEventListener("pagehide",()=>{stopCamera();window.speechSynthesis?.cancel();});
if(window.speechSynthesis){speechSynthesis.onvoiceschanged=voices;voices();}
updateControls();
(async()=>{try{render(await api("state"));$("reference").src=(await api("reference")).image;notice("문서를 촬영하거나 이미지 파일을 불러오세요.",false);}catch(error){notice(error.message);}})();
