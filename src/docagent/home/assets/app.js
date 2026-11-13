"use strict";
const $ = id => document.getElementById(id);
const token = document.querySelector('meta[name="session-token"]').content;
const audio = new window.GuidanceAudio();
let state = null, stream = null, timer = null, frameBusy = false, actionBusy = false;
let voice = null, cameraGeneration = 0, frameEpoch = 0, cameraStarting = false;
let captureInvalidated = false;
let microphone = null, recorder = null, recordingTimer = null, microphoneGeneration = 0;
let microphoneStarting = false;
let microphoneFinishing = false;
const headers = {"Content-Type":"application/json", "X-Session-Token":token};
function voices() {
  voice = (window.speechSynthesis?.getVoices() || []).find(v => v.localService && /^ko/i.test(v.lang));
  $("voice-status").textContent = voice ? "기기에 설치된 한국어 음성을 사용합니다." : "로컬 한국어 음성을 찾지 못했습니다. 화면 읽기 기능으로 안내를 확인할 수 있습니다.";
}
function speak(text) {
  if (!document.hidden && $("voice-enabled").checked) audio.speak(text, voice, Number($("rate").value));
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
  const busy = actionBusy || !state || !!microphone || microphoneFinishing;
  $("camera").disabled = busy || !!stream || cameraStarting;
  $("camera-stop").disabled = !stream && !cameraStarting;
  $("take").disabled = busy || !stream;
  for(const id of ["file","scan","rate","previous","next","cancel","clear","new-capture","open-read","open-question","open-checks"]) $(id).disabled = busy;
  for(const id of ["read-page","read-group","propose"]) $(id).disabled = busy || !state?.document_ready;
  $("privacy").disabled = busy || !state?.has_capture;
  $("verify").disabled = busy || !state?.can_verify;
  $("confirm").disabled = busy || group()?.pending_option_ids == null;
  $("camera-device").disabled = busy || !!stream || cameraStarting;
  $("guide").disabled = busy || !state?.document_ready;
  $("ask").disabled = busy || !state?.document_ready || !state?.questions_enabled;
  $("question").disabled = busy || !state?.document_ready;
  $("record").disabled = busy || !state?.document_ready || !state?.speech_enabled;
  $("record-stop").disabled = !microphone && !microphoneStarting;
  for(const input of document.querySelectorAll('#choices input')) input.disabled = busy || !state?.document_ready;
}
function clearPrivacyPreview() {
  $("masked").removeAttribute("src"); $("masked").hidden = true; $("privacy-status").textContent = "";
}
function invalidateCaptureView() {
  captureInvalidated = true;
  clearPrivacyPreview(); audio.stopSpeech();
  $("answer").textContent = "";
  $("question").value = "";
  $("question-sources").replaceChildren();
  $("results").replaceChildren();
  $("recording-status").textContent = "";
  if(state) render(state);
}
function render(value) {
  if (!value.session) return;
  // 새 파일/촬영을 시도한 뒤에는 실패 응답의 이전 서버 상태도 완료로 표시하지 않는다.
  if(captureInvalidated) {
    value = JSON.parse(JSON.stringify(value));
    value.document_ready = false; value.has_capture = false; value.can_verify = false; value.session.complete = false;
    value.document = null; value.capture_id = null;
    value.capture_ocr = null; value.content_check = null;
    for(const item of value.session.groups) item.verification = {status:"uncertain",message:"새 문서를 촬영한 뒤 다시 확인해주세요.",marked_option_ids:[]};
    value.session.current_group = value.session.groups[value.session.current_index];
  }
  state = value;
  $("capture-panel").hidden = value.document_ready;
  $("document-panel").hidden = !value.document_ready;
  $("legacy-selection").hidden = value.mode === "ocr";
  if (!value.document_ready) for (const id of ["read-panel", "question-panel", "checks-panel"]) $(id).hidden = true;
  $("rate").value=String(value.session.preferences.speech_rate);
  $("title").textContent = value.title;
  $("mode").textContent = value.purpose === "demo" ? "합성 문서 연습 모드 · 실제 동의서가 아닙니다" : "검토한 지원 양식";
  $("document-summary").textContent = value.document?.summary || "촬영 후 서명란과 서명 전에 읽을 원문을 안내합니다.";
  const ocr = value.capture_ocr || value.document?.ocr;
  $("ocr-text").textContent = ocr?.text || ocr?.message || "촬영본을 확인한 뒤 표시합니다.";
  $("model-status").textContent = value.questions_enabled
    ? "질문과 원문을 이 노트북의 로컬 모델로 처리합니다."
    : "질의응답이 꺼져 있습니다. 원문 읽기와 서명 안내를 이용할 수 있습니다.";
  const current = group();
  if (value.mode === "ocr") {
    const lines = value.capture_ocr?.lines || [];
    $("position").textContent = lines.length ? `현재 ${value.session.current_index + 1}번째 / ${lines.length}개 줄` : "";
    $("source").textContent = lines[value.session.current_index]?.text || "";
    $("read-group").textContent = "현재 줄 듣기";
    updateControls();
    return;
  }
  if (!current) {updateControls(); return;}
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
  if (path !== "reset_capture" || fallback) notice(result.instruction || fallback || "처리했습니다.");
  if (path === "capture" && result.document_ready) $("document-title").focus();
  return result;
}
async function operation(work) {
  if(actionBusy || !state || microphone || microphoneFinishing) return null;
  actionBusy = true; frameEpoch++; updateControls();
  try { return await work(); }
  catch(error) {
    if (!stream) audio.stop();
    try { render(await api("state")); } catch (_) {}
    if ($("answer").textContent === "원문에서 답변 근거를 찾고 있습니다.") $("answer").textContent = error.message;
    notice(error.message || "처리하지 못했습니다. 다시 시도해주세요.");
    return null;
  } finally { actionBusy = false; updateControls(); }
}
async function action(path, data = {}, fallback) {
  return operation(()=>requestAction(path,data,fallback));
}
function snapshot(maxSide = 960, jpegQuality = .92) {
  const video=$("video"), canvas=$("canvas");
  if (!video.videoWidth) throw new Error("카메라 화면을 기다려주세요.");
  const scale=Math.min(1,maxSide/Math.max(video.videoWidth,video.videoHeight));
  canvas.width=Math.round(video.videoWidth*scale); canvas.height=Math.round(video.videoHeight*scale);
  canvas.getContext("2d").drawImage(video,0,0,canvas.width,canvas.height);
  return canvas.toDataURL("image/jpeg",jpegQuality);
}
function stopCamera() {
  cameraGeneration++; clearInterval(timer); timer=null; cameraStarting=false;
  stream?.getTracks().forEach(track=>track.stop()); stream=null;
  $("video").srcObject=null; $("video").classList.remove("active");
  audio.stop();
  $("stability").value = 0;
  updateControls();
}
async function refreshCameras() {
  if (!navigator.mediaDevices?.enumerateDevices) return;
  try {
    const selected = $("camera-device").value;
    const devices = await navigator.mediaDevices.enumerateDevices();
    $("camera-device").replaceChildren(Object.assign(document.createElement("option"), {value:"", textContent:"기본 카메라"}));
    for (const device of devices.filter(item => item.kind === "videoinput" && item.deviceId)) {
      $("camera-device").append(Object.assign(document.createElement("option"), {
        value:device.deviceId, textContent:device.label || "연결된 카메라",
      }));
    }
    $("camera-device").value = selected;
  } catch (_) {
    $("camera-status").textContent = "카메라 목록을 읽지 못했습니다. 기본 카메라로 시작할 수 있습니다.";
  }
}

async function captureFrame(image) {
  if (actionBusy || !stream) return;
  return operation(async () => {
    stopCamera(); invalidateCaptureView();
    $("camera-status").textContent = "촬영했습니다. 사진의 글자를 읽고 있습니다.";
    notice("촬영했습니다. 사진의 글자를 읽고 있습니다.");
    const result = await requestAction("capture", {image, mode:"photo"});
    $("camera-status").textContent = result.document_ready
      ? "문서를 확인했습니다. 아래에서 원문을 듣거나 질문할 수 있습니다."
      : "촬영본을 확인하기 어렵습니다. 카메라를 다시 켜서 촬영해 주세요.";
    return result;
  });
}

$("camera").onclick=()=>{
  if(actionBusy || cameraStarting || stream) return;
  return operation(async()=>{
    stopCamera(); invalidateCaptureView();
    const generation=++cameraGeneration; cameraStarting=true; updateControls();
    let openingCamera = false;
    try {
      try {
        const started = await audio.start();
        $("audio-status").textContent = started ? "비프음이 켜졌습니다." : "이 브라우저에서 비프음을 시작할 수 없습니다.";
      }
      catch (_) { $("camera-status").textContent = "비프음을 시작하지 못했습니다. 음성·화면 안내로 진행합니다."; }
      if (generation!==cameraGeneration || document.hidden) {audio.stop(); return;}
      await requestAction("reset_capture"); // 실패하면 카메라 권한 요청으로 진행하지 않는다.
      if (generation!==cameraGeneration || document.hidden) return;
      const video = {width:{ideal:3840},height:{ideal:2160}};
      if ($("camera-device").value) video.deviceId = {exact:$("camera-device").value};
      openingCamera = true;
      const opened=await navigator.mediaDevices.getUserMedia({video,audio:false});
      if(generation!==cameraGeneration) {opened.getTracks().forEach(track=>track.stop()); return;}
      stream=opened; $("video").srcObject=stream; $("video").classList.add("active"); await $("video").play();
      if(generation!==cameraGeneration) return;
      await refreshCameras();
      if(generation!==cameraGeneration) return;
      notice($("auto-capture").checked ? "문서 네 모서리를 보여 주세요. 자동 촬영이 켜져 있습니다." : "문서 네 모서리를 보여 주세요. 준비되면 지금 촬영을 눌러 주세요.");
      $("camera-status").textContent = "문서를 찾고 있습니다. 안정되면 자동으로 촬영합니다.";
      timer=setInterval(async()=>{
        if(frameBusy || actionBusy || !stream) return;
        frameBusy=true; const epoch=frameEpoch;
        try {
          const image = snapshot();
          const result=await api("frame",{image});
          if(generation!==cameraGeneration || epoch!==frameEpoch || actionBusy || !stream) return;
          audio.update(result.beep_state);
          $("stability").value = result.stability_progress || 0;
          $("camera-status").textContent = ($("auto-capture").checked ? "자동 촬영 켜짐 · " : "수동 촬영 · ") + result.instruction;
          if (result.ready_to_capture && $("auto-capture").checked) {
            // 안내 프레임보다 큰 원본 해상도로 촬영한다. 서버가 이 촬영본의
            // 경계·정합·원문을 다시 검사하므로 직후 움직임도 통과 근거로 쓰지 않는다.
            await captureFrame(snapshot(4000, .96));
          } else if (result.should_announce && !audio.speaking) {
            notice(result.instruction);
          }
        } catch(error) {if(generation===cameraGeneration && epoch===frameEpoch && !actionBusy) notice(error.message,false);}
        finally {frameBusy=false;}
      },150);
    } catch(error) {
      if(generation!==cameraGeneration) return;
      stopCamera();
      if (openingCamera) throw new Error("카메라를 열지 못했습니다. 브라우저 권한을 확인하거나 이미지 파일을 불러오세요.");
      throw error;
    } finally {cameraStarting=false;updateControls();}
  });
};
$("camera-stop").onclick=()=>{stopCamera();audio.stopSpeech();notice("카메라를 껐습니다.");};
$("take").onclick=()=>{
  if(actionBusy || !stream) return;
  try { return captureFrame(snapshot(4000, .96)); }
  catch (error) { notice(error.message); }
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
$("stop").onclick=()=>{audio.stopSpeech();notice("읽기를 정지했습니다.",false);};
$("voice-enabled").onchange=()=>{if(!$("voice-enabled").checked) audio.stopSpeech();};
$("beep-enabled").onchange=()=>audio.setEnabled($("beep-enabled").checked);
$("guide").onclick=()=>action("guide");
$("ask").onclick=()=>operation(async()=>{
  const question = $("question").value.trim();
  if (!question) throw new Error("궁금한 내용을 입력하거나 마이크로 말해 주세요.");
  audio.stopSpeech();
  $("answer").textContent = "원문에서 답변 근거를 찾고 있습니다.";
  $("question-sources").replaceChildren();
  const result = await api("ask", {question, capture_id:state.capture_id});
  if (captureInvalidated || result.capture_id !== state.capture_id) return;
  $("answer").textContent = result.answer.answer;
  for (const source of result.answer.sources) {
    const item = document.createElement("p");
    item.textContent = `근거 · ${source.page}쪽\n${source.text}`;
    $("question-sources").append(item);
  }
  notice(result.answer.answer);
});

function stopRecording(discard = false) {
  clearTimeout(recordingTimer);
  recordingTimer = null;
  if (discard || microphoneStarting) microphoneGeneration++;
  microphoneStarting = false;
  if (recorder && recorder.state !== "inactive") {
    microphoneFinishing = !discard;
    recorder.stop();
  }
  microphone?.getTracks().forEach(track => track.stop());
  microphone = null;
  if (discard) {recorder = null;microphoneFinishing = false;}
  updateControls();
}

$("record").onclick=()=>operation(async()=>{
  if (!window.MediaRecorder) throw new Error("이 브라우저에서 음성 녹음을 지원하지 않습니다. 키보드로 질문해 주세요.");
  stopCamera(); audio.stopSpeech();
  const generation = ++microphoneGeneration;
  const captureId = state.capture_id;
  microphoneStarting = true;
  updateControls();
  try {
    const opened = await navigator.mediaDevices.getUserMedia({audio:{echoCancellation:true},video:false});
    if (generation !== microphoneGeneration) {
      opened.getTracks().forEach(track => track.stop());
      return;
    }
    microphoneStarting = false;
    microphone = opened;
    const chunks = [];
    const activeRecorder = new MediaRecorder(opened);
    recorder = activeRecorder;
    activeRecorder.ondataavailable = event => {if(event.data.size) chunks.push(event.data);};
    activeRecorder.onerror = () => {
      if (generation !== microphoneGeneration) return;
      stopRecording(true);
      notice("녹음을 완료하지 못했습니다. 다시 시도해 주세요.");
    };
    activeRecorder.onstop = async () => {
      opened.getTracks().forEach(track => track.stop());
      if (generation !== microphoneGeneration) return;
      clearTimeout(recordingTimer);
      microphoneFinishing = false;
      microphone = null;
      recorder = null;
      updateControls();
      await operation(async () => {
        const blob = new Blob(chunks, {type:activeRecorder.mimeType});
        if (!blob.size || blob.size > 4*1024*1024) throw new Error("녹음 크기를 확인하고 짧게 다시 질문해 주세요.");
        $("recording-status").textContent = "음성을 글자로 바꾸고 있습니다.";
        try {
          const audioData = await readFile(blob);
          const result = await api("transcribe", {audio:audioData, capture_id:captureId});
          if (generation !== microphoneGeneration || result.capture_id !== state.capture_id) return;
          $("question").value = result.text;
          $("recording-status").textContent = "인식한 질문을 확인한 뒤 질문하기를 누르세요.";
          notice(`질문을 이렇게 들었습니다. ${result.text}. 맞으면 질문하기를 누르세요.`);
          $("question").focus();
        } catch (error) {
          $("recording-status").textContent = "음성을 인식하지 못했습니다. 다시 녹음하거나 키보드로 입력해 주세요.";
          throw error;
        }
      });
    };
    activeRecorder.start();
    recordingTimer = setTimeout(() => {if (generation===microphoneGeneration) stopRecording();}, 30000);
    $("recording-status").textContent = "녹음 중입니다. 말씀하신 뒤 녹음 마치기를 누르세요. 최대 30초입니다.";
    notice("녹음 중입니다. 질문을 말씀해 주세요.", false);
  } catch (error) {
    stopRecording(true);
    throw new Error("마이크를 열지 못했습니다. 브라우저의 마이크 권한을 확인해 주세요.");
  } finally {
    microphoneStarting = false;
    updateControls();
  }
});
$("record-stop").onclick=()=>{
  const pending = microphoneStarting;
  stopRecording();
  if (pending) notice("마이크 시작을 취소했습니다.", false);
};
$("rate").onchange=()=>action("preferences",{speech_rate:Number($("rate").value)},"읽기 속도를 변경했습니다.");
for(const direction of ["previous","next"]) $(direction).onclick=()=>action("move",{direction},"항목을 이동했습니다.");
$("propose").onclick=async()=>{
  if(actionBusy || !group()) return;
  const result=await action("select",{group_id:group().id,option_ids:[...document.querySelectorAll('#choices input:checked')].map(x=>x.value)});
  if(result)notice($("selection").textContent);
};
$("confirm").onclick=()=>{if(!group())return;return action("confirm",{group_id:group().id},"선택을 확정했습니다. 작성 후 다시 촬영해서 결과를 확인해주세요.");};
$("cancel").onclick=()=>{if(!group())return;return action("cancel",{group_id:group().id},"변경 중인 선택을 취소했습니다.");};
$("verify").onclick=async()=>{
  if(actionBusy)return;
  stopCamera();
  const result=await action("verify");
  if(result?.mode === "ocr") $("results").textContent = result.instruction;
  else if(result) notice($("results").innerText);
};
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
function showPanel(name) {
  for(const id of ["read", "question", "checks"]) $(`${id}-panel`).hidden = id !== name;
  $(name === "checks" ? "verify-title" : `${name}-title`).focus();
}
$("open-read").onclick=()=>{showPanel("read");return $("read-page").onclick();};
$("open-question").onclick=()=>{showPanel("question");notice("마이크로 질문 말하기 또는 질문 입력을 사용할 수 있습니다.");};
$("open-checks").onclick=()=>{showPanel("checks");return $("verify").onclick();};
$("new-capture").onclick=()=>$("camera").onclick();
$("replay").onclick=()=>speak($("notice").textContent);
$("auto-capture").onchange=()=>{
  $("auto-status").textContent = $("auto-capture").checked ? "자동 촬영 켜짐" : "자동 촬영 꺼짐 · 지금 촬영 버튼을 이용하세요";
  notice($("auto-status").textContent);
};
$("test-beep").onclick=async()=>{
  audio.stopSpeech();
  $("beep-enabled").checked=true;audio.setEnabled(true);
  try {
    const started=await audio.start();
    $("audio-status").textContent=started ? "확인음을 재생했습니다. 들리지 않으면 노트북 출력 장치와 음량을 확인해 주세요." : "이 브라우저에서 비프음을 시작할 수 없습니다.";
  } catch (_) {$("audio-status").textContent="오디오를 시작하지 못했습니다. 브라우저의 소리 권한을 확인해 주세요.";}
};
document.addEventListener("keydown", event=>{
  if(event.key === "Escape") {audio.stopSpeech();if(microphone) stopRecording(true);notice("음성을 멈췄습니다.",false);}
});
window.addEventListener("pagehide",()=>{stopCamera();stopRecording(true);audio.stopSpeech();});
document.addEventListener("visibilitychange",()=>{
  if (document.hidden) {stopCamera();stopRecording(true);audio.stopSpeech();}
});
if(window.speechSynthesis){speechSynthesis.onvoiceschanged=voices;voices();}
updateControls();
(async()=>{try{render(await api("state"));if(state.mode !== "ocr") {$("reference").src=(await api("reference")).image;$("reference").hidden=false;}notice(state.document_ready ? "확인한 문서가 있습니다. 원문을 듣거나 질문할 수 있습니다." : "문서를 촬영하거나 이미지 파일을 불러오세요.",false);}catch(error){notice(error.message);}})();
