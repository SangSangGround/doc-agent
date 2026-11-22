/* node tests/home/audio_check.cjs — 비프음 ducking·발화 경쟁·오래된 프레임 검사 */
const assert=require('node:assert/strict'),fs=require('node:fs'),path=require('node:path'),vm=require('node:vm');
let now=1000,id=0;const timers=new Map(),utterances=[],oscillators=[],gains=[];
class Context {
  constructor(){this.currentTime=1;this.destination={};}
  resume(){return Promise.resolve();}
  createGain(){
    const values=[];const node={connect(){},disconnect(){},gain:{values,setTargetAtTime(v){values.push(v);},setValueAtTime(){},linearRampToValueAtTime(){}}};
    gains.push(node);return node;
  }
  createOscillator(){const o={frequency:{},connect(){},disconnect(){},start(){},stop(){this.stops=(this.stops||0)+1;}};oscillators.push(o);return o;}
}
const window={AudioContext:Context,speechSynthesis:{cancel(){},speak(u){utterances.push(u);}}};
const context=vm.createContext({window,SpeechSynthesisUtterance:class {constructor(text){this.text=text;}},Date:{now:()=>now},
  setTimeout(fn){timers.set(++id,fn);return id;},clearTimeout(i){timers.delete(i);}});
vm.runInContext(fs.readFileSync(path.join(__dirname,'../../src/docagent/home/assets/audio.js'),'utf8'),context);
(async()=>{
  const audio=new window.GuidanceAudio(),voice={lang:'ko-KR'};await audio.start();
  const volume=()=>gains[0].gain.values.at(-1);
  assert.equal(oscillators.length,1);assert.equal(volume(),.30);
  audio.speak('첫 안내',voice,1);assert.equal(volume(),.07);
  audio.speak('새 안내',voice,1);utterances[0].onend();assert.equal(volume(),.07);
  utterances[1].onerror();assert.equal(volume(),.30);
  audio.speak('취소할 안내',voice,1);audio.stopSpeech();assert.equal(volume(),.30);
  audio.setEnabled(false);audio.speak('음성만',voice,1);utterances.at(-1).onend();assert.equal(volume(),0);
  audio.setEnabled(true);audio.update('ready');assert.equal(oscillators.length,2);
  assert.equal(oscillators[1].frequency.value,1100);
  now+=3000;audio.schedule();assert.equal(oscillators.length,2);
  audio.stop();assert.equal(timers.size,0);assert.equal(audio.active,false);assert.equal(oscillators[0].stops,2);
  console.log('PASS beep ducking, stale speech callbacks, mute, stale frames and stop');
})().catch(error=>{console.error(error);process.exitCode=1;});
