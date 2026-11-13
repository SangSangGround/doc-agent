"use strict";

// 음성 재생 중에는 비프음 전용 GainNode의 볼륨만 낮춘다.
class GuidanceAudio {
  constructor() {
    this.context = null;
    this.output = null;
    this.timer = null;
    this.oscillators = new Set();
    this.active = false;
    this.enabled = true;
    this.speaking = false;
    this.speechRevision = 0;
    this.lastFrameAt = 0;
    this.status = "adjusting";
  }

  async start() {
    const AudioContext = window.AudioContext || window.webkitAudioContext;
    if (!AudioContext) return false;
    if (!this.context) {
      this.context = new AudioContext();
      this.output = this.context.createGain();
      this.output.connect(this.context.destination);
    }
    await this.context.resume();
    this.active = true;
    this.lastFrameAt = 0;
    this.updateVolume();
    this.schedule();
    // 클릭 직후 확인음을 내서 오디오가 시작됐음을 알린다.
    if (this.enabled) this.pulse(880);
    return true;
  }

  updateVolume() {
    if (!this.output) return;
    const volume = this.enabled ? (this.speaking ? 0.07 : 0.30) : 0;
    this.output.gain.setTargetAtTime(volume, this.context.currentTime, 0.03);
  }

  setEnabled(enabled) {
    this.enabled = enabled;
    this.updateVolume();
  }

  update(status) {
    const changed = status !== this.status || !this.lastFrameAt;
    this.status = status;
    this.lastFrameAt = Date.now();
    if (changed && this.active) this.schedule();
  }

  schedule() {
    clearTimeout(this.timer);
    if (!this.active) return;
    const patterns = {
      adjusting: {frequency: 660, interval: 1100},
      stabilizing: {frequency: 880, interval: 400},
      ready: {frequency: 1100, interval: 200},
    };
    const pattern = patterns[this.status] || patterns.adjusting;
    // 영상이 멈추면 이전의 준비 상태를 비프음으로 계속 알리지 않는다.
    if (this.enabled && this.lastFrameAt && Date.now() - this.lastFrameAt < 2500) {
      this.pulse(pattern.frequency);
    }
    this.timer = setTimeout(() => this.schedule(), pattern.interval);
  }

  pulse(frequency) {
    const oscillator = this.context.createOscillator();
    const envelope = this.context.createGain();
    const now = this.context.currentTime;
    oscillator.frequency.value = frequency;
    envelope.gain.setValueAtTime(0, now);
    envelope.gain.linearRampToValueAtTime(1, now + 0.012);
    envelope.gain.linearRampToValueAtTime(0, now + 0.11);
    oscillator.connect(envelope);
    envelope.connect(this.output);
    this.oscillators.add(oscillator);
    oscillator.onended = () => {
      oscillator.disconnect();
      envelope.disconnect();
      this.oscillators.delete(oscillator);
    };
    oscillator.start(now);
    oscillator.stop(now + 0.12);
  }

  speak(text, voice, rate) {
    this.stopSpeech();
    if (!voice || !text || !window.speechSynthesis) return;
    const revision = this.speechRevision;
    const chunks = text.match(/[^\n.!?。]{1,150}(?:[\n.!?。]+|$)|[^\n.!?。]{1,150}/g) || [text];
    this.speaking = true;
    this.updateVolume();
    const next = () => {
      if (revision !== this.speechRevision) return;
      const chunk = chunks.shift();
      if (!chunk) { this.speaking = false; this.updateVolume(); return; }
      const utterance = new SpeechSynthesisUtterance(chunk);
      utterance.voice = voice;
      utterance.lang = voice.lang;
      utterance.rate = rate;
      utterance.onend = next;
      utterance.onerror = () => {
        if (revision !== this.speechRevision) return;
        this.speaking = false; this.updateVolume();
      };
      window.speechSynthesis.speak(utterance);
    };
    next();
  }

  stopSpeech() {
    this.speechRevision++;
    window.speechSynthesis?.cancel();
    this.speaking = false;
    this.updateVolume();
  }

  stop() {
    this.active = false;
    this.lastFrameAt = 0;
    clearTimeout(this.timer);
    this.timer = null;
    for (const oscillator of this.oscillators) oscillator.stop();
  }
}

window.GuidanceAudio = GuidanceAudio;
