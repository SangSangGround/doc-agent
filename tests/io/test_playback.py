from dataclasses import replace
from pathlib import Path

import pytest

from docagent import guide
from docagent.errors import ToolExecutionError
from docagent.guidance.rules import FALLBACK, Observation, RuleCatalog
from docagent.io.playback import GuidancePlayer, SayBackend

CATALOG=Path(__file__).resolve().parents[2]/'examples/guidance/catalog.demo.json'
OBS=Path(__file__).resolve().parents[2]/'examples/guidance/observation.demo.json'


def observation():
    import json
    return Observation.from_dict(json.loads(OBS.read_text(encoding='utf-8')))


class Recorder:
    def __init__(self): self.spoken=[]; self.state='idle'; self.stops=0
    def play(self,text): self.spoken.append(text); self.state='playing'
    def stop(self): self.stops+=1; self.state='stopped'
    def wait(self): self.state='completed'


def test_exact_play_stop_and_replay():
    catalog=RuleCatalog.load(CATALOG); backend=Recorder(); player=GuidancePlayer(catalog,backend)
    with pytest.raises(ToolExecutionError): player.replay()
    result=player.play(observation(),demo=True)
    player.stop();assert backend.state=='stopped'
    assert player.replay()==result
    player.wait()
    assert backend.state=='completed'
    assert backend.spoken==[result.instruction,result.instruction]


def test_failure_speaks_only_fixed_fallback_and_replaces_previous():
    backend=Recorder();player=GuidancePlayer(RuleCatalog.load(CATALOG),backend)
    player.play(observation(),demo=True)
    result=player.play(replace(observation(),surrounding_text='PII-DO-NOT-SPEAK'),demo=True)
    player.replay()
    assert result.status=='no_match'
    assert backend.spoken[-2:]==[FALLBACK,FALLBACK]
    assert 'PII-DO-NOT-SPEAK' not in ''.join(backend.spoken)


def test_engine_failure_is_not_success(monkeypatch,capsys):
    class Failing(Recorder):
        def wait(self): raise ToolExecutionError('must not expose private details')
    backend=Failing();monkeypatch.setattr(guide,'SayBackend',lambda **kw:backend)
    assert guide.main(['--catalog',str(CATALOG),'--observation',str(OBS),'--demo','--speak'])==1
    assert backend.state=='stopped'
    assert 'private details' not in capsys.readouterr().err


def test_interactive_stop_replay_quit(monkeypatch,capsys):
    backend=Recorder();monkeypatch.setattr(guide,'SayBackend',lambda **kw:backend)
    commands=iter(['s','r','q']);monkeypatch.setattr('builtins.input',lambda:next(commands))
    assert guide.main(['--catalog',str(CATALOG),'--observation',str(OBS),'--demo','--speak','--interactive'])==0
    assert len(backend.spoken)==2
    assert backend.spoken[0]==backend.spoken[1]
    assert backend.state=='stopped'


def test_say_sends_text_as_stdin_and_terminates_only_owned_process(monkeypatch):
    import io
    import docagent.io.playback as playback
    calls=[]
    class Buffer(io.BytesIO):
        def close(self): self.captured=self.getvalue()
    class Process:
        def __init__(self,args,**kw):
            calls.append((args,kw,self));self.stdin=Buffer();self.code=None;self.terminated=False
        def poll(self): return self.code
        def terminate(self): self.terminated=True;self.code=-15
        def wait(self,**kw): self.code=0 if self.code is None else self.code;return self.code
    monkeypatch.setattr(playback.sys,'platform','darwin')
    monkeypatch.setattr(playback.Path,'is_file',lambda p:True)
    monkeypatch.setattr(playback.subprocess,'Popen',Process)
    b=SayBackend()
    sentence='-o /tmp/unwanted; $(do-not-execute)\n승인 문구'
    b.play(sentence)
    assert calls[0][0]==['/usr/bin/say','-v','Yuna']
    assert calls[0][2].stdin.captured==sentence.encode()
    assert 'shell' not in calls[0][1]
    b.stop();assert calls[0][2].terminated
    assert b.state=='stopped'
    b.play('두 번째 문구');b.wait();assert b.state=='completed'


def test_empty_or_truncated_audio_is_not_success(tmp_path):
    import struct
    from docagent.io.playback import _has_audio_frames
    path=tmp_path/'sound.aiff'
    def write(frames,data):
        comm=b'COMM'+struct.pack('>IHI',6,1,frames)
        sound=b'SSND'+struct.pack('>III',len(data)+8,0,0)+data
        body=b'AIFC'+comm+sound
        path.write_bytes(b'FORM'+struct.pack('>I',len(body))+body)
    write(0,b'');assert not _has_audio_frames(path)
    write(2,b'\x01\x02\x03\x04');assert _has_audio_frames(path)
    path.write_bytes(path.read_bytes()[:-2]);assert not _has_audio_frames(path)
