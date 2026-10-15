import json
import subprocess

from docagent import doctor


def test_keys_never_appear_and_doctor_makes_no_network_calls(monkeypatch,capsys):
    monkeypatch.setenv('ROBOFLOW_API_KEY','SYNTHETIC_SECRET_DO_NOT_PRINT')
    monkeypatch.setattr(doctor,'_load_dotenv',lambda:None)
    monkeypatch.setattr(doctor.shutil,'which',lambda name:'/fake/'+name)
    monkeypatch.setattr(doctor,'probe',lambda command:'List of available languages\nkor\neng\nYuna ko_KR sample\n')
    assert doctor.main(['--profile','core'])==0
    result=capsys.readouterr().out
    assert 'SYNTHETIC_SECRET_DO_NOT_PRINT' not in result
    assert json.loads(result)['checks']['api_key']['available'] is True


def test_missing_language_and_tts_are_profile_specific(monkeypatch):
    monkeypatch.setattr(doctor.shutil,'which',lambda name:None)
    core=doctor.inspect_environment(profile='core')
    ocr=doctor.inspect_environment(profile='ocr')
    assert core['status']=='ready'
    assert {'kor','eng','tesseract'} <= set(ocr['missing'])
    assert doctor.inspect_environment(profile='tts')['checks']['local_tts']['available'] is False


def test_probe_timeout_does_not_expose_command_output(monkeypatch):
    def fail(*a,**kw): raise subprocess.TimeoutExpired('private-command',5,output='secret-output')
    monkeypatch.setattr(doctor.subprocess,'run',fail)
    assert doctor.probe(['fake']) is None
