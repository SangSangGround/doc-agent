"""v8 generic dialogue controller (pure state machine, no camera/audio I/O).

Input : LinkedWorkflow (workflow_link_v8) + utterances / keyboard intents / arrival events.
Output: list of Action('say'|'guide'|'stop_guide'|'retry'|'finished').

Document-agnostic: every document-specific word comes from the workflow JSON.
Safety:
  * 'guide' is emitted ONLY for fields whose position was reviewed (nav_status verified).
  * office-only fields never enter the plan (linker).
  * AI-inferred conditions are spoken with hedging ('...로 보입니다').
  * '완료' is recorded as user-reported; only a verifier result may say '확인했습니다'.

States: DOCUMENT_INTRO -> SELECT_APPLICATION_TYPE -> EXPLAIN_APPLICATION -> WAIT_FOR_CONSENT
 -> EXPLAIN_FIELD -> (CHOOSE_OPTION) -> ASK_TO_NAVIGATE | FIELD_UNAVAILABLE -> TRACKING -> ARRIVED
 -> WAIT_FOR_WRITING -> VERIFY_WRITING -> NEXT_FIELD -> ... -> DOCUMENT_FINISHED, plus PAUSED.
"""
from __future__ import annotations

import re
import time
from dataclasses import dataclass

from voice_intents_v8 import Intent, parse_intent

DOCUMENT_INTRO = 'DOCUMENT_INTRO'
SELECT_APPLICATION_TYPE = 'SELECT_APPLICATION_TYPE'
EXPLAIN_APPLICATION = 'EXPLAIN_APPLICATION'
WAIT_FOR_CONSENT = 'WAIT_FOR_CONSENT'
EXPLAIN_FIELD = 'EXPLAIN_FIELD'
CHOOSE_OPTION = 'CHOOSE_OPTION'
ASK_TO_NAVIGATE = 'ASK_TO_NAVIGATE'
FIELD_UNAVAILABLE = 'FIELD_UNAVAILABLE'
TRACKING = 'TRACKING'
ARRIVED = 'ARRIVED'
WAIT_FOR_WRITING = 'WAIT_FOR_WRITING'
VERIFY_WRITING = 'VERIFY_WRITING'
NEXT_FIELD = 'NEXT_FIELD'
PAUSED = 'PAUSED'
DOCUMENT_FINISHED = 'DOCUMENT_FINISHED'

FIELD_STATES = (EXPLAIN_FIELD, CHOOSE_OPTION, ASK_TO_NAVIGATE, FIELD_UNAVAILABLE, TRACKING,
                ARRIVED, WAIT_FOR_WRITING, VERIFY_WRITING, DOCUMENT_FINISHED)

# Field result statuses
ST_DONE = 'user_reported_done'          # user said 완료; NOT camera-verified
ST_VERIFIED = 'camera_verified'         # only via verifier hook
ST_SKIPPED = 'skipped_by_user'
ST_NOT_APPLICABLE = 'not_applicable'
ST_NOT_SELECTED = 'not_selected'        # other options of a chosen single-choice group
ST_UNGUIDED = 'unguided_unverified'     # explained only: position not reviewed
ST_UNCONFIRMED = 'left_unconfirmed'     # moved on without saying 완료

KIND = {'check_box': '체크박스', 'input_field': '작성란', 'signature_field': '서명란'}
ARRIVAL_TEXT = {
    'check_box': "목표 위치에 도달했습니다. 해당 칸에 체크 표시를 해 주세요. 표시하셨다면 '완료'라고 말씀해 주세요.",
    'signature_field': "목표 위치에 도달했습니다. 해당 위치에 서명하거나 도장을 찍어 주세요. 마치셨다면 '완료'라고 말씀해 주세요.",
    'input_field': "목표 위치에 도달했습니다. 해당 위치에 작성해 주세요. 작성하셨다면 '완료'라고 말씀해 주세요.",
}

_UNITS = ['', '한', '두', '세', '네', '다섯', '여섯', '일곱', '여덟', '아홉']
_TENS = ['', '열', '스물', '서른', '마흔', '쉰', '예순', '일흔', '여든', '아흔']


def native_number(n):
    """Korean native counter form (한, 두, ..., 열한, 스무, 스물한)."""
    if not 1 <= n <= 99:
        return str(n)
    t, u = divmod(n, 10)
    if t == 2 and u == 0:
        return '스무'
    return _TENS[t] + _UNITS[u]


def ordinal(n):
    return '첫 번째' if n == 1 else f'{native_number(n)} 번째'


def count(n, unit='개'):
    return f'{native_number(n)} {unit}' if n else '없음'


def josa(word, with_final, without_final):
    """Pick a particle by the final consonant of the last Hangul syllable."""
    w = re.sub(r'[^가-힣0-9]', '', str(word))
    if not w:
        return without_final
    c = w[-1]
    if '가' <= c <= '힣':
        return with_final if (ord(c) - 0xAC00) % 28 else without_final
    return with_final if c in '0136780' else without_final


def speak(text):
    """Clean printed form text for TTS (brackets, middle dots, check marks)."""
    t = str(text or '')
    t = re.sub(r'[「」『』\[\]□■※√✓]', ' ', t)
    t = re.sub(r'[·ㆍ•]', ', ', t)
    t = t.replace('(', ' ').replace(')', ' ')
    t = re.sub(r'\s+([,.])', r'\1', t)
    return re.sub(r'\s+', ' ', t).strip()


def names_list(names, limit=4):
    names = [speak(n) for n in names]
    if len(names) <= limit:
        return ', '.join(names)
    return ', '.join(names[:limit]) + f' 등 {count(len(names))}'


@dataclass
class Action:
    kind: str                   # say | guide | stop_guide | retry | finished
    text: str = ''
    field_id: str | None = None


class DialogueController:
    def __init__(self, linked, verifier=None, clock=time.monotonic):
        self.wf = linked
        self.verifier = verifier      # callable(field_id) -> 'verified' | 'not_found' | None
        self.clock = clock
        self.state = DOCUMENT_INTRO
        self.type_id = None
        self.plan = []
        self.index = 0
        self.option_choice = {}
        self.status = {}
        self.paused_from = None
        self.guiding = None
        self.transitions = []
        self.log = []

    # ------------------------------------------------------------ helpers
    def _set(self, state):
        if state != self.state:
            self.transitions.append((self.state, state))
        self.state = state

    @property
    def current_type(self):
        return self.wf.type_by_id(self.type_id) if self.type_id else None

    def current_step(self):
        return self.plan[self.index] if self.plan and 0 <= self.index < len(self.plan) else None

    def current_field_id(self):
        step = self.current_step()
        if step is None:
            return None
        if step.kind == 'field':
            return step.field_ids[0]
        return self.option_choice.get(self.index)

    def option_aliases(self):
        """Alias lists for the choice context of the current state (for parse_intent)."""
        if self.state in (SELECT_APPLICATION_TYPE, WAIT_FOR_CONSENT) and self.wf.app_types:
            return [t.aliases for t in self.wf.app_types]
        if self.state == CHOOSE_OPTION:
            return [self.wf.fields[f].aliases for f in self.current_step().field_ids]
        return None

    def _field_names(self, ids):
        return [self.wf.fields[f].name for f in ids]

    # ------------------------------------------------------------ public API
    def start(self):
        out = []
        out.append(Action('say', self._intro_text()))
        types = self.wf.app_types
        if len(types) > 1:
            self._ask_type(out, detailed=False)
        elif len(types) == 1:
            self._choose_type(0, out)
        else:
            self._choose_type(None, out)
        return out

    def handle_text(self, text):
        intent = parse_intent(text, self.option_aliases())
        self.log.append((self.clock(), self.state, str(text), intent.kind if intent else None))
        return self.handle_intent(intent)

    def handle_intent(self, intent):
        out = []
        if intent is None:
            return [Action('retry')]
        if self._global(intent, out):
            return out
        handler = getattr(self, '_on_' + self.state.lower(), None)
        if handler is None:
            return [Action('retry')]
        handler(intent, out)
        return out

    def on_arrived(self, field_id):
        """Runtime reports the finger reached the guided target (proximity, not contact)."""
        if self.state != TRACKING or field_id != self.guiding:
            return []
        out = [Action('stop_guide', field_id=field_id)]
        self.guiding = None
        self._set(ARRIVED)
        out.append(Action('say', ARRIVAL_TEXT.get(self.wf.fields[field_id].class_name, ARRIVAL_TEXT['input_field'])))
        self._set(WAIT_FOR_WRITING)
        return out

    def results(self):
        rows = []
        for i, step in enumerate(self.plan):
            for fid in step.field_ids:
                f = self.wf.fields[fid]
                rows.append({'step': i + 1, 'field_id': fid, 'name': f.name, 'class_name': f.class_name,
                             'nav_status': f.nav_status, 'status': self.status.get(fid, 'pending')})
        return {'document_id': self.wf.document_id, 'application_type': self.type_id, 'state': self.state,
                'fields': rows, 'transitions': self.transitions,
                'note': 'user_reported_done means the user SAID it was written; no camera verification.'}

    # ------------------------------------------------------------ texts
    def _intro_text(self):
        parts = []
        if self.wf.title:
            parts.append(f'문서를 인식했습니다. 이 문서는 {speak(self.wf.title)}입니다.')
        else:
            parts.append('문서를 인식했지만 제목은 확인하지 못했습니다.')
        if self.wf.description:
            parts.append(speak(self.wf.description))
        if self.wf.ai_inferred and not self.wf.human_reviewed:
            parts.append('문서 설명은 AI가 사진을 읽고 정리한 것이라 틀릴 수 있습니다.')
        return ' '.join(parts)

    def _type_question(self, detailed):
        names = [speak(t.spoken_name) for t in self.wf.app_types]
        q = speak(self.wf.selection_question) or f'{", ".join(names)} 중 어떤 신청을 진행하시겠습니까?'
        if detailed:
            q += ' ' + ' '.join(f'{ordinal(i + 1)}, {n}.' for i, n in enumerate(names))
            q += ' 이름이나 번호로 말씀해 주세요.'
        return q

    def _application_text(self):
        t = self.current_type
        steps = self.plan
        nav = sum(any(self.wf.fields[f].navigable for f in s.field_ids) for s in steps)
        parts = []
        if t:
            name = speak(t.spoken_name)
            parts.append(f'{name} 신청을 선택하셨습니다.')
            if t.description:
                parts.append(speak(t.description))
            titles = []
            plan_ids = {f for s in steps for f in s.field_ids}
            for sid in t.section_ids:
                sec = self.wf.sections[sid]
                if plan_ids.intersection(sec['field_ids']) and sec.get('title'):
                    titles.append(speak(sec['title']))
            if titles:
                parts.append(f'{name} 신청에는 {names_list(titles)} 등의 작성 항목이 있습니다.')
        parts.append(f'안내할 항목은 모두 {count(len(steps))}입니다.')
        if nav:
            parts.append(f'이 중 손가락 위치 안내가 가능한 항목은 {count(nav)}입니다.')
        else:
            parts.append('이 중 위치 검수가 끝난 항목이 없어 손가락 위치 안내는 할 수 없고, 항목 설명만 해 드립니다.')
        if self.wf.excluded_official():
            parts.append('기관 담당자가 작성하는 칸은 안내하지 않습니다.')
        parts.append('각 항목을 순서대로 안내해 드릴까요?')
        return ' '.join(parts)

    def _field_text(self, fid, detailed=False, with_ordinal=True):
        f = self.wf.fields[fid]
        kind = KIND.get(f.class_name, '입력 항목')
        head = f'{speak(f.name)}, {kind}입니다.'
        if with_ordinal:
            head = f'{ordinal(self.index + 1)} 항목은 ' + head
        parts = [head]
        if f.description:
            parts.append(speak(f.description))
        if f.requirement == 'conditional' and f.condition_text:
            parts.append(f'작성 조건이 있는 항목으로 보입니다. {speak(f.condition_text)}')
        elif f.requirement == 'optional':
            parts.append('선택해서 작성하는 항목으로 보입니다.')
        g = self.wf.groups.get(f.choice_group_id) if f.choice_group_id else None
        if g and g['type'] == 'multi_choice':
            parts.append('해당하는 경우에만 표시하는 선택지로 보입니다.')
        if f.role == 'third_party':
            parts.append('신청인이 아닌 다른 사람이 작성하거나 서명하는 칸일 수 있습니다.')
        if f.uncertain or f.confidence == 'low':
            parts.append('이 항목의 의미는 확실하지 않으니 필요하면 다른 분께 확인해 주세요.')
        if detailed and f.source_quote:
            parts.append(f'문서에 인쇄된 관련 문구는 다음과 같습니다. {speak(f.source_quote)}')
        return ' '.join(parts)

    def _nav_question(self, fid):
        f = self.wf.fields[fid]
        if f.navigable:
            if f.class_name == 'check_box' and f.choice_group_id and \
                    self.wf.groups.get(f.choice_group_id, {}).get('type') == 'multi_choice':
                return "해당하시면 작성 위치로 안내하겠습니다. 이동하시겠습니까? 해당하지 않으면 '아니요'라고 말씀해 주세요."
            return '해당 항목의 작성 위치로 이동하시겠습니까?'
        return ("이 항목은 작성 위치 검수가 끝나지 않아 손가락 안내를 할 수 없습니다. "
                "다음 항목으로 넘어갈까요?")

    def _choice_text(self, detailed=False):
        step = self.current_step()
        q = speak(step.question) or '다음 중 해당하는 하나를 고르는 항목입니다.'
        opts = ' '.join(f'{ordinal(i + 1)}, {speak(self.wf.fields[f].name)}.' for i, f in enumerate(step.field_ids))
        text = f'{ordinal(self.index + 1)} 항목은 하나만 고르는 체크박스입니다. {q} {opts} 해당하는 번호를 말씀해 주세요.'
        text += " 해당하는 것이 없으면 '건너뛰기'라고 말씀해 주세요."
        if detailed:
            quotes = [self.wf.fields[f].source_quote for f in step.field_ids if self.wf.fields[f].source_quote]
            if quotes:
                text += f' 문서에 인쇄된 관련 문구는 다음과 같습니다. {speak(quotes[0])}'
        return text

    def _summary_text(self):
        names = {k: [] for k in (ST_DONE, ST_VERIFIED, ST_SKIPPED, ST_UNGUIDED, ST_UNCONFIRMED, 'pending')}
        for i, step in enumerate(self.plan):
            if step.kind == 'choice':
                chosen = self.option_choice.get(i)
                if chosen is None:
                    st = self.status.get(step.field_ids[0], 'pending')
                    if st == ST_NOT_APPLICABLE:
                        continue
                    names.setdefault(st, []).append(step.question or self.wf.fields[step.field_ids[0]].name)
                    continue
                ids = [chosen]
            else:
                ids = step.field_ids
            for fid in ids:
                st = self.status.get(fid, 'pending')
                names.setdefault(st, []).append(self.wf.fields[fid].name)
        done = len(names[ST_DONE]) + len(names[ST_VERIFIED])
        parts = ['모든 항목 안내를 마쳤습니다.']
        parts.append(f'작성 완료로 기록한 항목은 {count(done)}입니다.' if done else '작성 완료로 기록한 항목은 없습니다.')
        if names[ST_SKIPPED]:
            parts.append(f'건너뛴 항목은 {names_list(names[ST_SKIPPED])}입니다.')
        if names[ST_UNGUIDED]:
            parts.append(f'위치 확인이 안 되어 손가락 안내를 하지 못한 항목은 {names_list(names[ST_UNGUIDED])}입니다.')
        if names[ST_UNCONFIRMED]:
            parts.append(f'작성 완료 여부를 확인하지 못한 항목은 {names_list(names[ST_UNCONFIRMED])}입니다.')
        if names['pending']:
            parts.append(f'아직 안내하지 않은 항목은 {names_list(names["pending"])}입니다.')
        if names[ST_DONE] and not names[ST_VERIFIED]:
            parts.append('작성 완료는 말씀하신 내용으로 기록한 것이며, 카메라로 작성 결과를 확인하지는 않았습니다.')
        parts.append("이전 항목으로 돌아가려면 '이전', 다른 신청을 하려면 '다른 신청'이라고 말씀해 주세요.")
        return ' '.join(parts)

    # ------------------------------------------------------------ transitions
    def _ask_type(self, out, detailed):
        self._stop_guiding(out)
        self._set(SELECT_APPLICATION_TYPE)
        out.append(Action('say', self._type_question(detailed)))

    def _choose_type(self, idx, out):
        self._stop_guiding(out)
        t = self.wf.app_types[idx] if idx is not None else None
        plan = self.wf.plan_for(t.id if t else None)
        if not plan:
            out.append(Action('say', '선택한 신청에 안내할 수 있는 작성 항목이 없습니다.'))
            if self.wf.app_types:
                self._ask_type(out, detailed=True)
            else:
                self._set(DOCUMENT_FINISHED)
                out.append(Action('finished'))
            return
        self.type_id = t.id if t else None
        self.plan, self.index = plan, 0
        self.option_choice, self.status = {}, {}
        self._set(EXPLAIN_APPLICATION)
        out.append(Action('say', self._application_text()))
        self._set(WAIT_FOR_CONSENT)

    def _explain_step(self, out, prefix=''):
        step = self.current_step()
        self._set(EXPLAIN_FIELD)
        if step.kind == 'choice':
            self.option_choice.pop(self.index, None)
            self._set(CHOOSE_OPTION)
            out.append(Action('say', (prefix + ' ' + self._choice_text()).strip()))
            return
        fid = step.field_ids[0]
        self._after_field_selected(fid, out, prefix + ' ' + self._field_text(fid))

    def _after_field_selected(self, fid, out, text):
        f = self.wf.fields[fid]
        self._set(ASK_TO_NAVIGATE if f.navigable else FIELD_UNAVAILABLE)
        out.append(Action('say', (text.strip() + ' ' + self._nav_question(fid)).strip()))

    def _start_guiding(self, out, fid, text='작성 위치 안내를 시작합니다. 종이와 오른손 검지가 카메라에 보이게 해 주세요.'):
        f = self.wf.fields[fid]
        if not f.navigable:          # defensive: never guide to an unreviewed point
            self._set(FIELD_UNAVAILABLE)
            out.append(Action('say', self._nav_question(fid)))
            return
        self.guiding = fid
        self._set(TRACKING)
        out.append(Action('say', text))
        out.append(Action('guide', field_id=fid))

    def _stop_guiding(self, out):
        if self.guiding is not None:
            out.append(Action('stop_guide', field_id=self.guiding))
            self.guiding = None

    def _mark(self, fid, status):
        self.status[fid] = status

    def _advance(self, out, prefix=''):
        self._stop_guiding(out)
        self._set(NEXT_FIELD)
        self.index += 1
        if self.index >= len(self.plan):
            self.index = len(self.plan)
            self._set(DOCUMENT_FINISHED)
            out.append(Action('say', (prefix + ' ' + self._summary_text()).strip()))
            out.append(Action('finished'))
            return
        self._explain_step(out, (prefix + ' 다음 항목을 안내하겠습니다.').strip())

    def _verify(self, out, fid):
        self._set(VERIFY_WRITING)
        result = None
        if self.verifier is not None:
            try:
                result = self.verifier(fid)
            except Exception as exc:     # a verifier failure must never block the dialogue
                print('[VERIFY ERROR]', repr(exc), flush=True)
        if result == 'verified':
            self._mark(fid, ST_VERIFIED)
            msg = '표시된 것으로 확인했습니다.'
        elif result == 'not_found':
            self._mark(fid, ST_DONE)
            msg = '카메라로는 표시를 확인하지 못했습니다. 작성 완료로만 기록합니다.'
        else:
            self._mark(fid, ST_DONE)
            msg = '작성 완료로 기록했습니다.'
        self._advance(out, msg)

    def _prompt(self, out, detailed=True):
        s = self.state
        if s == DOCUMENT_INTRO:
            out.append(Action('say', self._intro_text()))
        elif s == SELECT_APPLICATION_TYPE:
            out.append(Action('say', self._type_question(detailed=True)))
        elif s == WAIT_FOR_CONSENT:
            out.append(Action('say', self._application_text()))
        elif s == CHOOSE_OPTION:
            out.append(Action('say', self._choice_text(detailed)))
        elif s in (ASK_TO_NAVIGATE, FIELD_UNAVAILABLE):
            fid = self.current_field_id()
            out.append(Action('say', self._field_text(fid, detailed) + ' ' + self._nav_question(fid)))
        elif s == TRACKING:
            out.append(Action('say', f'{speak(self.wf.fields[self.guiding].name)} 위치로 안내하고 있습니다. '
                                     '오른손 검지를 종이 위에 올려 주세요.'))
        elif s == WAIT_FOR_WRITING:
            fid = self.current_field_id()
            out.append(Action('say', ARRIVAL_TEXT.get(self.wf.fields[fid].class_name, ARRIVAL_TEXT['input_field'])))
        elif s == DOCUMENT_FINISHED:
            out.append(Action('say', self._summary_text()))
        elif s == PAUSED:
            out.append(Action('say', "안내가 일시 중지되어 있습니다. '계속'이라고 말씀하시면 이어서 안내합니다."))

    # ------------------------------------------------------------ global commands
    def _global(self, intent, out):
        k = intent.kind
        if self.state == PAUSED:
            if k in ('resume', 'yes'):
                prev, self.paused_from = self.paused_from, None
                self._set(prev)
                if prev == TRACKING and self.current_field_id():
                    self._start_guiding(out, self.current_field_id(), '안내를 재개합니다.')
                else:
                    out.append(Action('say', '안내를 재개합니다.'))
                    self._prompt(out, detailed=False)
            else:
                self._prompt(out)
            return True
        if k == 'pause':
            self._stop_guiding(out)
            self.paused_from = self.state
            self._set(PAUSED)
            out.append(Action('say', "안내를 일시 중지합니다. 다시 시작하려면 '계속'이라고 말씀해 주세요."))
            return True
        if k == 'overview':
            out.append(Action('say', self._intro_text()))
            if self.state != TRACKING:
                self._prompt(out, detailed=False)
            return True
        if k == 'repeat':
            self._prompt(out, detailed=True)
            return True
        if k == 'change_type':
            if self.wf.app_types:
                self._ask_type(out, detailed=True)
            else:
                out.append(Action('say', '이 문서에는 고를 수 있는 신청 유형이 없습니다.'))
            return True
        if k == 'prev' and self.state in FIELD_STATES and self.plan:
            self._stop_guiding(out)
            if self.index <= 0:
                self.index = 0
                self._explain_step(out, '첫 번째 항목입니다.')
            else:
                self.index = min(self.index, len(self.plan)) - 1
                self._explain_step(out, '이전 항목으로 돌아갑니다.')
            return True
        return False

    # ------------------------------------------------------------ state handlers
    def _on_select_application_type(self, intent, out):
        if intent.kind == 'choice':
            self._choose_type(intent.value, out)
        else:
            names = ', '.join(speak(t.spoken_name) for t in self.wf.app_types)
            out.append(Action('say', f'{names} 중에서 하나를 말씀해 주세요. 번호로 말씀하셔도 됩니다.'))

    def _on_wait_for_consent(self, intent, out):
        k = intent.kind
        if k in ('yes', 'resume', 'next'):
            self.index = 0
            self._explain_step(out)
        elif k == 'choice':
            self._choose_type(intent.value, out)
        elif k == 'no':
            if self.wf.app_types:
                self._set(SELECT_APPLICATION_TYPE)
                out.append(Action('say', "알겠습니다. 다른 신청을 하시려면 신청 유형 이름을 말씀해 주세요. "
                                         "안내를 멈추려면 '중단'이라고 말씀해 주세요."))
            else:
                out.append(Action('say', "알겠습니다. 준비되시면 '예'라고 말씀해 주세요."))
        else:
            out.append(Action('say', "항목을 순서대로 안내해 드릴까요? '예' 또는 '아니요'로 답해 주세요."))

    def _on_choose_option(self, intent, out):
        step = self.current_step()
        k = intent.kind
        if k == 'choice' and 0 <= intent.value < len(step.field_ids):
            fid = step.field_ids[intent.value]
            self.option_choice[self.index] = fid
            for other in step.field_ids:
                if other != fid:
                    self._mark(other, ST_NOT_SELECTED)
            name = speak(self.wf.fields[fid].name)
            self._after_field_selected(fid, out, f'{name}{josa(name, "을", "를")} 선택하셨습니다.')
        elif k in ('skip', 'no', 'next'):
            for fid in step.field_ids:
                self._mark(fid, ST_NOT_APPLICABLE)
            self._advance(out, '이 항목은 해당 없음으로 넘어갑니다.')
        else:
            out.append(Action('say', "해당하는 번호를 말씀해 주세요. 예를 들어 '첫 번째'라고 말씀하시면 됩니다."))

    def _on_ask_to_navigate(self, intent, out):
        fid = self.current_field_id()
        k = intent.kind
        if k in ('yes', 'resume', 'position'):
            self._start_guiding(out, fid)
        elif k in ('no', 'skip', 'next'):
            f = self.wf.fields[fid]
            multi = f.choice_group_id and self.wf.groups.get(f.choice_group_id, {}).get('type') == 'multi_choice'
            if k == 'no' and multi:
                self._mark(fid, ST_NOT_APPLICABLE)
                self._advance(out, '해당하지 않는 것으로 넘어갑니다.')
            else:
                self._mark(fid, ST_SKIPPED)
                self._advance(out, '이 항목을 건너뜁니다.')
        elif k == 'done':
            self._verify(out, fid)
        else:
            out.append(Action('say', "작성 위치로 안내할까요? '예' 또는 '아니요'로 답해 주세요."))

    def _on_field_unavailable(self, intent, out):
        fid = self.current_field_id()
        k = intent.kind
        if k in ('yes', 'next', 'skip', 'resume'):
            self._mark(fid, ST_UNGUIDED)
            self._advance(out)
        elif k == 'done':
            self._verify(out, fid)
        elif k == 'position':
            out.append(Action('say', "이 항목은 위치 검수가 끝나지 않아 위치를 안내할 수 없습니다. 넘어가려면 '다음'이라고 말씀해 주세요."))
        elif k == 'no':
            out.append(Action('say', "알겠습니다. 이 항목은 다른 분의 도움을 받아 작성하시고, 준비되면 '다음'이라고 말씀해 주세요."))
        else:
            out.append(Action('say', "다음 항목으로 넘어가려면 '다음' 또는 '예'라고 말씀해 주세요."))

    def _on_tracking(self, intent, out):
        fid = self.guiding
        k = intent.kind
        if k == 'done':
            self._stop_guiding(out)
            self._verify(out, fid)
        elif k in ('skip', 'next'):
            self._mark(fid, ST_SKIPPED)
            self._advance(out, '위치 안내를 멈추고 이 항목을 건너뜁니다.')
        elif k == 'no':
            self._stop_guiding(out)
            self._set(ASK_TO_NAVIGATE)
            out.append(Action('say', "위치 안내를 멈췄습니다. 다시 안내받으려면 '예', 건너뛰려면 '건너뛰기'라고 말씀해 주세요."))
        elif k in ('position', 'yes'):
            self._prompt(out)
        else:
            out.append(Action('say', "위치 안내 중입니다. 작성하셨다면 '완료', 멈추려면 '중단'이라고 말씀해 주세요."))

    def _on_wait_for_writing(self, intent, out):
        fid = self.current_field_id()
        k = intent.kind
        if k in ('done', 'yes'):
            self._verify(out, fid)
        elif k == 'next':
            self._mark(fid, ST_UNCONFIRMED)
            self._advance(out, '작성 완료 여부를 확인하지 못한 항목으로 남겨 둡니다.')
        elif k == 'skip':
            self._mark(fid, ST_SKIPPED)
            self._advance(out, '이 항목을 건너뜁니다.')
        elif k == 'position':
            self._start_guiding(out, fid, '다시 위치를 안내합니다.')
        else:
            out.append(Action('say', "작성을 마치시면 '완료'라고 말씀해 주세요. 위치를 다시 찾으려면 '위치 알려줘'라고 말씀해 주세요."))

    def _on_document_finished(self, intent, out):
        out.append(Action('say', "안내를 마쳤습니다. 이전 항목으로 돌아가려면 '이전', 다른 신청을 하려면 '다른 신청'이라고 말씀해 주세요."))
