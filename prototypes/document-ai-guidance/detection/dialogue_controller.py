"""Pure, I/O-free dialogue state machine. No TTS/STT/camera side effects.

Events are consumed by an adapter that already knows the v7.5 method names.
"""
from __future__ import annotations
from dataclasses import dataclass
from enum import Enum
import re
from workflow_schema import validate_workflow
from navigation_bridge import NavigationRegistry


class State(str, Enum):
    INIT = 'INIT'
    SELECT_TYPE = 'SELECT_APPLICATION_TYPE'
    SELECT_CHOICE = 'SELECT_CHOICE_GROUP'
    WAIT_CONSENT = 'WAIT_FOR_CONSENT'
    ASK_TO_NAVIGATE = 'ASK_TO_NAVIGATE'
    TRACKING = 'TRACKING'
    WAIT_WRITING = 'WAIT_FOR_WRITING'
    FINISHED = 'DOCUMENT_FINISHED'
    PAUSED = 'PAUSED'


@dataclass(frozen=True)
class Event:
    kind: str  # SPEAK, START_TRACKING, STOP_TRACKING, PAUSE_TRACKING, RESUME_TRACKING, FINISHED
    text: str = ''
    field_id: str = ''
    target_id: str = ''


def normalize_voice(text: str) -> str:
    return re.sub(r'[\s.,!?。，！？]+', '', (text or '').strip().lower())


YES = {normalize_voice(v) for v in ('예', '네', '응', '좋아', '진행해', '시작해')}
NO = {normalize_voice(v) for v in ('아니요', '아니', '아뇨', '싫어')}
DONE = {normalize_voice(v) for v in ('완료', '다 썼어', '다썼어', '작성 완료', '작성했어')}
SKIP = {normalize_voice(v) for v in ('건너뛰기', '건너 뛰기', '스킵')}
BACK = {normalize_voice(v) for v in ('이전', '이전 항목', '뒤로')}
REPEAT = {normalize_voice(v) for v in ('다시 설명해줘', '다시 말해줘', '위치 알려줘')}
PAUSE = {normalize_voice(v) for v in ('중단', '일시 중단', '멈춰')}
RESUME = {normalize_voice(v) for v in ('계속', '재개')}
NEXT = {normalize_voice(v) for v in ('다음', '다음 항목')}


class DialogueController:
    def __init__(self, workflow: dict, navigation: NavigationRegistry | None = None):
        self.workflow = validate_workflow(workflow)
        self.navigation = navigation
        self.state = State.INIT
        self._resume_state: State | None = None
        self.selected_type: str | None = None
        self.active_fields: list[dict] = []
        self.index = 0
        self.field_status: dict[str, str] = {f['id']: 'pending' for f in workflow['fields']}
        self.excluded_fields: list[str] = []
        self.pending_choice_groups: list[tuple[str, list[str]]] = []

    @property
    def field(self) -> dict | None:
        if not self.active_fields or self.index >= len(self.active_fields):
            return None
        return self.active_fields[self.index]

    def start(self) -> list[Event]:
        if self.state != State.INIT:
            return [Event('SPEAK', '이미 문서 안내를 시작했습니다.')]
        intro = f"이 문서는 {self.workflow['title']}입니다. {self.workflow['description']}"
        types = self.workflow['application_types']
        if types:
            self.state = State.SELECT_TYPE
            names = ', '.join(app['name'] for app in types)
            return [Event('SPEAK', f'{intro} 진행할 신청 유형을 말씀해주세요. 선택 가능한 유형은 {names}입니다.')]
        self._activate_fields()
        return self._ask_choice_or_consent(intro)

    def _activate_fields(self) -> None:
        chosen = self.selected_type
        matches = [f for f in self.workflow['fields'] if
                   not f['applicable_type_ids'] or (chosen is not None and chosen in f['applicable_type_ids'])]
        self.excluded_fields = [f['id'] for f in matches if f['role'] != 'applicant']
        self.active_fields = [f for f in matches if f['role'] == 'applicant']
        self.index = 0
        present = {f['id'] for f in self.active_fields}
        self.pending_choice_groups = []
        for group in self.workflow['choice_groups']:
            if not group['exclusive']:
                continue
            options = [fid for fid in group['field_ids'] if fid in present]
            if len(options) > 1:
                self.pending_choice_groups.append((group['id'], options))

    def _ask_choice_or_consent(self, preface: str = '') -> list[Event]:
        if self.pending_choice_groups:
            self.state = State.SELECT_CHOICE
            _, options = self.pending_choice_groups[0]
            options_text = ', '.join(f['name'] for f in self.active_fields if f['id'] in options)
            return [Event('SPEAK', f'{preface} 문서 분석상 한 가지를 고르는 선택란이 있습니다. 원문 검수 후 이용해주세요. {options_text} 중 어느 항목을 선택하시겠습니까?')]
        self.state = State.WAIT_CONSENT
        names = ', '.join(f['name'] for f in self.active_fields[:6])
        suffix = ' 등' if len(self.active_fields) > 6 else ''
        return [Event('SPEAK', f'{preface} 신청인 작성 항목 후보는 {len(self.active_fields)}개입니다. {names}{suffix}. 순서대로 설명해 드릴까요?')]

    def _explain_field(self) -> list[Event]:
        f = self.field
        if f is None:
            self.state = State.FINISHED
            count_done = sum(v == 'user_claimed' for v in self.field_status.values())
            count_skip = sum(v == 'skipped' for v in self.field_status.values())
            count_unverified = sum(self.field_status[f['id']] in ('pending', 'unverified') for f in self.active_fields)
            return [Event('SPEAK', f'안내 순서가 끝났습니다. 사용자 완료 보고 {count_done}개, 건너뜀 {count_skip}개, 아직 확인되지 않은 항목 {count_unverified}개입니다. 실제 필기 상태를 사진으로 검증한 결과는 아닙니다.'),
                    Event('FINISHED')]
        self.state = State.ASK_TO_NAVIGATE
        number = self.index + 1
        total = len(self.active_fields)
        details = f['explanation'] or '이 항목의 정확한 설명은 확인이 필요합니다.'
        uncertain = ' 이 항목의 의미나 작성 조건은 추가 검수가 필요합니다.' if f['review_required'] else ''
        cond = f" 인쇄된 작성 조건: {f['condition']}." if f['condition'] else ''
        req = ('필수 여부는 확인되지 않았습니다.' if f['requirement'] == 'unknown' else
               f"문서 분석상 { {'required':'필수', 'optional':'선택', 'conditional':'조건부'}[f['requirement']] } 항목입니다. 단, 최종적으로 원문 확인이 필요합니다.")
        return [Event('SPEAK', f'{number}번째 항목, {f["name"]}입니다. {details}{cond} {req}{uncertain} 작성 위치로 안내할까요?', field_id=f['id'])]

    def _advance(self) -> list[Event]:
        self.index += 1
        return self._explain_field()

    def _type_from_utterance(self, utterance: str) -> str | None:
        matches = []
        for app in self.workflow['application_types']:
            if any(normalize_voice(name) == utterance for name in [app['name'], *app['aliases']]):
                matches.append(app['id'])
        return matches[0] if len(matches) == 1 else None

    def on_user(self, spoken: str) -> list[Event]:
        utterance = normalize_voice(spoken)
        if not utterance:
            # Silence is handled by STT frontend; do not endlessly prompt.
            return []
        if self.state == State.INIT:
            return [Event('SPEAK', '안내가 시작되지 않았습니다.')]
        if self.state == State.PAUSED:
            if utterance not in RESUME:
                return [Event('SPEAK', '안내가 일시 중단된 상태입니다. 계속이라고 말씀해주세요.')]
            self.state = self._resume_state or State.WAIT_CONSENT
            events = [Event('SPEAK', '안내를 재개합니다.')]
            if self.state == State.TRACKING and self.field and self.navigation:
                tid = self.navigation.lookup(self.field['id'])
                if tid:
                    events.append(Event('RESUME_TRACKING', field_id=self.field['id'], target_id=tid))
                else:
                    self.state = State.ASK_TO_NAVIGATE
                    events.append(Event('SPEAK', '이 항목의 검수된 위치 정보를 찾을 수 없습니다.'))
            return events
        if utterance in PAUSE and self.state != State.FINISHED:
            self._resume_state = self.state
            self.state = State.PAUSED
            return [Event('PAUSE_TRACKING'), Event('SPEAK', '안내를 일시 중단했습니다. 계속이라고 말씀하면 재개합니다.')]
        if self.state == State.FINISHED:
            return [Event('SPEAK', '이 문서의 안내 순서는 종료되었습니다.')]
        if utterance in REPEAT:
            if self.state == State.ASK_TO_NAVIGATE:
                return self._explain_field()
            if self.state == State.SELECT_TYPE:
                return [Event('SPEAK', '신청 유형을 선택해 주세요. ' + ', '.join(t['name'] for t in self.workflow['application_types']))]
            if self.state == State.SELECT_CHOICE:
                return self._ask_choice_or_consent()
            if self.state == State.TRACKING:
                return [Event('SPEAK', '손가락을 문서 위에서 움직여주세요. 위치를 계속 추적합니다.')]
            if self.state == State.WAIT_WRITING:
                return [Event('SPEAK', '현재 항목에 작성하신 다음 완료라고 말씀해주세요. 실제 작성 여부는 아직 확인하지 않았습니다.')]
            return [Event('SPEAK', '순서대로 안내할까요? 예 또는 아니요로 답해주세요.')]
        if utterance in BACK and self.active_fields and self.state in {State.ASK_TO_NAVIGATE, State.TRACKING, State.WAIT_WRITING}:
            if self.state == State.TRACKING:
                stop = [Event('STOP_TRACKING')]
            else:
                stop = []
            self.index = max(0, self.index - 1)
            return stop + self._explain_field()
        if self.state == State.SELECT_TYPE:
            selected = self._type_from_utterance(utterance)
            if not selected:
                return [Event('SPEAK', '선택한 유형을 알아듣지 못했습니다. 신청 유형을 다시 말씀해주세요.')]
            self.selected_type = selected
            self._activate_fields()
            label = next(x['name'] for x in self.workflow['application_types'] if x['id'] == selected)
            return self._ask_choice_or_consent(f'{label}을 선택하셨습니다.')
        if self.state == State.SELECT_CHOICE:
            _, choices = self.pending_choice_groups[0]
            matched = [f for f in self.active_fields if f['id'] in choices and normalize_voice(f['name']) == utterance]
            if len(matched) != 1:
                return [Event('SPEAK', '선택할 항목 이름을 다시 말씀해주세요. ' + ', '.join(f['name'] for f in self.active_fields if f['id'] in choices))]
            chosen = matched[0]
            self.active_fields = [f for f in self.active_fields if f['id'] not in choices or f['id'] == chosen['id']]
            for fid in choices:
                if fid != chosen['id']:
                    self.field_status[fid] = 'not_applicable'
            self.pending_choice_groups.pop(0)
            return self._ask_choice_or_consent(f'{chosen["name"]}을 선택하셨습니다.')
        if self.state == State.WAIT_CONSENT:
            if utterance in YES:
                return self._explain_field()
            if utterance in NO:
                self.state = State.FINISHED
                return [Event('SPEAK', '작성 안내를 종료합니다.'), Event('FINISHED')]
            return [Event('SPEAK', '진행하시려면 예, 종료하시려면 아니요라고 말씀해주세요.')]
        if self.state == State.ASK_TO_NAVIGATE:
            if utterance in SKIP or utterance in NEXT or utterance in NO:
                self.field_status[self.field['id']] = 'skipped'
                return self._advance()
            if utterance in YES:
                fid = self.field['id']
                tid = self.navigation.lookup(fid) if self.navigation else None
                if tid is None:
                    self.field_status[fid] = 'unverified'
                    return [Event('SPEAK', '이 작성란은 실제 위치가 검수되지 않았거나 추가 확인이 필요하므로 손가락 위치 안내를 시작할 수 없습니다. 건너뛰기라고 말씀하시면 다음 항목으로 이동합니다.', field_id=fid)]
                self.state = State.TRACKING
                return [Event('SPEAK', '검수된 작성 위치로 안내를 시작합니다.', field_id=fid),
                        Event('START_TRACKING', field_id=fid, target_id=tid)]
            return [Event('SPEAK', '위치를 안내할까요? 예 또는 건너뛰기라고 답해주세요.')]
        if self.state == State.TRACKING:
            # Only the tracking engine signals ARRIVED. Spoken '완료' cannot spoof arrival.
            return [Event('SPEAK', '목표 위치로 이동 중입니다. 위치 안내가 끝난 다음 작성해주세요.')]
        if self.state == State.WAIT_WRITING:
            if utterance in DONE or utterance in YES:
                self.field_status[self.field['id']] = 'user_claimed'
                return [Event('SPEAK', '사용자가 작성 완료했다고 보고한 것으로 기록했습니다. 실제 필기 상태는 아직 영상으로 확인하지 않았습니다.'), *self._advance()]
            if utterance in SKIP or utterance in NEXT:
                self.field_status[self.field['id']] = 'skipped'
                return self._advance()
            return [Event('SPEAK', '작성하셨다면 완료, 작성하지 않으셨다면 건너뛰기라고 말씀해주세요.')]
        return [Event('SPEAK', '다시 말씀해주세요.')]

    def on_arrived(self) -> list[Event]:
        if self.state != State.TRACKING or self.field is None:
            return []
        self.state = State.WAIT_WRITING
        return [Event('STOP_TRACKING', field_id=self.field['id']),
                Event('SPEAK', '목표 위치에 도달했습니다. 작성하셨다면 완료라고 말씀해주세요. 사진으로 필기 여부를 확인한 것은 아닙니다.', field_id=self.field['id'])]

    def on_tracking_lost(self) -> list[Event]:
        if self.state != State.TRACKING:
            return []
        self._resume_state = self.state
        self.state = State.PAUSED
        return [Event('PAUSE_TRACKING'), Event('SPEAK', '문서 또는 손 추적이 불안정하여 위치 안내를 중단했습니다. 촬영 상태를 복구한 뒤 계속이라고 말씀해주세요.')]

    def summary(self) -> dict:
        return {'state': self.state.value, 'selected_application_type': self.selected_type,
                'field_status': dict(self.field_status), 'excluded_non_applicant_ids': self.excluded_fields}
