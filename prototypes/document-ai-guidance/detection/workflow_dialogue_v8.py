"""v8 linked-workflow conversation controller. No camera, microphone or TTS side effects.

Consumes workflow_link_v8.LinkedWorkflow (NOT the older P0 JSON schema).
Navigation is permitted only for fields passing the v8 Linker's native-manifest gates.
"""
from __future__ import annotations

from dataclasses import dataclass
import re


@dataclass(frozen=True)
class Event:
    kind: str                 # SPEAK, START_TRACKING, STOP_TRACKING, FINISHED
    text: str = ""
    field_id: str = ""


@dataclass
class Item:
    kind: str                 # field, choice
    ids: list[str]
    group_id: str | None = None
    question: str = ""


def nav_allowed(f) -> bool:
    """Fail closed: reviewed applicant, and no AI/manifest official or third-party hint."""
    return (f.navigable and f.role == 'applicant' and f.reviewed_role == 'applicant'
            and f.ai_role not in ('official', 'third_party')
            and not f.uncertain and f.confidence != 'low')


def norm(text: str) -> str:
    return re.sub(r"[^0-9가-힣]", "", str(text or "").lower())


# Google STT can omit / mishear one-syllable replies such as "예".
# Accept longer explicit affirmative replies too; never infer consent from
# a vague single vowel ("에" / "애") or an unrelated sentence.
YES = {norm(x) for x in (
    "네", "예", "응", "좋아", "시작", "시작해", "진행해",
    "네 진행해 주세요", "예 진행해 주세요", "진행해 주세요", "진행할게요",
    "네 안내해 주세요", "예 안내해 주세요", "안내해 주세요",
    "네 시작해 주세요", "예 시작해 주세요", "시작해 주세요",
    "네 맞아요", "예 맞아요", "맞아요", "네 맞습니다",
    "네 부탁해요", "네 좋아요", "좋습니다", "네 그렇게 해 주세요",
)}
# Stronger affirmative wording is accepted for generic navigation consent,
# but it must NOT assert signer identity, legal conditions or writing completion.
YES_PLAIN = {norm(x) for x in (
    "예", "네", "응", "네 맞아요", "예 맞아요", "맞아요", "네 맞습니다",
)}
YES_SIGNER = YES_PLAIN | {norm(x) for x in (
    "본인 맞아요", "네 본인 맞아요", "예 본인 맞아요",
    "네 제가 서명합니다", "제가 직접 서명합니다",
)}
YES_CONDITION = YES_PLAIN | {norm(x) for x in (
    "조건에 해당해요", "조건에 해당합니다", "네 해당해요", "해당돼요","조건에 해당해","해당해","해당해요"
)}
NO = {norm(x) for x in (
    "아니", "아니요", "아뇨", "아니요 괜찮아요", "아니요 안 할게요",
    "안 할게요", "진행하지 않을게요",
)}
DONE = {norm(x) for x in ("완료", "다 썼어", "다썼어", "작성완료", "다했어", "작성했어")}
SKIP = {norm(x) for x in ("건너뛰기", "건너 뛰기", "스킵")}
NEXT = {norm(x) for x in ("다음", "다음항목", "넘어가")}
PREV = {norm(x) for x in ("이전", "이전항목", "뒤로")}
REPEAT = {norm(x) for x in ( "다시 듣기",
    "다시듣기",
    "다시 들려줘",
    "다시 들려주세요",
    "다시 말해줘",
    "다시 말씀해 주세요",
    "다시설명",
    "다시설명해줘",
    "다시말해줘",
    "설명",
    "위치알려줘")}
PAUSE = {norm(x) for x in ("중단", "멈춰", "일시중지", "정지")}
RESUME = {norm(x) for x in ("계속", "재개", "다시시작", "이어하기")}
OVERVIEW = {norm(x) for x in ("전체설명", "문서설명", "문서내용")}
UNSURE = {norm(x) for x in ("모르겠어", "잘모르겠어", "확인안됨", "불확실")}


class WorkflowDialogueV8:
    def __init__(self, linked):
        self.linked = linked
        self.state = 'INIT'
        self.resume_state = None
        self.auto_resume_tracking = False
        self.selected_type = None
        self.pending_type_id = None  # uncertain STT name requires explicit confirmation
        self.steps: list[Item] = []
        self.index = 0
        self.current_track = None
        self.status = {fid: 'pending' for fid in linked.fields}
        self.choice_selections = {}
        self.signer_confirmed = set()
        self.last_spoken = ''

    def say(self, message: str, field_id: str = '') -> list[Event]:
        # 1. 문서 기호를 TTS 친화적으로 변환
        message = re.sub(r'\[\s*\]', '체크박스 ', message)
        message = re.sub(r'[√✓✔]\s*표', '체크 표시', message)
        message = re.sub(r'(?<![A-Za-z])[Vv]\s*표', '체크 표시', message)
        message = message.replace('※', '').strip()

        # 2. 설명이 포함된 긴 안내에 다시 듣기 문구 추가
        repeat_hint = '다시 듣고 싶다면 다시 듣기라고 말씀해 주세요.'

        if len(message) >= 90 and repeat_hint not in message:
            message = f'{message} {repeat_hint}'

        # 3. 직전 안내 저장
        self.last_spoken = message

        return [Event('SPEAK', message, field_id)]

    @property
    def item(self) -> Item | None:
        return self.steps[self.index] if 0 <= self.index < len(self.steps) else None

    @property
    def field(self):
        item = self.item
        if item and item.kind == 'field' and item.ids:
            return self.linked.fields[item.ids[0]]
        return None

    def intro(self):
        title = str(self.linked.title or '제목이 확인되지 않은 문서').strip()
        desc = self.linked.description or '문서의 용도는 별도 확인이 필요합니다.'
        # Omit selectable title checkboxes only in the overall introduction.
        # No form-specific names or application types are hardcoded here.
        marker = r'(?:\[\s*\]|[☐☑□▢]|체크박스)'
        if re.search(marker, title):
            stem = re.split(marker, title, maxsplit=1)[0].strip(' ,.:;\u00b7\u318d')
            suffix = re.search(r'(신청서|신고서|동의서|확인서|청구서|요청서|서식)\s*$', title)
            if stem:
                title = (stem + (' ' + suffix.group(1) if suffix and not stem.endswith(suffix.group(1)) else ''))
        return (f'이 문서는 {title}입니다. {desc} '
                '안내 내용은 인공지능 분석 결과로, 작성 조건을 확정한 것은 아닙니다.')
    
    def _type_options_spoken(self):
        number_words = {
            1: "일번",
            2: "이번",
            3: "삼번",
        }

        options = []

        for idx, app in enumerate(self.linked.app_types, 1):
            number = number_words.get(idx, f"{idx}번")
            options.append(f"{number} {app.spoken_name}")

        return ", ".join(options)

    def start(self):
        if self.state != 'INIT':
            return self.say('이미 안내가 시작됐습니다.')
        if self.linked.app_types:
            self.state = 'SELECT_TYPE'
            names = ', '.join(t.spoken_name for t in self.linked.app_types)
            question = self.linked.selection_question or '어떤 신청을 진행하시겠습니까?'
            return self.say(f'{self.intro()} {question} 선택 가능한 유형은 {names}입니다.'
                            f'원하는 유형의 번호나 이름을 말씀해 주세요.')
        self._load_plan(None)
        return self._ask_consent(self.intro())

    def _load_plan(self, type_id):
        raw = self.linked.plan_for(type_id)
        # A reviewed but AI-official field might still be labelled navigable by
        # an older linker. Never ask the applicant to fill an office-only field.
        original = []
        for step in raw:
            ids = [fid for fid in step.field_ids
                   if self.linked.fields[fid].ai_role != 'official'
                   and self.linked.fields[fid].reviewed_role != 'official'
                   and self.linked.fields[fid].role != 'official']
            if ids:
                kind = step.kind if len(ids) > 1 else 'field'
                original.append(Item(kind, ids, step.group_id, step.question))
        group_members = {}
        for step in original:
            if step.group_id and step.group_id in self.linked.groups:
                g = self.linked.groups[step.group_id]
                if g['type'] == 'multi_choice':
                    group_members.setdefault(step.group_id, []).extend(step.ids)
        used = set()
        result = []
        for step in original:
            gid = step.group_id
            if gid and gid in group_members:
                if gid in used:
                    continue
                used.add(gid)
                unique = list(dict.fromkeys(group_members[gid]))
                result.append(Item('choice', unique, gid, self.linked.groups[gid].get('question', '')))
            else:
                result.append(Item(step.kind, list(step.ids), gid, step.question))
        self.steps = result
        self.index = 0
        self.selected_type = type_id

    def _ask_consent(self, preface=''):
        if not self.steps:
            self.state = 'FINISHED'
            self.current_track = None
            return [*self.say(
                '신청 유형은 선택됐지만, 연결된 작성 안내 항목이 0개입니다. '
                '아직 문서 안내를 시작할 수 없습니다. '
                '문서 워크플로와 작성 항목 연결을 확인해야 합니다.'
            ), Event('FINISHED')]
        self.state = 'WAIT_CONSENT'
        nav_count = sum(any(nav_allowed(self.linked.fields[fid]) for fid in it.ids) for it in self.steps)
        text = f'{preface} 작성 안내 후보는 {len(self.steps)}단계이고, 그중 {nav_count}단계에 검수된 안내 좌표가 있습니다. '
        text += '안내 좌표가 없는 항목은 위치를 추측하지 않고 보류합니다. 항목 설명을 시작할까요? 네 진행해 주세요 또는 아니요로 답해 주세요.'
        return self.say(text)

    def _finish(self):
        self.state = 'FINISHED'
        self.current_track = None
        done = sum(v == 'user_claimed' for v in self.status.values())
        skipped = sum(v in {'skipped', 'not_applicable', 'needs_third_party'} for v in self.status.values())
        unverified = sum(v == 'unverified' for v in self.status.values())
        active_ids = {fid for step in self.steps for fid in step.ids}
        pending = sum(self.status[fid] == 'pending' for fid in active_ids)
        return [*self.say(f'안내 순서가 끝났습니다. 사용자 완료 보고 {done}개, 건너뛰거나 해당하지 않는 항목 {skipped}개, 위치 미검증으로 보류한 항목 {unverified}개, 안내 경로상 아직 완료 보고가 없는 항목 {pending}개입니다. 실제 작성 결과를 촬영해 검증한 것은 아닙니다.'), Event('FINISHED')]

    def _next(self):
        self.index += 1
        return self._present()

    def _present(self):
        item = self.item
        if item is None:
            return self._finish()
        if item.kind == 'choice':
            self.state = 'SELECT_CHOICE'
            names = ' '.join(f'{i}번 {self.linked.fields[fid].name}.' for i, fid in enumerate(item.ids, 1))
            g = self.linked.groups.get(item.group_id, {})
            mode = '한 가지' if g.get('type') == 'single_choice' else '해당하는 항목들을'
            question = item.question or '어떤 항목에 해당하시나요?'
            return self.say(f'{question} {mode} 선택해 주세요. {names} 해당 번호 또는 정확한 항목 이름으로 말씀해 주세요. 해당하지 않거나 모르겠으면 건너뛰기라고 말씀해 주세요.')
        f = self.field
        assert f is not None
        if f.role == 'third_party':
            self.state = 'WAIT_EXTERNAL'
            return self.say(f'{f.name} 항목은 다른 사람이 작성하거나 서명해야 하는 것으로 해석됐습니다. 신청인에게 해당 위치로 손가락 안내를 하지 않습니다. 해당 관계자가 작성했음을 확인하셨다면 작성 완료, 아니면 건너뛰기라고 말씀해 주세요.', f.id)
        if f.class_name == 'signature_field' and f.id not in self.signer_confirmed:
            self.state = 'ASK_SIGNER'
            return self.say(f'{f.name}은 서명 또는 날인이 필요한 항목입니다. 신청인과 서명해야 하는 사람이 같다고 단정할 수 없습니다. 해당 서명을 직접 해야 하는 당사자가 본인이 맞나요? 네 본인 맞아요 또는 아니요로 말씀해 주세요. 다른 사람의 서명이라면 본인이 대신 서명하면 안 됩니다.', f.id)
        if f.requirement == 'conditional' or f.condition_text:
            self.state = 'ASK_CONDITION'
            wording = f.source_quote or f.condition_text
            return self.say(f'{f.name}은 조건부 작성 항목입니다. 문서에는 다음과 같이 안내되어 있습니다. {wording or "작성 조건이 확인되지 않았습니다."} 본인에게 이 조건이 해당하나요? 조건에 해당해요 또는 아니요로 답해 주세요. 확인하기 어려우면 건너뛰기라고 말씀해 주세요.', f.id)
        return self._explain_field()

    def _explain_field(self):
        f = self.field
        if not f:
            return self._finish()
        self.state = 'ASK_NAV'
        detail = f.description or '구체적인 작성 내용은 원본 문서 확인이 필요합니다.'
        requirement_hint = {
            'required': 'AI 분석상 필수 항목으로 분류됐습니다.',
            'optional': 'AI 분석상 선택 항목으로 분류됐습니다.',
            'unknown': '필수 여부는 확인되지 않았습니다.',
            'conditional': '문서에 적힌 작성 조건을 먼저 확인해야 합니다.',
        }.get(f.requirement, '작성 여부 확인이 필요합니다.')
        uncertain = ' 의미 또는 작성 조건에 불확실성이 있어 원문 확인이 필요합니다.' if f.uncertain or f.confidence == 'low' else ''
        if nav_allowed(f):
            tail = '검수된 작성 위치로 손가락을 안내할까요? 네 진행해주세요 또는 건너뛰기라고 말씀해 주세요.'
        else:
            tail = '이 항목은 안전하게 확인된 신청인용 안내 좌표가 없습니다. 위치를 추측해 안내하지 않겠습니다. 건너뛰기라고 말씀해 주세요.'
        return self.say(f'{self.index+1}번째 항목은 {f.name}입니다. {detail} {requirement_hint}{uncertain} 최종 작성 여부는 원문과 본인의 상황을 기준으로 확인해야 합니다. {tail}', f.id)

    @staticmethod
    def _type_forms(app):
        """Build exact, document-derived aliases; never substring-match a sentence.

        Bare "기록" is a safe spoken form of "기록 신청". Do not derive
        alternative application semantics such as 신규/해지 from fixed rules.
        """
        forms = {norm(x) for x in [app.spoken_name, *app.aliases]}
        for value in list(forms):
            if value.endswith('신청서') and len(value) > len('신청서'):
                forms.add(value[:-len('신청서')])
            if value.endswith('신청') and len(value) > len('신청'):
                forms.add(value[:-len('신청')])
        return {x for x in forms if x}

    @staticmethod
    def _matches_type_spoken(voice, alias):
        """Accept short polite requests or repeated SAME label, not mixed labels.

        Matching is anchored to the full utterance. For example, '기록이요'
        passes for 기록 but '기록 또는 삭제' cannot choose either.
        """
        if voice == alias:
            return True
        suffixes = (
            '이요', '요', '으로요', '로요', '으로', '로',
            '할게요', '할게', '할래요', '할래', '하겠습니다',
            '해주세요', '해줘', '해줘요', '할거예요', '할거야',
            '선택', '선택할게요', '선택하겠습니다', '선택해주세요',
            '신청', '신청이요', '신청할게요', '신청할래요',
            '신청하고싶어요', '신청하겠습니다', '신청해주세요',
            '하고싶어요', '하고싶어', '으로할게요', '로할게요',
            '으로할래요', '로할래요', '으로해주세요', '로해주세요',
            '으로하겠습니다', '로하겠습니다',
        )
        if any(voice == alias + ending for ending in suffixes):
            return True
        if any(voice == prefix + alias + ending
               for prefix in ('저는', '나는')
               for ending in ('', '이요', '요', '로할게요', '으로할게요', '신청할게요')):
            return True
        # STT sometimes transcribes a frustrated repetition as one utterance.
        return voice in {alias * 2, alias * 3}

    def _select_type(self, voice):
        matches = []

        spoken_numbers = {
            1: {"일번", "한번", "첫번째"},
            2: {"이번", "두번", "두번째"},
            3: {"삼번", "세번", "세번째"},
        }

        for idx, app in enumerate(self.linked.app_types, 1):
            nums = {
                str(idx),
                f'{idx}번',
                f'{idx}번이요',
                f'{idx}번으로',
                f'{idx}번할게요',
                f'{idx}번선택',
                f'{idx}번해주세요',
            }

            nums.update(spoken_numbers.get(idx, set()))

            forms = self._type_forms(app)

            if voice in nums or any(
                self._matches_type_spoken(voice, a)
                for a in forms
            ):
                matches.append(app)

        return matches[0] if len(matches) == 1 else None

    @staticmethod
    def _strip_type_request(voice):
        """Strip ordinary Korean politeness only; never match substrings in sentences."""
        suffixes = (
            '신청할게요', '신청해주세요', '신청하겠습니다', '신청하고싶어요',
            '선택할게요', '선택해주세요', '선택하겠습니다',
            '으로할게요', '로할게요', '으로해주세요', '로해주세요',
            '할게요', '할래요', '해주세요', '해줘요', '해줘',
            '신청이요', '신청', '이요', '으로요', '로요', '요',
        )
        for prefix in ('저는', '나는'):
            if voice.startswith(prefix):
                voice = voice[len(prefix):]
                break
        for suffix in suffixes:
            if voice.endswith(suffix) and len(voice) > len(suffix):
                return voice[:-len(suffix)]
        return voice

    @staticmethod
    def _single_syllable_difference(a, b):
        """Only suggest, never auto-select, when one Korean syllable differs."""
        return (2 <= len(a) == len(b) <= 7 and
                sum(x != y for x, y in zip(a, b)) == 1)

    def _suggest_type(self, voice):
        """Unique, near-sounding option -> confirmation, not immediate navigation.

        Example: an STT transcript '기억' can be a one-syllable mismatch for
        the form-provided alias '기록'. This is document-independent; aliases
        still come only from the detected workflow.
        """
        # Never infer a choice from text that mentions multiple alternatives.
        if any(t in voice for t in ('또는', '그리고', '이랑', '하고', '혹은')):
            return None
        core = self._strip_type_request(voice)
        candidates = []
        for app in self.linked.app_types:
            forms = self._type_forms(app)
            if any(self._single_syllable_difference(core, alias) for alias in forms):
                candidates.append(app)
        return candidates[0] if len(candidates) == 1 else None

    @staticmethod
    def _select_options(text, fields):
        """Only exact printed labels or explicit numbered choices; no substring guessing."""
        s = norm(text)
        if not s:
            return []
        if s in {'모두', '전부', '전체'}:
            return []    # Never infer checking every checkbox
        numeric = re.findall(r'([1-9])[ ]*번', str(text))
        if not numeric and re.fullmatch(r'[1-9]', s):
            numeric = [s]
        if numeric:
            nums = [int(n) for n in numeric]
            if nums and all(1 <= n <= len(fields) for n in nums):
                return list(dict.fromkeys(fields[n-1].id for n in nums))
            return []
        matches = []
        for f in fields:
            opts = {norm(f.name), norm(f.detected_label), *(norm(a) for a in f.aliases)}
            if s in opts:
                matches.append(f.id)
        return matches if len(matches) == 1 else []

    def on_user(self, spoken):
        v = norm(spoken)
        print(
        f"[VOICE DEBUG] state={self.state} "
        f"heard={spoken!r} normalized={v!r}",
        flush=True
    )
        if not v:
            return []
        if v in REPEAT:
         if not self.last_spoken:
            return self.say('아직 다시 들을 안내가 없습니다.')
         return self.say(self.last_spoken)
        
        family = ('YES' if v in YES else 'NO' if v in NO else
                  'AMBIGUOUS_SHORT' if v in {'에', '애', '예이'} else 'OTHER')
        print(f'[DIALOGUE ROUTE] state={self.state} command={family}', flush=True)
        # "에" and "애" often come from a poorly recognized "예". Never
        # silently convert them to consent (especially signer/condition states).
        if v in {'에', '애', '예이'} and self.state in {
                'WAIT_CONSENT', 'ASK_NAV', 'ASK_SIGNER', 'ASK_CONDITION', 'CONFIRM_TYPE'}:
            return self.say('짧은 대답을 정확히 인식하지 못했습니다. 진행하려면 네 진행해 주세요, 진행하지 않으려면 아니요라고 말씀해 주세요.')
        if self.state == 'INIT':
            return self.say('먼저 안내를 시작해 주세요.')
        if v in OVERVIEW:
            return self.say(self.intro())
        if self.state == 'PAUSED':
            if v in PAUSE:
                # Explicit user pause overrides camera-based automatic resumption.
                self.auto_resume_tracking = False
                return self.say('안내를 일시 중단했습니다. 다시 시작하려면 계속이라고 말씀해 주세요.')
            if v in RESUME:
                self.state = self.resume_state or 'ASK_NAV'
                self.resume_state = None
                self.auto_resume_tracking = False
                if self.state == 'TRACKING' and self.current_track:
                    return [*self.say('손가락 안내를 재개합니다.'),
                            Event('START_TRACKING', field_id=self.current_track)]
                return self.say('안내를 재개합니다. ' + self.last_spoken)
            if self.auto_resume_tracking:
                return self.say('문서를 자동으로 다시 찾고 있습니다. 인식되면 손가락 안내를 재개하겠습니다.')
            return self.say('안내가 일시 중단된 상태입니다. 계속이라고 말씀해 주세요.')
        if v in PAUSE and self.state != 'FINISHED':
            self.auto_resume_tracking = False
            self.resume_state = self.state
            self.state = 'PAUSED'
            return [Event('STOP_TRACKING'), *self.say('안내를 일시 중단합니다. 계속이라고 말씀하면 재개합니다.')]
        if self.state == 'FINISHED':
            return self.say('안내가 종료됐습니다.')
        
        if v in PREV and self.steps:
            events = [Event('STOP_TRACKING')] if self.state == 'TRACKING' else []
            self.current_track = None
            self.index = max(0, self.index-1)
            return events + self._present()
        if self.state == 'CONFIRM_TYPE':
            proposed = self.linked.type_by_id(self.pending_type_id) if hasattr(self.linked, 'type_by_id') else next(
                (x for x in self.linked.app_types if x.id == self.pending_type_id), None)
            if v in YES_PLAIN and proposed is not None:
                self.pending_type_id = None
                self._load_plan(proposed.id)
                return self._ask_consent(f'{proposed.spoken_name}을 선택하셨습니다. {proposed.description}')
            if v in NO:
                self.pending_type_id = None
                self.state = 'SELECT_TYPE'
                return self.say('확인했습니다. 신청 유형을 다시 말씀해 주세요. 번호로도 선택할 수 있습니다.')
            # A corrected, precise name or number should also work directly.
            self.pending_type_id = None
            self.state = 'SELECT_TYPE'
            return self.on_user(spoken)
        if self.state == 'SELECT_TYPE':
            choice = self._select_type(v)
            result = choice.id if choice is not None else 'NO_UNIQUE_MATCH'
            print(f'[TYPE MATCH] heard={spoken!r} normalized={v!r} result={result}', flush=True)
            if choice is None:
                suggestion = self._suggest_type(v)
                if suggestion is not None:
                    self.pending_type_id = suggestion.id
                    self.state = 'CONFIRM_TYPE'
                    return self.say(
                        f'{suggestion.spoken_name} 신청을 말씀하신 건가요? '
                        '네 맞아요 또는 아니요로 확인해 주세요.'
                    )
                return self.say(
                    '신청 유형을 정확히 확인하지 못했습니다. '
                    '번호 또는 신청 유형 이름으로 다시 말씀해 주세요.'
                )
            self._load_plan(choice.id)
            print(f'[TYPE PLAN] selected={self.selected_type!r} steps={len(self.steps)} '
                  f'raw={len(self.linked.plan_for(choice.id))}', flush=True)
            return self._ask_consent(
                f'{choice.spoken_name}을 선택하셨습니다. {choice.description}'
            )
        if self.state == 'WAIT_CONSENT':
            if v in YES or v == norm("해주세요"):
                return self._present()
            if v in NO:
                return self._finish()
            return self.say('시작할까요? 네 진행해 주세요 또는 아니요로 말씀해 주세요.')
        if self.state == 'SELECT_CHOICE':
            if v in SKIP or v in UNSURE:
                for fid in self.item.ids:
                    self.status[fid] = 'unverified'
                return self._next()
            group = self.linked.groups.get(self.item.group_id, {})
            opts = [self.linked.fields[fid] for fid in self.item.ids]
            selected = self._select_options(spoken, opts)
            if not selected:
                return self.say('선택 항목을 정확히 알아듣지 못했습니다. 항목 번호를 말씀해 주세요. 여러 개라면 1번 3번처럼 말씀할 수 있습니다.')
            if group.get('type') == 'single_choice' and len(selected) != 1:
                return self.say('한 가지만 선택할 수 있는 항목입니다. 하나를 다시 말씀해 주세요.')
            self.choice_selections[self.item.group_id] = list(selected)
            for fid in self.item.ids:
                if fid not in selected:
                    self.status[fid] = 'not_applicable'
            self.steps[self.index:self.index+1] = [Item('field', [fid], self.item.group_id) for fid in selected]
            return [*self.say('선택하신 항목은 ' + ', '.join(self.linked.fields[fid].name for fid in selected) + '입니다.'), *self._present()]
        if self.state == 'ASK_SIGNER':
            if v in YES_SIGNER:
                self.signer_confirmed.add(self.field.id)
                return self._present()
            if v in NO:
                self.status[self.field.id] = 'needs_third_party'
                return [*self.say('해당 서명은 서명권자가 직접 작성해야 합니다. 신청인 손가락 안내는 건너뜁니다.'), *self._next()]
            if v in SKIP or v in UNSURE:
                self.status[self.field.id] = 'unverified'
                return self._next()
            return self.say('본인이 해당 서명 당사자가 맞으면 네, 아니면 아니요라고 말씀해 주세요.')
        if self.state == 'ASK_CONDITION':
            if v in YES_CONDITION:
                return self._explain_field()
            if v in NO:
                self.status[self.field.id] = 'not_applicable'
                return self._next()
            if v in SKIP or v in UNSURE:
                self.status[self.field.id] = 'unverified'
                return self._next()
            return self.say('조건에 해당하면 조건에 해당해요, 해당하지 않으면 아니요, 잘 모르겠으면 건너뛰기라고 말씀해 주세요.')
        if self.state == 'WAIT_EXTERNAL':
            if v in DONE:
                self.status[self.field.id] = 'user_claimed'
                return self._next()
            if v in SKIP or v in NEXT:
                self.status[self.field.id] = 'skipped'
                return self._next()
            return self.say('해당 관계자의 작성 여부를 확인하셨다면 작성 완료, 아니면 건너뛰기라고 말씀해 주세요.')
        if self.state == 'ASK_NAV':
            # DEMO_V8_RECOVERY_V1: navigation request aliases, not legal consent.
            if v in {norm(x) for x in ('해주세요', '해 주세요', '네 해주세요',
                                               '안내해주세요', '위치 안내해주세요',
                                               '진행해주세요')}:
                v = norm('네 진행해 주세요')
            if v in SKIP or v in NEXT or v in NO:
                self.status[self.field.id] = 'unverified' if not nav_allowed(self.field) else 'skipped'
                return self._next()
            if v in YES or v in {
                norm("내 진행해 주세요"),
                norm("내 안내해 주세요"),
                norm("내 안내해"),
                norm("내 안내해줘"),
                norm("네 진행해"),
                norm("네"),
            }:
                f = self.field
                if not nav_allowed(f):
                    self.status[f.id] = 'unverified'
                    return self.say('위치로 안내할까요? 네 진행해주세요 또는 건너뛰기라고 말씀해 주세요.')
                self.state = 'TRACKING'
                self.current_track = f.id
                return [*self.say(f'{f.name} 작성 위치 안내를 시작합니다.', f.id), Event('START_TRACKING', field_id=f.id)]
            return self.say('위치로 안내할까요? 네 진행해주세요 또는 건너뛰기라고 말씀해 주세요.')
        if self.state == 'TRACKING':
            if v in SKIP:
                self.status[self.field.id] = 'skipped'
                self.current_track = None
                return [Event('STOP_TRACKING'), *self._next()]
            return self.say('손가락 안내 중입니다. 목표 근처 도착음을 들은 다음 작성해 주세요. 안내를 중단하려면 중단이라고 말씀해 주세요.')
        if self.state == 'WAIT_WRITING':
            if v in DONE:
                self.status[self.field.id] = 'user_claimed'
                return [*self.say('작성 완료했다고 말씀하신 것으로 기록했습니다. 실제 작성 여부를 사진으로 확인한 것은 아닙니다.'), *self._next()]
            if v in SKIP or v in NEXT:
                self.status[self.field.id] = 'skipped'
                return self._next()
            return self.say('작성하셨다면 작성 완료, 작성하지 않았다면 건너뛰기라고 말씀해 주세요.')
        return self.say('다시 말씀해주세요.')

    def on_arrived(self, field_id):
        if self.state != 'TRACKING' or self.current_track != field_id:
            return []
        self.state = 'WAIT_WRITING'
        self.current_track = None
        return [Event('STOP_TRACKING'), *self.say('목표 위치 근처에 도달했습니다. 이는 실제 접촉이나 작성 여부를 확인한 것이 아닙니다. 작성한 후 작성 완료라고 말씀해 주세요.', field_id)]

    def on_tracking_lost(self):
        if self.state != 'TRACKING':
            return []
        self.resume_state = 'TRACKING'
        self.state = 'PAUSED'
        self.auto_resume_tracking = True
        return [Event('STOP_TRACKING'), *self.say(
            '문서 인식이 잠시 끊어졌습니다. 자동으로 문서를 다시 찾고 있습니다.')]

    def on_tracking_restored(self):
        """Called by live camera loop only after stable, same-field re-acquisition."""
        if (self.state != 'PAUSED' or self.resume_state != 'TRACKING'
                or not self.auto_resume_tracking or not self.current_track):
            return []
        field = self.linked.fields.get(self.current_track)
        if field is None or not nav_allowed(field) or field.target_mm is None:
            return []
        self.state = 'TRACKING'
        self.resume_state = None
        self.auto_resume_tracking = False
        return [Event('START_TRACKING', field_id=field.id),
                *self.say('문서가 인식되었습니다. 안내를 시작합니다. 손가락을 종이 위에 올려주세요.', field.id)]

    def summary(self):
        return {'state': self.state, 'selected_type': self.selected_type, 'index': self.index,
                'statuses': dict(self.status), 'choices': dict(self.choice_selections)}
