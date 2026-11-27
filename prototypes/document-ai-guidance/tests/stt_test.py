import speech_recognition as sr

recognizer = sr.Recognizer()

with sr.Microphone() as source:

    print("주변 소음 측정 중...")
    recognizer.adjust_for_ambient_noise(
        source, duration=1
    )

    print("음성 명령을 말해주세요.")

    while True:

        try:
            audio = recognizer.listen(
                source,
                timeout=5,
                phrase_time_limit=5
            )

            text = recognizer.recognize_google(
                audio,
                language="ko-KR"
            )

            print("인식 결과:", text)

            if "종료" in text:
                break

        except sr.WaitTimeoutError:
            print("음성 입력 대기 중...")

        except sr.UnknownValueError:
            print("음성을 인식하지 못했습니다.")

        except sr.RequestError as e:
            print("STT 오류:", e)