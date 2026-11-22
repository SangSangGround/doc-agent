import CoreGraphics
import Foundation
import XCTest
@testable import DocAgentCore

final class WritingFlowTests: XCTestCase {
    private func render(_ size: CGSize, draw: (CGContext) -> Void) throws -> CGImage {
        let ctx = try XCTUnwrap(CGContext(data: nil, width: Int(size.width), height: Int(size.height),
            bitsPerComponent: 8, bytesPerRow: 0, space: CGColorSpaceCreateDeviceRGB(),
            bitmapInfo: CGImageAlphaInfo.noneSkipLast.rawValue))
        draw(ctx)
        return try XCTUnwrap(ctx.makeImage())
    }

    func testReadsFreshInkAndAbstainsOnMissingOutline() throws {
        let field = WritingTarget(id: "one", label: "찬성", kind: .checkbox,
                                  bounds: CGRect(x: 0.2, y: 0.1, width: 0.1, height: 0.1))
        for (outline, mark, expected): (Bool, Bool, CheckboxReading) in
            [(true, false, .empty), (true, true, .marked), (false, false, .unclear), (false, true, .unclear)] {
            let image = try render(CGSize(width: 400, height: 400)) { c in
                c.setFillColor(gray: 0.95, alpha: 1)
                c.fill(CGRect(x: 0, y: 0, width: 400, height: 400))
                c.setStrokeColor(gray: 0.1, alpha: 1)
                c.setLineWidth(2)
                if outline { c.stroke(CGRect(x: 80, y: 320, width: 40, height: 40)) }
                if mark {
                    c.setLineWidth(3)
                    c.move(to: CGPoint(x: 87,y: 343)); c.addLine(to: CGPoint(x: 98,y: 330))
                    c.addLine(to: CGPoint(x: 115,y: 353)); c.strokePath()
                }
            }
            XCTAssertEqual(CheckboxInspection.read(field, in: image), expected)
        }
    }

    func testRemoteFieldsBelongToThisUprightImageAndHaveContainedDistinctRegions() throws {
        func response(_ boxes: [DetectedWritingFields.Field]) -> DetectedWritingFields {
            .init(imageSHA256: "known", coordinateSpace: "upright-top-left-normalized", fields: boxes)
        }
        let box = DetectedWritingFields.Field(id: "a", label: "동의하지 않음", kind: "checkbox", bounds: [0.2,0.5,0.1,0.1])
        XCTAssertEqual(try response([box]).resolve(for: "known").first?.label, box.label)
        XCTAssertThrowsError(try response([box]).resolve(for: "other"))
        XCTAssertThrowsError(try response([box, box]).resolve(for: "known"))
        XCTAssertThrowsError(try response([.init(id: "a", label: "서명", kind: "signature", bounds: [0.95,0.5,0.2,0.1])]).resolve(for: "known"))
    }

    func testFollowupsUseCurrentConversationInCacheKey() throws {
        let row = PrintedLine(id: 0, text: "근로복지공단", uncertain: false)
        let first = ConsentTurn(question: "제공 기관은?", reply: .page([row]))
        let followup = "그럼 얼마나?"
        let blank = try CloudQuestionRequest(question: followup, lines: [row], history: [])
        let withHistory = try CloudQuestionRequest(question: followup, lines: [row], history: [first])
        var cache = SessionAnswerCache()
        let key = try SessionAnswerCache.key(for: blank)
        cache.record(.page([row]), for: key)
        XCTAssertNotNil(cache.lookup(key))
        XCTAssertNil(cache.lookup(try SessionAnswerCache.key(for: withHistory)))
        XCTAssertTrue(SessionAnswerCache.permitsImmediateRepeat("이 문서의 개인정보 수집 목적은 뭐야?"))
        XCTAssertFalse(SessionAnswerCache.permitsImmediateRepeat("그 기관의 보유 기간은?"))
        XCTAssertFalse(SessionAnswerCache.permitsImmediateRepeat("수집 목적을 다른 예시로 설명해 줘"))
    }

    func testSubmittedPhotoInkAndAnswers() throws {
        guard let path = ProcessInfo.processInfo.environment["DOCAGENT_DEMO_PHOTO"] else {
            throw XCTSkip("Set DOCAGENT_DEMO_PHOTO to the supplied unmodified HEIC for the photo replay.")
        }
        let bytes = try Data(contentsOf: URL(fileURLWithPath: path))
        let demo = try XCTUnwrap(DemoConsentForm.exactPhoto(bytes))
        let reference = try WritingReference(photo: bytes, rectify: false)
        let boxes = demo.targets.filter { $0.kind == .checkbox }
        XCTAssertEqual(boxes.count, 4)
        XCTAssertEqual(demo.targets.filter { $0.kind == .signature }.count, 1)
        let w = CGFloat(reference.image.width), h = CGFloat(reference.image.height)
        var emptyHits = 0, markedHits = 0, noDecision = 0
        for mask in 0..<16 {
            let sample = try render(CGSize(width: w, height: h)) { c in
                c.draw(reference.image, in: CGRect(x: 0, y: 0, width: w, height: h))
                c.setStrokeColor(red: 0.03, green: 0.05, blue: 0.13, alpha: 1)
                for (idx, box) in boxes.enumerated() where mask & (1 << idx) != 0 {
                    let b = box.bounds
                    func point(_ u: CGFloat, _ v: CGFloat) -> CGPoint {
                        CGPoint(x: (b.minX + u*b.width)*w, y: (1-b.minY-v*b.height)*h)
                    }
                    c.setLineWidth(max(2, 0.10*b.width*w)); c.setLineCap(.round)
                    c.move(to: point(0.12,0.50)); c.addLine(to: point(0.43,0.78)); c.addLine(to: point(0.89,0.18)); c.strokePath()
                }
            }
            for (idx, box) in boxes.enumerated() {
                let checked = mask & (1 << idx) != 0
                let result = CheckboxInspection.read(box, in: sample)
                if result == .unclear { noDecision += 1 }
                if checked && result == .marked { markedHits += 1 }
                if !checked && result == .empty { emptyHits += 1 }
            }
        }
        print("writing-photo-replay marked=\(markedHits)/32 empty=\(emptyHits)/32 unclear=\(noDecision)/64")
        XCTAssertEqual(markedHits, 32); XCTAssertEqual(emptyHits, 32)
        let prepared = ["이거 어떤 서류야?", "한 줄로 요약해줘", "내 정보는 누구에게 제공돼?", "동의를 거부하면 어떻게 돼?"]
        let live = ["그럼 거기는?", "지난달 이 대출 이용자는 몇 명이야?", "내가 방금 어느 칸에 체크했어?", "이거 어떤 서류야? 그리고 언제까지 제출해?"]
        var times = [Double]()
        for _ in 0..<30 {
            for question in prepared {
                let start = DispatchTime.now().uptimeNanoseconds
                let result = DemoDocumentAnswers.prepared(question, match: demo, page: demo.lines)
                times.append(Double(DispatchTime.now().uptimeNanoseconds - start) / 1_000_000)
                XCTAssertNotNil(result)
                XCTAssertTrue(result?.source.contains("인식 줄 번호") == true)
            }
        }
        for q in live { XCTAssertNil(DemoDocumentAnswers.prepared(q, match: demo, page: demo.lines)) }
        times.sort()
        print("writing-demo-qa local=4/4 delegated=4/4 lookup-samples=\(times.count) p50-ms=\(times[59]) p95-ms=\(times[113])")
    }
}
