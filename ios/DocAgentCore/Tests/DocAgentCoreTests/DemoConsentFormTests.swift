import XCTest
@testable import DocAgentCore

final class DemoConsentFormTests: XCTestCase {
    private let reference = DemoConsentForm.rows

    private func lines(transform: (CGPoint) -> CGPoint = { $0 }) -> [PrintedLine] {
        reference.enumerated().map { i, row in
            let r = row.rect
            let points = [CGPoint(x: r.minX, y: r.minY), CGPoint(x: r.minX, y: r.maxY),
                          CGPoint(x: r.maxX, y: r.minY), CGPoint(x: r.maxX, y: r.maxY)].map(transform)
            let xs = points.map(\.x), ys = points.map(\.y)
            let rect = CGRect(x: xs.min()!, y: 1 - ys.max()!, width: xs.max()!-xs.min()!, height: ys.max()!-ys.min()!)
            return PrintedLine(id: i, text: row.text, confidence: 0.69, bounds: rect)
        }
    }

    func testMappedTargetsStayOnTheSameCapturedStill() throws {
        let original = try XCTUnwrap(DemoConsentForm.match(lines()))
        func perspective(_ p: CGPoint) -> CGPoint {
            let scale = 1 + 0.04 * p.y
            return CGPoint(x: (0.76*p.x - 0.033*p.y + 0.12)/scale, y: (0.034*p.x + 0.70*p.y + 0.08)/scale)
        }
        let matched = try XCTUnwrap(DemoConsentForm.match(lines(transform: perspective)))
        XCTAssertEqual(matched.targets.map(\.id), ["collection-yes","collection-no","provision-yes","provision-no","applicant-signature"])
        for (field, test) in zip(original.targets, matched.targets) {
            let expected = perspective(CGPoint(x: field.bounds.midX, y: field.bounds.midY))
            XCTAssertEqual(expected.x, test.bounds.midX, accuracy: 0.0015)
            XCTAssertEqual(expected.y, test.bounds.midY, accuracy: 0.0015)
        }
    }

    func testHeadingImitationOtherRevisionsAndMovedChoiceRowsFailClosed() {
        let all = lines()
        XCTAssertNil(DemoConsentForm.match(Array(all.prefix(8))))
        XCTAssertNil(DemoConsentForm.match(all.map { $0.withText($0.text.replacingOccurrences(of: "2024.1.1.", with: "2024.1.11.")) }))
        XCTAssertNil(DemoConsentForm.match(all.map { $0.withText($0.text.replacingOccurrences(of: "별지 제3호서식", with: "별지 제4호서식")) }))
        XCTAssertNil(DemoConsentForm.match(all.map { $0.withText($0.text.replacingOccurrences(of: "최대 10년", with: "최대 5년")) }))
        XCTAssertNil(DemoConsentForm.match(all.map { line in
            guard line.text.contains("동의하지 않음") else { return line }
            return PrintedLine(id: line.id, text: line.text, confidence: 1, bounds: line.bounds?.offsetBy(dx: 0, dy: 0.04))
        }))
        XCTAssertNil(DemoConsentForm.exactPhoto(Data(repeating: 0, count: 1_479_312)))
    }

    func testCheckedChoicesAndUnfamiliarWordingRemainLive() throws {
        let all = lines().map { line in
            line.withText(line.text.replacingOccurrences(of: "동의함 □", with: "동의함 ☑")
                .replacingOccurrences(of: "이름:", with: "이름: 테스트")
                .replacingOccurrences(of: "이용 목적을 달성할 때까지 보유 및 이용합니다.", with: "이용 목적 변경 후 새 조건을 적용합니다."))
        }
        let retained = try XCTUnwrap(DemoConsentForm.match(all)).lines.map(\.text)
        XCTAssertEqual(retained.filter { $0.contains("☑") }.count, 2)
        XCTAssertTrue(retained.contains("이름: 테스트"))
        XCTAssertTrue(retained.contains("이용 목적 변경 후 새 조건을 적용합니다."))
    }
}
