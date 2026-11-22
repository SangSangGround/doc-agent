import CoreGraphics
import Foundation

public enum CheckboxReading: String, Sendable {
    case marked = "표시가 보여요"
    case empty = "빈칸으로 보여요"
    case unclear = "표시를 확인하기 어려워요"
}

/// Reads fresh pixels. Even a registered blank form can be photographed again with ink.
public enum CheckboxInspection {
    public static func read(_ field: WritingTarget, in image: CGImage) -> CheckboxReading {
        guard field.kind == .checkbox else { return .unclear }
        let w = CGFloat(image.width), h = CGFloat(image.height)
        let box = CGRect(x: field.bounds.minX * w, y: field.bounds.minY * h,
                         width: field.bounds.width * w, height: field.bounds.height * h)
        guard min(box.width, box.height) >= 8 else { return .unclear }
        let roi = box.insetBy(dx: -0.55 * box.width, dy: -0.55 * box.height).integral
        guard CGRect(x: 0, y: 0, width: w, height: h).contains(roi), let crop = image.cropping(to: roi)
        else { return .unclear }
        let side = 64
        var pixels = [UInt8](repeating: 255, count: side * side)
        let drawn = pixels.withUnsafeMutableBytes { bytes -> Bool in
            guard let c = CGContext(data: bytes.baseAddress, width: side, height: side, bitsPerComponent: 8,
                                    bytesPerRow: side, space: CGColorSpaceCreateDeviceGray(),
                                    bitmapInfo: CGImageAlphaInfo.none.rawValue) else { return false }
            c.interpolationQuality = .medium
            c.draw(crop, in: CGRect(x: 0, y: 0, width: side, height: side))
            return true
        }
        guard drawn else { return .unclear }
        let sorted = pixels.sorted()
        let paper = Int(sorted[sorted.count * 4 / 5])
        let dark = Int(sorted[sorted.count / 25])
        guard paper - dark >= 28 else { return .unclear }
        let limit = UInt8(max(0, paper - max(22, (paper - dark) * 38 / 100)))
        func horizontal(_ y: Int, _ left: Int, _ right: Int) -> Double {
            let hits = (left...right).filter { x in (-1...1).contains { pixels[(y+$0)*side+x] <= limit } }
            return Double(hits.count) / Double(right-left+1)
        }
        func vertical(_ x: Int, _ top: Int, _ bottom: Int) -> Double {
            let hits = (top...bottom).filter { y in (-1...1).contains { pixels[y*side+x+$0] <= limit } }
            return Double(hits.count) / Double(bottom-top+1)
        }
        // Fit the four printed edges near the proposed field; a text glyph or weak outline should abstain.
        var best = 0.0
        var interior: (Int, Int, Int, Int)?
        for cy in stride(from: 23, through: 41, by: 3) {
            for cx in stride(from: 23, through: 41, by: 3) {
                for halfW in stride(from: 11, through: 18, by: 2) {
                    for halfH in stride(from: 11, through: 18, by: 2) {
                        let l = cx-halfW, r = cx+halfW, t = cy-halfH, b = cy+halfH
                        let edges = [horizontal(t,l+2,r-2), horizontal(b,l+2,r-2),
                                     vertical(l,t+2,b-2), vertical(r,t+2,b-2)]
                        guard edges.min()! >= 0.56 else { continue }
                        // Xcode 16 컴파일러가 한 식으로는 타입 추론 시간을 넘겨서 두 줄로 나눈다. 계산 결과는 같다.
                        let offset: Int = abs(cx-32) + abs(cy-32) + abs(halfW-15) + abs(halfH-15)
                        let score: Double = edges.reduce(0, +) / 4 - Double(offset) * 0.002
                        if score > best {
                            best = score
                            interior = (l+4,r-4,t+4,b-4)
                        }
                    }
                }
            }
        }
        guard best >= 0.75, let (l,r,t,b) = interior else { return .unclear }
        var ink = 0
        for y in t...b {
            for x in l...r where pixels[y*side+x] <= limit { ink += 1 }
        }
        let fraction = Double(ink) / Double((r-l+1)*(b-t+1))
        if fraction >= 0.062 { return .marked }
        if fraction <= 0.015 { return .empty }
        return .unclear
    }
}
