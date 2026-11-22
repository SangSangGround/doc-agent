import SwiftUI

enum DocumentStyle {
    static let canvas = color(0xFFF9EE, dark: 0x211A17)
    static let card = color(0xFFFFFF, dark: 0x302623)
    static let raised = color(0xFFEFCA, dark: 0x473722)
    static let ink = color(0x2C1E19, dark: 0xFFF7EA)
    static let muted = color(0x665348, dark: 0xD9C7B7)
    static let accent = color(0xA62C20, dark: 0xFFB896)
    static let accentFill = color(0xFFE4D2, dark: 0x52352B)
    static let caution = color(0x854200, dark: 0xFFCA88)
    static let border = color(0xDCBD9D, dark: 0x665446)
    static let darkInk = color(0x211B19)
    static let gold = color(0xFFC83D)
    static let orange = color(0xFC7B30)
    static let header = LinearGradient(colors: [color(0xFFDA68), gold], startPoint: .topLeading, endPoint: .bottomTrailing)
    static let homeWash = LinearGradient(colors: [color(0xFFE16B, dark: 0x4B3822), color(0xFFF0C7, dark: 0x352820), canvas],
                                         startPoint: .top, endPoint: .bottom)
    static let hero = LinearGradient(colors: [color(0x981E27), color(0xB53720), color(0xAA3A15)],
                                    startPoint: .topLeading, endPoint: .bottomTrailing)
    static let action = LinearGradient(colors: [color(0xA62320), color(0xAB381B)],
                                       startPoint: .leading, endPoint: .trailing)
    private static func color(_ light: Int, dark: Int) -> Color {
        Color(uiColor: UIColor { traits in
            let hex = traits.userInterfaceStyle == .dark ? dark : light
            return UIColor(red: CGFloat((hex >> 16) & 255) / 255, green: CGFloat((hex >> 8) & 255) / 255,
                           blue: CGFloat(hex & 255) / 255, alpha: 1)
        })
    }

    static func color(_ hex: Int) -> Color {
        Color(red: Double((hex >> 16) & 255) / 255, green: Double((hex >> 8) & 255) / 255,
              blue: Double(hex & 255) / 255)
    }
}

extension View {
    func documentCard() -> some View {
        background(DocumentStyle.card, in: RoundedRectangle(cornerRadius: 20, style: .continuous))
            .overlay {
                RoundedRectangle(cornerRadius: 20, style: .continuous)
                    .strokeBorder(DocumentStyle.border, lineWidth: 1).allowsHitTesting(false)
            }
    }
}

struct DocumentButtonStyle: ButtonStyle {
    @Environment(\.isEnabled) private var enabled
    var prominent = true
    var whiteSurface = false

    func makeBody(configuration: Configuration) -> some View {
        configuration.label.font(.headline).multilineTextAlignment(.center)
            .padding(.horizontal, 14).padding(.vertical, 12)
            .frame(maxWidth: .infinity, minHeight: 52)
            .foregroundStyle(enabled && prominent ? .white : enabled ? DocumentStyle.accent : DocumentStyle.muted)
            .background {
                if enabled && prominent { RoundedRectangle(cornerRadius: 14).fill(DocumentStyle.action) }
                else { RoundedRectangle(cornerRadius: 14).fill(enabled && !whiteSurface ? DocumentStyle.accentFill : DocumentStyle.card) }
            }
            .overlay {
                RoundedRectangle(cornerRadius: 14)
                    .strokeBorder(enabled && prominent ? .white.opacity(0.4) : DocumentStyle.border)
                    .allowsHitTesting(false)
            }
            .opacity(configuration.isPressed ? 0.84 : 1)
    }
}

struct DocumentTile: View {
    let symbol: String
    let tint: Color
    let fill: Color

    var body: some View {
        Image(systemName: symbol).font(.title3.weight(.semibold))
            .foregroundStyle(tint).frame(width: 48, height: 48)
            .background(fill, in: RoundedRectangle(cornerRadius: 14))
            .accessibilityHidden(true)
    }
}

struct DocumentRow: View {
    let title: String
    let detail: String
    let symbol: String
    let tint: Color
    let fill: Color
    var accessory = "chevron.right"

    var body: some View {
        HStack(spacing: 14) {
            DocumentTile(symbol: symbol, tint: tint, fill: fill)
            VStack(alignment: .leading, spacing: 4) {
                Text(title).font(.headline).foregroundStyle(DocumentStyle.ink)
                Text(detail).font(.subheadline).foregroundStyle(DocumentStyle.muted).fixedSize(horizontal: false, vertical: true)
            }
            Spacer(minLength: 0)
            Image(systemName: accessory).font(.subheadline.weight(.semibold)).foregroundStyle(DocumentStyle.accent)
                .accessibilityHidden(true)
        }
        .multilineTextAlignment(.leading).padding(16).documentCard().contentShape(Rectangle())
    }
}

struct DocumentIllustration: View {
    var body: some View {
        ZStack {
            Circle().fill(DocumentStyle.gold).frame(width: 16, height: 16).offset(x: -53, y: -17)
            Circle().fill(DocumentStyle.orange).frame(width: 10, height: 10).offset(x: 51, y: -33)
            RoundedRectangle(cornerRadius: 12).fill(DocumentStyle.accent)
                .frame(width: 64, height: 77).rotationEffect(.degrees(-13)).offset(x: -6, y: 4)
            VStack(alignment: .leading, spacing: 6) {
                ForEach([25, 37, 32, 21], id: \.self) { width in
                    Capsule().fill(DocumentStyle.accent).frame(width: CGFloat(width), height: 3)
                }
            }
            .frame(width: 64, height: 77)
            .background(.white, in: RoundedRectangle(cornerRadius: 10))
            .rotationEffect(.degrees(8))
            Image(systemName: "waveform").font(.headline.bold()).foregroundStyle(DocumentStyle.darkInk)
                .frame(width: 37, height: 37).background(DocumentStyle.gold, in: Circle()).offset(x: 37, y: 25)
        }
        .frame(width: 124, height: 98).accessibilityHidden(true)
    }
}

struct ReadingWave: View {
    var tint = DocumentStyle.gold
    private let heights: [CGFloat] = [7, 13, 20, 10, 24, 33, 21, 38, 29, 17, 25, 34, 16, 10, 18, 12, 6]
    var body: some View {
        HStack(alignment: .center, spacing: 3) {
            ForEach(heights.indices, id: \.self) { index in
                Capsule().fill(tint).frame(width: 3, height: heights[index])
            }
        }.accessibilityHidden(true)
    }
}
