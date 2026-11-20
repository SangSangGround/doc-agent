import SwiftUI

enum DocumentStyle {
    static let canvas = color(0xF7F6FB, 0x17151E)
    static let card = color(0xFFFFFF, 0x242130)
    static let ink = color(0x252234, 0xFAF8FF)
    static let muted = color(0x5C566C, 0xD2CBDF)
    static let accent = color(0x6343AF, 0xD0B9FF)
    static let lavender = color(0xE9DFFE, 0x423255)
    static let lilac = color(0xF8EEF7, 0x392D42)
    static let mint = color(0xE1F4EA, 0x253B33)
    static let pine = color(0x276D55, 0x9FDFC1)
    static let peach = color(0xFCE9DF, 0x412E29)
    static let clay = color(0x98512D, 0xF4BC9A)
    static let border = color(0xE8E3EF, 0x484152)
    static let action = LinearGradient(colors: [Color(red: 0.43, green: 0.30, blue: 0.75),
                                               Color(red: 0.34, green: 0.21, blue: 0.65)],
                                       startPoint: .topLeading, endPoint: .bottomTrailing)
    static let wash = LinearGradient(colors: [lavender, lilac], startPoint: .topLeading, endPoint: .bottomTrailing)

    private static func color(_ light: Int, _ dark: Int) -> Color {
        Color(uiColor: UIColor { traits in
            let rgb = traits.userInterfaceStyle == .dark ? dark : light
            return UIColor(red: CGFloat((rgb >> 16) & 255) / 255, green: CGFloat((rgb >> 8) & 255) / 255,
                           blue: CGFloat(rgb & 255) / 255, alpha: 1)
        })
    }
}

extension View {
    func documentCard() -> some View {
        background(DocumentStyle.card, in: RoundedRectangle(cornerRadius: 21))
            .overlay { RoundedRectangle(cornerRadius: 21).strokeBorder(DocumentStyle.border, lineWidth: 1).allowsHitTesting(false) }
    }
}

struct DocumentButtonStyle: ButtonStyle {
    @Environment(\.isEnabled) private var enabled
    var prominent = true
    func makeBody(configuration: Configuration) -> some View {
        configuration.label.font(.headline).multilineTextAlignment(.center)
            .padding(.horizontal, 14).padding(.vertical, 11)
            .frame(maxWidth: .infinity, minHeight: 52)
            .foregroundStyle(enabled && prominent ? Color.white : enabled ? DocumentStyle.accent : DocumentStyle.muted)
            .background {
                if enabled && prominent { RoundedRectangle(cornerRadius: 14).fill(DocumentStyle.action) }
                else { RoundedRectangle(cornerRadius: 14).fill(enabled ? DocumentStyle.lavender : DocumentStyle.canvas) }
            }
            .overlay { if !enabled { RoundedRectangle(cornerRadius: 14).strokeBorder(DocumentStyle.border) } }
            .opacity(configuration.isPressed ? 0.78 : 1)
    }
}

struct DocumentTile: View {
    let symbol: String
    let tint: Color
    let fill: Color
    var body: some View {
        Image(systemName: symbol).font(.system(size: 22, weight: .semibold))
            .foregroundStyle(tint).frame(width: 48, height: 48)
            .background(fill, in: RoundedRectangle(cornerRadius: 15)).accessibilityHidden(true)
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

/// Layered sheets and a sound wave echo the document reader's two primary actions.
struct DocumentIllustration: View {
    var body: some View {
        ZStack {
            Ellipse().fill(.purple.opacity(0.10)).frame(width: 94, height: 13).offset(y: 39)
            RoundedRectangle(cornerRadius: 9).fill(LinearGradient(colors: [.init(red: 0.68, green: 0.56, blue: 0.92),
                                                                           .init(red: 0.47, green: 0.34, blue: 0.77)],
                                                                   startPoint: .topLeading, endPoint: .bottomTrailing))
                .frame(width: 52, height: 64).rotationEffect(.degrees(-16)).offset(x: -12, y: 1)
            ZStack(alignment: .leading) {
                RoundedRectangle(cornerRadius: 8).fill(Color(red: 1, green: 0.98, blue: 1))
                VStack(alignment: .leading, spacing: 6) {
                    Capsule().fill(Color(red: 0.55, green: 0.40, blue: 0.84)).frame(width: 21, height: 5)
                    Capsule().fill(Color(red: 0.80, green: 0.73, blue: 0.92)).frame(width: 31, height: 3)
                    Capsule().fill(Color(red: 0.80, green: 0.73, blue: 0.92)).frame(width: 25, height: 3)
                }.padding(.leading, 9)
            }
            .frame(width: 53, height: 65).rotationEffect(.degrees(10)).offset(x: 7, y: -5)
            ZStack {
                RoundedRectangle(cornerRadius: 11).fill(LinearGradient(colors: [.init(red: 1, green: 0.78, blue: 0.77),
                                                                              .init(red: 0.89, green: 0.57, blue: 0.72)],
                                                                        startPoint: .topLeading, endPoint: .bottomTrailing))
                Image(systemName: "waveform").font(.system(size: 18, weight: .bold)).foregroundStyle(.white)
            }.frame(width: 35, height: 29).rotationEffect(.degrees(-7)).offset(x: 32, y: 19)
        }
        .frame(width: 108, height: 96).accessibilityHidden(true)
    }
}
