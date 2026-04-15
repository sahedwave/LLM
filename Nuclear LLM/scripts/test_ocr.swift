import Foundation
import PDFKit
import Vision
import AppKit

let arguments = CommandLine.arguments
guard arguments.count >= 3 else {
    fputs("Usage: swift test_ocr.swift <pdf-path> <page-index>\n", stderr)
    exit(1)
}

let pdfPath = arguments[1]
let pageIndex = Int(arguments[2]) ?? 0

func renderPage(_ page: PDFPage, scale: CGFloat = 1.0) -> CGImage? {
    let bounds = page.bounds(for: .mediaBox)
    let width = Int(bounds.width * scale)
    let height = Int(bounds.height * scale)
    guard let colorSpace = CGColorSpace(name: CGColorSpace.sRGB),
          let context = CGContext(
              data: nil,
              width: width,
              height: height,
              bitsPerComponent: 8,
              bytesPerRow: 0,
              space: colorSpace,
              bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue
          )
    else {
        return nil
    }

    context.setFillColor(NSColor.white.cgColor)
    context.fill(CGRect(x: 0, y: 0, width: width, height: height))
    context.saveGState()
    context.scaleBy(x: scale, y: scale)
    context.concatenate(page.transform(for: .mediaBox))
    page.draw(with: .mediaBox, to: context)
    context.restoreGState()
    return context.makeImage()
}

func ocrLines(from image: CGImage) throws -> [String] {
    let request = VNRecognizeTextRequest()
    request.recognitionLevel = .fast
    request.usesLanguageCorrection = false

    let handler = VNImageRequestHandler(cgImage: image, options: [:])
    try handler.perform([request])

    let observations = request.results ?? []
    let sorted = observations.sorted { left, right in
        let a = left.boundingBox
        let b = right.boundingBox
        if abs(a.minY - b.minY) > 0.01 {
            return a.minY > b.minY
        }
        return a.minX < b.minX
    }

    return sorted.compactMap { observation in
        observation.topCandidates(1).first?.string.trimmingCharacters(in: .whitespacesAndNewlines)
    }.filter { !$0.isEmpty }
}

guard let document = PDFDocument(url: URL(fileURLWithPath: pdfPath)),
      let page = document.page(at: pageIndex),
      let image = renderPage(page)
else {
    fputs("Unable to open or render PDF page.\n", stderr)
    exit(1)
}

do {
    try ocrLines(from: image).forEach { print($0) }
} catch {
    fputs("OCR failed: \(error)\n", stderr)
    exit(1)
}
