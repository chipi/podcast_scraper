import Foundation
import Vision
import AppKit

// For each image: pixel size, the LARGEST face's box, and the eye centre — top-left origin, 0..1.
let dir = CommandLine.arguments[1]
let files = try FileManager.default.contentsOfDirectory(atPath: dir).filter { $0.hasSuffix(".jpg") }.sorted()
for f in files {
  let url = URL(fileURLWithPath: dir).appendingPathComponent(f)
  guard let img = NSImage(contentsOf: url),
        let cg = img.cgImage(forProposedRect: nil, context: nil, hints: nil) else { print("\(f)\tERR"); continue }
  let req = VNDetectFaceLandmarksRequest()
  try? VNImageRequestHandler(cgImage: cg, options: [:]).perform([req])
  let faces = (req.results ?? []).sorted { $0.boundingBox.width * $0.boundingBox.height > $1.boundingBox.width * $1.boundingBox.height }
  guard let face = faces.first else { print("\(f)\t\(cg.width)\t\(cg.height)\tNOFACE"); continue }
  let b = face.boundingBox // normalised, origin bottom-left
  var eyeY = b.origin.y + b.height * 0.62
  if let l = face.landmarks?.leftEye, let r = face.landmarks?.rightEye {
    let pts = (l.normalizedPoints + r.normalizedPoints)
    let avg = pts.map { $0.y }.reduce(0, +) / CGFloat(pts.count)
    eyeY = b.origin.y + avg * b.height
  }
  // Convert to top-left origin.
  let top = 1 - (b.origin.y + b.height), bottom = 1 - b.origin.y
  let left = b.origin.x, right = b.origin.x + b.width
  print(String(format: "%@\t%d\t%d\t%.4f\t%.4f\t%.4f\t%.4f\t%.4f\t%d", f, cg.width, cg.height, left, top, right, bottom, 1 - eyeY, faces.count))
}
