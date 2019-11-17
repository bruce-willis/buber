//
//  DottedLineView.swift
//  Buber
//
//  Created by Alexey Salangin on 11/16/19.
//  Copyright © 2019 Alexey Salangin. All rights reserved.
//

import UIKit

class DottedLineView: UIView {
    var lineColor: UIColor = UIColor.black
    var lineWidth: CGFloat = CGFloat(4)
    var round: Bool = false
    var horizontal: Bool = true

    override init(frame: CGRect) {
        super.init(frame: frame)
        initBackgroundColor()
    }

    required init?(coder aDecoder: NSCoder) {
        super.init(coder: aDecoder)
        initBackgroundColor()
    }

    override public func prepareForInterfaceBuilder() {
        initBackgroundColor()
    }

    override func draw(_ rect: CGRect) {

        let path = UIBezierPath()
        path.lineWidth = lineWidth

        if round {
            configureRoundPath(path: path, rect: rect)
        } else {
            configurePath(path: path, rect: rect)
        }

        lineColor.setStroke()

        path.stroke()
    }

    func initBackgroundColor() {
        if backgroundColor == nil {
            backgroundColor = UIColor.clear
        }
    }

    private func configurePath(path: UIBezierPath, rect: CGRect) {
        if horizontal {
            let center = rect.height * 0.5
            let drawWidth = rect.size.width - (rect.size.width.truncatingRemainder(dividingBy: (lineWidth * 2))) + lineWidth
            let startPositionX = (rect.size.width - drawWidth) * 0.5 + lineWidth

            path.move(to: CGPoint(x: startPositionX, y: center))
            path.addLine(to: CGPoint(x: drawWidth, y: center))

        } else {
            let center = rect.width * 0.5
            let drawHeight = rect.size.height - (rect.size.height.truncatingRemainder(dividingBy: (lineWidth * 2))) + lineWidth
            let startPositionY = (rect.size.height - drawHeight) * 0.5 + lineWidth

            path.move(to: CGPoint(x: center, y: startPositionY))
            path.addLine(to: CGPoint(x: center, y: drawHeight))
        }

        let dashes: [CGFloat] = [lineWidth, lineWidth]
        path.setLineDash(dashes, count: dashes.count, phase: 0)
        path.lineCapStyle = CGLineCap.butt
    }

    private func configureRoundPath(path: UIBezierPath, rect: CGRect) {
        if horizontal {
            let center = rect.height * 0.5
            let drawWidth = rect.size.width - (rect.size.width.truncatingRemainder(dividingBy: (lineWidth * 2)))
            let startPositionX = (rect.size.width - drawWidth) * 0.5 + lineWidth

            path.move(to: CGPoint(x: startPositionX, y: center))
            path.addLine(to: CGPoint(x: drawWidth, y: center))

        } else {
            let center = rect.width * 0.5
            let drawHeight = rect.size.height - (rect.size.height.truncatingRemainder(dividingBy: (lineWidth * 2)))
            let startPositionY = (rect.size.height - drawHeight) * 0.5 + lineWidth

            path.move(to: CGPoint(x: center, y: startPositionY))
            path.addLine(to: CGPoint(x: center, y: drawHeight))
        }

        let dashes: [CGFloat] = [0, lineWidth * 2]
        path.setLineDash(dashes, count: dashes.count, phase: 0)
        path.lineCapStyle = CGLineCap.round
    }

}
