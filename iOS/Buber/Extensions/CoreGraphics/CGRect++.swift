//
//  CGRect++.swift
//  Buber
//
//  Created by Alexey Salangin on 11/16/19.
//  Copyright © 2019 Alexey Salangin. All rights reserved.
//

import CoreGraphics

extension CGRect {
    var center: CGPoint {
        get { return CGPoint(x: centerX, y: centerY) }
        set { centerX = newValue.x; centerY = newValue.y }
    }

    var centerX: CGFloat {
        get { return midX }
        set { origin.x = newValue - width * 0.5 }
    }

    var centerY: CGFloat {
        get { return midY }
        set { origin.y = newValue - height * 0.5 }
    }
}
