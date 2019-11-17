//
//  Array++.swift
//  Buber
//
//  Created by Alexey Salangin on 11/17/19.
//  Copyright © 2019 Alexey Salangin. All rights reserved.
//

import Foundation

extension Array {
    func shifted(withDistance distance: Int = 1) -> Array<Element> {
        let offsetIndex = distance >= 0 ?
            self.index(startIndex, offsetBy: distance, limitedBy: endIndex) :
            self.index(endIndex, offsetBy: distance, limitedBy: startIndex)

        guard let index = offsetIndex else { return self }
        return Array(self[index ..< endIndex] + self[startIndex ..< index])
    }

    mutating func shiftInPlace(withDistance distance: Int = 1) {
        self = shifted(withDistance: distance)
    }

}
