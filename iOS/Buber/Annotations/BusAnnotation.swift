//
//  BusAnnotationView.swift
//  Buber
//
//  Created by Alexey Salangin on 11/16/19.
//  Copyright © 2019 Alexey Salangin. All rights reserved.
//

import MapKit

final class BusAnnotation: MovingAnnotation {
    init(route: [CLLocationCoordinate2D], relativePosition: Double) {
        let count = route.count
        let offset = Int(Double(count) * relativePosition)
        let newRoute = route.shifted(withDistance: offset)
        super.init(route: newRoute)
    }
}
