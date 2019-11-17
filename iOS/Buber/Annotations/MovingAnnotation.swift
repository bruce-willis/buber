//
//  MovingAnnotation.swift
//  Buber
//
//  Created by Alexey Salangin on 11/15/19.
//  Copyright © 2019 Alexey Salangin. All rights reserved.
//

import MapKit
import UIKit

class MovingAnnotation: NSObject, MKAnnotation {
    dynamic var coordinate: CLLocationCoordinate2D
    dynamic var title: String?
    dynamic var subtitle: String?
    dynamic var image: UIImage?
    private var route: [CLLocationCoordinate2D]
    private let velocity = 3.0

    init(route: [CLLocationCoordinate2D]) {
        self.coordinate = route[0]
        self.route = route
        super.init()
    }

    private var currentIndex = 0

    private var count: Int { return self.route.count }

    func start() {
        func move() {
            let currentCoordinate = self.coordinate
            let nextCoordinate = self.route[(self.currentIndex + 1) % self.count]
            let distance = currentCoordinate.distance(to: nextCoordinate)
            var duration = distance / self.velocity

            if duration > 0.3 {
                let newDuration = 0.1
                let coef = newDuration / duration
                duration = newDuration
                let dest = self.interpolate(start: currentCoordinate, end: nextCoordinate, value: coef)
                UIView.animate(
                    withDuration: duration,
                    delay: 0.0,
                    options: [.curveLinear],
                    animations: {
                        self.coordinate = dest
                    },
                    completion: { _ in
                        move()
                    }
                )
            } else {
                UIView.animate(
                    withDuration: duration,
                    delay: 0.0,
                    options: [.curveLinear],
                    animations: {
                        self.currentIndex += 1
                        self.coordinate = self.route[self.currentIndex % self.count]
                    },
                    completion: { _ in
                        move()
                    }
                )
            }
        }
        move()
    }

    private func interpolate(start: CLLocationCoordinate2D, end: CLLocationCoordinate2D, value: Double) -> CLLocationCoordinate2D {
        return CLLocationCoordinate2D(
            latitude: start.latitude + (end.latitude - start.latitude) * value,
            longitude: start.longitude + (end.longitude - start.longitude) * value
        )
    }
}
