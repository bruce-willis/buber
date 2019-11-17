//
//  CLLocationCoordinate2D++.swift
//  Buber
//
//  Created by Alexey Salangin on 11/15/19.
//  Copyright © 2019 Alexey Salangin. All rights reserved.
//

import CoreLocation

extension CLLocationCoordinate2D {
    func distance(to: CLLocationCoordinate2D) -> CLLocationDistance {
        let source = CLLocation(latitude: self.latitude, longitude: self.longitude)
        let destination = CLLocation(latitude: to.latitude, longitude: to.longitude)
        return destination.distance(from: source)
    }
}
