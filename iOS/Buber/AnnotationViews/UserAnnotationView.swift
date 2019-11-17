//
//  UserAnnotationView.swift
//  Buber
//
//  Created by Alexey Salangin on 11/16/19.
//  Copyright © 2019 Alexey Salangin. All rights reserved.
//

import MapKit
import Pulsator

class UserAnnotationView: MKAnnotationView {
    override init(annotation: MKAnnotation?, reuseIdentifier: String?) {
        super.init(annotation: annotation, reuseIdentifier: reuseIdentifier)

        self.addHalo()
    }

    @available(*, unavailable)
    required init?(coder aDecoder: NSCoder) {
        fatalError("init(coder:) has not been implemented")
    }

    private func addHalo() {
        let pulsator = Pulsator()
        pulsator.position = self.center
        pulsator.numPulse = 5
        pulsator.radius = 30
        pulsator.animationDuration = 3
        pulsator.backgroundColor = UIColor(red: 0, green: 0.455, blue: 0.756, alpha: 1).cgColor
        self.layer.addSublayer(pulsator)
        pulsator.start()
    }
}
