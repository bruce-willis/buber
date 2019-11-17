//
//  AppDelegate.swift
//  Buber
//
//  Created by Alexey Salangin on 11/15/19.
//  Copyright © 2019 Alexey Salangin. All rights reserved.
//

import UIKit

@UIApplicationMain
class AppDelegate: UIResponder, UIApplicationDelegate {
    var window: UIWindow?

    func application(_ application: UIApplication, didFinishLaunchingWithOptions launchOptions: [UIApplication.LaunchOptionsKey: Any]?) -> Bool {
        self.window = UIWindow()
        self.window?.rootViewController = ViewController()
        self.window?.makeKeyAndVisible()

        return true
    }
}

